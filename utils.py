import re
import string
import faiss
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from typing import Union
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

sigmoid = nn.Sigmoid()
softmax= nn.Softmax(dim = -1)
criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()

class Probe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=4096)
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = self.layer_norm(x)
        return self.linear(x)
    
class ImprovedProbe(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.layer_norm_input = nn.LayerNorm(normalized_shape=input_size)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)

        self.layer_norm1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, x):
        x = self.layer_norm_input(x)

        x = self.fc1(x)
        x = self.silu(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.silu(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)
        return self.fc3(x)
    
class CustomDataset(Dataset):
    def __init__(self, df, model, df_column, args):
        self.df = df
        self.model = model
        self.pad_id = model.tokenizer.pad_token_id
        self.args = args
        # self.max_length = max_length
        self.df_column = df_column
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        tokens1=self.model.to_tokens(self.df[f'{self.df_column}'][index]).squeeze().to('cpu')

        if self.args.is_cot:
            tokens2 = self.model.to_tokens(f"{self.df[f'{self.df_column}'][index]}"+'\n'+f"{self.df['pred'][index]}").squeeze().to('cpu')
            max_length = 4096
        else:
            tokens2 = self.model.to_tokens(f"{self.df[f'{self.df_column}'][index]+ ' '+self.df['pred'][index]}").squeeze().to('cpu')
            max_length = 2048
    
        tp = torch.tensor([self.pad_id]) 
        tensor_padding = tp.repeat((max_length - tokens2.shape[0]))
        return_tokens = torch.cat((tensor_padding, tokens2))
        
        # import pdb;pdb.set_trace()
        
        pred_len = tokens2.shape[-1] - tokens1.shape[-1]        
        acc = self.df['acc'][index]      
    
        return {
            'input_tokens': return_tokens,
            'label': acc,
            'pred_len': pred_len
        }

class StopOnPunctuationWithLogit(StoppingCriteria):
    def __init__(self, tokenizer, confidence_threshold=0.4, stop_tokens=[".", "?", "!"], is_q2q=False):
        self.tokenizer = tokenizer
        self.stop_token_ids = tokenizer.convert_tokens_to_ids(stop_tokens)
        self.confidence_threshold = confidence_threshold
        self.confidence_log = 1
        self.is_q2q = is_q2q

    def __call__(self, input_ids, scores, **kwargs):
        if self.is_q2q:
            if input_ids[0, -1] in self.stop_token_ids:
                return True
            return False
        else:
            logits = scores[-1]  
            probabilities = torch.softmax(logits, dim=-1)  
            max_confidence = torch.max(probabilities).item()  

            if max_confidence < self.confidence_log:
                self.confidence_log = max_confidence

            if input_ids[0, -1] in self.stop_token_ids and self.confidence_log <= self.confidence_threshold:
                return True
            return False
        
def make_loss(model: nn.Module, input_tensor: torch.Tensor, labels: torch.Tensor) \
    -> Union[torch.Tensor, torch.Tensor]:
    output = model(input_tensor)
    if output.shape[-1] == 1:
        logit = sigmoid(output).squeeze()
        labels=labels.type(torch.float32)
        loss = criterion_bce(logit, labels)
    else:
        logit = softmax(output)
        loss = criterion_ce(logit, labels)
    return loss, logit

def _input_tensor_method1(cache_activation, labels, pred_lens, args):
    result = []
    new_labels = torch.repeat_interleave(labels, pred_lens).to(args.device)
    for i in range(cache_activation.size(0)):
            
        sliced_tensor = cache_activation[i, -pred_lens[i]:, :]
        result.append(sliced_tensor)

        input_tensor = torch.cat(result, dim=0)
    return input_tensor, new_labels

def optim_scheduler_loss(loss, optim, scheduler):
    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()

def return_acc(logit, labels):
    labels = labels.to('cpu')
    logit = logit.to('cpu')
    
    if logit.shape[-1] == 2:
        correct_predictions = (torch.argmax(logit, dim = -1) == labels).sum().item()
    else:
        correct_predictions = ((logit>0.5).int() == labels).sum().item()
        
    total_samples = labels.size(0)
    accuracy = correct_predictions / total_samples   
    return accuracy

def method_1_train(model, optim, scheduler, activations, labels, pred_lens, args):
    optim.zero_grad()
    input_tensor, new_labels = _input_tensor_method1(activations, labels, pred_lens, args)
    loss, _ = make_loss(model = model, input_tensor=input_tensor, labels=new_labels)
    # optim_scheduler_loss(loss, optim, scheduler)
    loss.backward()
    
    optim.step()
    scheduler.step()
    return round(loss.item(),4), optim.param_groups[0]['lr']

def method_1_eval(model, activations, labels, pred_lens, args):
    input_tensor, new_labels = _input_tensor_method1(activations, labels, pred_lens, args)
    loss, logit = make_loss(model = model, input_tensor=input_tensor, labels=new_labels)
    accuracy = return_acc(logit, new_labels)
    return round(accuracy, 4), len(labels), loss

def _method_2_util(model, activations, labels, pred_lens, args):
    input_tensor, new_labels = _input_tensor_method1(activations, labels, pred_lens, args)
    
    result = torch.split(input_tensor, pred_lens.tolist())
    averages = [torch.mean(t, dim=0, keepdim=True) for t in result]
    logit = torch.cat(averages, dim=0)
    
    loss, logit = make_loss(model, logit, labels.to(args.device))
    return loss, logit

def method_2_train(model, optim, scheduler, activations, labels, pred_lens, args):
    optim.zero_grad()
    loss, logit = _method_2_util(model, activations, labels, pred_lens, args)
    # optim_scheduler_loss(loss, optim, scheduler)
    loss.backward()
    optim.step()
    scheduler.step()
    return round(loss.item(),4), optim.param_groups[0]['lr']

def method_2_eval(model, activations, labels, pred_lens, args):
    loss, logit = _method_2_util(model, activations, labels, pred_lens, args)
    accuracy = return_acc(logit, labels)
    return round(accuracy, 4), len(labels), loss

def _method_3_util(model, activations, labels, pred_lens, args):
    input_ids = activations[:,-1,:].squeeze()
    logit=model(input_ids)
    logit = softmax(logit)
    loss = criterion_ce(logit, labels.to(args.device))

    return loss, logit

def method_3_train(model, optim, scheduler, activations, labels, pred_lens, args):
    optim.zero_grad()
    loss, _ = _method_3_util(model, activations, labels, pred_lens, args)
    # optim_scheduler_loss(loss, optim, scheduler)
    loss.backward()
    optim.step()
    scheduler.step()
    return round(loss.item(), 4), optim.param_groups[0]['lr']

def method_3_eval(model, activations, labels, pred_lens, args):
    loss, logit = _method_3_util(model, activations, labels, pred_lens, args)
    accuracy=return_acc(logit, labels)
    
    return round(accuracy, 4), len(labels), loss 

def nan_pred_answer_drop(df):
    error_nums = []
    for num, i in enumerate(df['pred']):
        if isinstance(i, float):
            error_nums.append(num)

    df=df.drop(error_nums, axis =0).reset_index()
    return df

def len_drop(df):
    df['len'] = df['question_with_prompt'].apply(lambda x: len(x))
    nums = []
    for num, i in enumerate(df['len']):
        if i > 800:
            nums.append(num)
    
    df=df.drop(nums, axis =0).reset_index()
    
    df = df.drop(['level_0', 'len'], axis = 1)
    return df

def len_drop_pred(df):
    df['len'] = df['pred'].apply(lambda x: len(x))
    nums = []
    for num, i in enumerate(df['len']):
        if i > 30:
            nums.append(num)
    
    df=df.drop(nums, axis =0).reset_index() 
    return df

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text

def acc_checking(df):
    accs = []
    for i in tqdm(range(len(df))):
        count = 0
        for j in df['answer'][i]:
            if preprocess_text(j) in preprocess_text(df['pred'][i]):
                count += 1
            else: pass
        if count == 0:
            accs.append(0)
        else:
            accs.append(1)
    return pd.DataFrame(accs)

def split_A(q):
    return q.split('\n')[2].replace('A','').replace(':','').strip()

class Config_Maker():
    def __init__(self, model, method, layer, position, device):
        self.method = method
        self.layer = layer
        self.position = position
        self.device = device
        self.d_model = model.cfg.d_model
        self.model_id = model.cfg.tokenizer_name
        self.num_classes = 2
def load_prober(_ds, cfg):
    '''
    mistral
    method: tokens_mean
    layer: 12, 14, 16, 18, 20, 22
    module: attn_out, mlp_out, resid_mid, resid_post
    gemma-2b
    method: tokens_mean
    layer: 6, 8, 10, 12, 14, 16
    module: resid_mid, resid_post
    '''
    prober=ImprovedProbe(input_size=cfg.d_model, output_size=cfg.num_classes).to(cfg.device)
    if 'mistralai/Mistral-7B-Instruct-v0.1' == cfg.model_id:
        prober.load_state_dict(torch.load(f'ckpt/probing_ckpt/Mistral-7B-Instruct-v0.1_{cfg.method}_probe_2_l{cfg.layer}_{cfg.position}_1.pt'))
    elif 'google/gemma-2b' == cfg.model_id:
        # prober.load_state_dict(torch.load(f'ckpt/prob_model_cot_v2/gemma-2b_v2_linear9995_{cfg.method}_2_l{cfg.layer}_{cfg.position}_1.pt')) # v2 
        if _ds == 25:
            prober.load_state_dict(torch.load(f'ckpt/_25/0.25_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep1.pt')) # v1
        elif _ds == 50:
            prober.load_state_dict(torch.load(f'ckpt/_5/0.5_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep1.pt')) # v1
        elif _ds == 75:
            prober.load_state_dict(torch.load(f'ckpt/_75/0.75_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep1.pt')) # v1
        elif _ds == 777:
            prober.load_state_dict(torch.load(f'ckpt/_75_full/0.75_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep.pt')) # v1
        elif _ds == 3:
            prober.load_state_dict(torch.load(f'ckpt/_3/in3_1.0_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep1.pt')) # v1
        elif _ds == 333:
            prober.load_state_dict(torch.load(f'ckpt/_3_3/in3_0.33_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep.pt')) # v1
        elif _ds == 366:
            prober.load_state_dict(torch.load(f'ckpt/_3_6/in3_0.66_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep.pt')) # v1
        elif _ds == 3000:
            prober.load_state_dict(torch.load(f'ckpt/_3_1000/in3_1000_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep11.pt')) # v1
        elif _ds == 1000:
            prober.load_state_dict(torch.load(f'ckpt/_1000/1000_gemma-2b_{cfg.method}_2_l{cfg.layer}_{cfg.position}_ep11.pt')) # v1
        else:
            prober.load_state_dict(torch.load(f'ckpt/prob_model_cot_v1/gemma-2b_linear995_{cfg.method}_probe_2_l{cfg.layer}_{cfg.position}_1.pt')) # v1
    
    else: assert 'model_id 랑 맞는 prober가 존재하지 않음.'
    prober.eval()
    return prober

def cache_output(cfg, prober, cache):
    activations_attn_out_12 = cache[cfg.position, cfg.layer]
    logit=prober(activations_attn_out_12)
    return logit

def preprocessing(df, args):
    print('*'*30)
    print(args.dataset_name)
    print('*'*30)
    # import pdb;pdb.set_trace()
    if (args.dataset_name == 'hotpotqa') or (args.dataset_name == '2wikimultihopqa') or (args.dataset_name == 'musique'):
        # import pdb;pdb.set_trace()
        df['answer']=df['answer'].apply(lambda x: x.replace('[','').replace(']','').split("' '"))
        def remove_special_ch(x):
            return [i.replace("'",'') for i in x]
        df['answer']=df['answer'].apply(lambda x: remove_special_ch(x))
        # import pdb;pdb.set_trace()
    else:
        pass
    return df

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def encode_query(model_retr, query):
    return model_retr.encode(query)

# def gen_answer(query):
#     gened=model.generate(**tokenizer(query, return_tensors='pt').to(device),
#                         max_new_tokens = 100,
#                         pad_token_id=tokenizer.eos_token_id)
#     return tokenizer.decode(gened[0], skip_special_tokens=True)

def find_topk_sim(model_retr, query: str, index: faiss, k: int):
    D, I = index.search(np.array(torch.tensor(encode_query(model_retr, query)).unsqueeze(0)), k=k)
    return D, I

def batch_topk_sim(model_retr, query: str, index: faiss, k: int):
    D, I = index.search(encode_query(model_retr, query), k=k)
    return D, I

def load_prober_cfg_gemma_2b(model, config, position, device, start, end, step):
    return [config(model, 'tokens_mean', j, position, device) for j in range(start, end, step)]    

def load_prober_models(_ds, cfg_list):
    probers = [load_prober(_ds, cfg) for cfg in cfg_list]
    return probers

def return_prober_logit_gemma_2b(method_function, cfg_list, model_list):
    return [method_function(cfg, model).to('cpu') for cfg, model in zip(cfg_list, model_list)]

def evaluator(df, metric, pred_list,args):
    acc = []
    pred_lists = []
    pred_to_train = []
    if args.is_cot:
        if (args.retr_method == 'dragin') or (args.retr_method == 'fix-length-retrieval') or (args.retr_method == 'fix-sentence'):
            for pred in pred_list:
                if 'answer' in pred.lower():
                    pred_lists.append(''.join(''.join(pred.lower().split('answer')[:1]).split('\n\n')[:1]).replace(':','').replace('</s>','').replace('<eos>','').strip())
                else:
                    pred_lists.append(''.join(pred.split('\n\n')[:1]).replace('</s>','').replace('<eos>','').strip())
            
        else:
            for pred in pred_list:
                pred=pred.split('\n\n')[4] 
                if len(pred.split('\n')) > 7:
                    new_pred = '\n'.join(pred.split('\n')[8:])
                    pred_to_train.append(new_pred)
                else:
                    new_pred = '\n'.join(pred.split('\n')[1:])
                    pred_to_train.append(new_pred)
                
                pred_lists.append(new_pred.replace('</s>','').replace('<eos>','').replace('Answer:','').strip())
        
    else:
        for pred in pred_list:
            
            new_pred=pred.split('\n\n')[2]
            pred_lists.append(new_pred.replace('</s>','').replace('<eos>','').replace('Answer:','').strip())
    
    for num, ans in enumerate(tqdm(df['answer'][:args.steps_limit+1])):
        
        ans2 = [normalize_answer(an) for an in ans] 
        pred_list2 = normalize_answer(pred_lists[num]) 
        try: 
            pred_list3 = normalize_answer(pred_lists[num].split('\n')[1]) 
        except:
            pred_list3 = normalize_answer(pred_lists[num]) 
        
        try:
            if (args.dataset_name == 'hotpotqa') or (args.dataset_name == '2wikimultihopqa') or (args.dataset_name == 'musique') or (args.dataset_name == 'iirc'):
                metric([pred_list3], ans2) 
            else:
                metric(pred_list3, ans2)
        except: continue
        
        answer_is_in =0
        for k in ans2:
            if k in pred_list2:
            
                answer_is_in += 1
        
        if answer_is_in==0:
            acc.append(0)
        else:
            acc.append(1)
    
    print(args.retr_method)
    
    print('acc: ', sum(acc)/len(acc))
    return acc, metric, pred_to_train
######## --------------- dragin
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.special import softmax
import spacy
nlp = spacy.load("en_core_web_sm")
#%%

class BasicGenerator:
    def __init__(self, model_name_or_path):
        # logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda:0", 
                    trust_remote_code = "falcon" in model_name_or_path)
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
class BasicRAG:
    def __init__(self, bm25, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                self.retriever = bm25
                
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            docs = self.retriever.retrieve(query)
            return docs
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
class AttnWeightRAG(BasicRAG):
    def __init__(self, bm25, args):
        super().__init__(bm25, args)
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in 
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])

                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])
        atten_tmp = self.generator.model(input_ids.to(self.generator.model.device), output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 해당 환각 단어를 얻으십시오 attention
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # 초과하는 각 임계값을 계산합니다. token 전술한 내용에서 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 품사를 분석하고 내용 단어에 해당하는 속성을 유지합니다.
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])
        
    def inference(self, question, demo, case):
        text = ""
        while True:
            
            old_len = len(text)
            
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)

            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, 
                self.generate_max_length, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            # import pdb;pdb.set_trace()
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                # import pdb;pdb.set_trace()
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                
                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                
                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                
                elif self.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 
                else:
                    raise NotImplemented

                
                docs = self.retrieve(retrieve_question) # docs list 형태로 나옴
            
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc.text}\n"
                prompt += "Question: "
            
                tmp_li = [case, text, ptext.strip()]
                # import pdb;pdb.set_trace()
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # prompt += " Rationale: "
                # import pdb;pdb.set_trace()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                
            
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
            
            # 토큰 수가 generate_max_length보다 작은지 확인
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
    
class FixLengthRAG(BasicRAG):
    def __init__(self, bm25, args):
        super().__init__(bm25, args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs = self.retrieve(retrieve_question)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                print('fix_length')
                new_text, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
            else:
                print('fix sentence')
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, max_length=self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
                retrieve_question = sentences[0]
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
    
