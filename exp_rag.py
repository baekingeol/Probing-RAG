#%%
import argparse
from argparse import Namespace

import time
import json
from tqdm import tqdm
from functools import partial

import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
from transformers import StoppingCriteria, StoppingCriteriaList

from transformer_lens import HookedTransformer

import torch
from torch.utils.data import DataLoader, Dataset

from metrics.metrcis import EmF1Metric, SupportEmF1Metric 

from utils import AttnWeightRAG, FixLengthRAG, StopOnPunctuationWithLogit, Config_Maker, preprocessing, batch_topk_sim
from utils import load_prober_cfg_gemma_2b, load_prober_models, return_prober_logit_gemma_2b, evaluator
from prompts import inst_prompt, cot_prompt, retr_qa, retr_qa_cot2

def main(args):
    steps_limit =args.steps_limit # 100
    threshold = args.threshold # 0.5
    is_sparse = args.is_sparse # True
    retr_method = args.retr_method  # probing, none, simple
    position = args.position # 'resid_post'
    dataset_name = args.dataset_name
    is_cot = args.is_cot
    model_id = args.model_id
    tr_or_dev = args.tr_or_dev
    _ds = args.ds # 25, 50, 75, 1000, else
    metric = EmF1Metric()
    print('*'*70)
    print(f"threshold: {threshold}, retr_method: {retr_method}, position: {position},\ndataset_name: {dataset_name}, model_id: {model_id}, steps_limit: {steps_limit} \n ablation: {args.ablation}, prober_ds_len: {_ds}")
    print('*'*70)
    #%%
    if is_cot:
        prompt_function_data=cot_prompt
        prompt_function_retr = retr_qa_cot2
        savename_is_cot = 'cot'
        max_new_tokens = 150
        
    if is_sparse:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core import Document
        retr_type = 'sparse'
        print('sparse retrieval loading...')
        docstore2 = SimpleDocumentStore.from_persist_path(f"index/sparse_index/llama_index_bm25_model_{dataset_name}_2.json") #
        bm25=BM25Retriever.from_defaults(docstore=docstore2, similarity_top_k=5)
    else:
        retr_type = 'dense'
        print('dense retrieval loading...')
        model_retr_id = 'facebook/contriever-msmarco'
        model_retr = SentenceTransformer(model_retr_id)
        index = faiss.read_index(f'index/dense_index/contriever_{dataset_name}_2.bin') #
    print('finish!!')
    print('*'*70)
    if (dataset_name =='hotpotqa') and (tr_or_dev=='dev'): path = f'index/hotpotqa/hotpot_{tr_or_dev}_distractor_v1.json'
    elif (dataset_name =='hotpotqa') and (tr_or_dev=='train'): path = f'index/hotpotqa/hotpot_{tr_or_dev}_v1.1.json'
    elif dataset_name =='nq': path = f'index/nq/biencoder-nq-{tr_or_dev}.json'
    elif dataset_name =='trivia': path = f'index/trivia/biencoder-trivia-{tr_or_dev}.json'
    elif dataset_name =='2wikimultihopqa': path = f'index/2wikimultihopqa/{tr_or_dev}.json' # TODO einsum error fix when do model.generate
    elif dataset_name =='musique': path = f'index/musique/musique_full_v1.0_{tr_or_dev}.jsonl'
    elif dataset_name == 'iirc': path = f"index/iirc/{tr_or_dev}.json"
    
    if (args.dataset_name == 'hotpotqa') or (args.dataset_name == '2wikimultihopqa') or (args.dataset_name == 'musique') or (args.dataset_name == 'iirc'):
        metric = SupportEmF1Metric()    
        answer_name = 'answer'
    else: 
        metric = EmF1Metric()
        answer_name = 'answers'
    
    dataset = []
    if dataset_name =='musique':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
    else:
        with open(path) as f:  #문제
            js = json.load(f) 
        if dataset_name == 'iirc':
            for tmp in tqdm(js):
                for example in tmp['questions']:
                    qid = example["qid"]
                    question = example['question']

                    ans = example['answer']

                    if ans['type'] == 'none':
                        continue
                    elif ans['type'] == 'value' or ans['type'] == 'binary':
                        answer = [ans['answer_value']]
                    elif ans['type'] == 'span':
                        answer = [v['text'].strip() for v in ans['answer_spans']]
                    
                    # context = example['context']
                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answer,
                        # 'ctxs': context,
                    })                
        else: dataset = js
            
    corpus = pd.read_csv(f'index/documents/{dataset_name}_index_2.csv') 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = HookedTransformer.from_pretrained(model_id, device = device)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    if retr_method == 'probing':


        if 'google/gemma-2b' == model_id:
            # cfg_list = load_prober_cfg_gemma_2b(model, Config_Maker, position, device, 0,17, 1)
            cfg_list = load_prober_cfg_gemma_2b(model, Config_Maker, position, device, 6,17, 2)
            probers = load_prober_models(_ds, cfg_list)
            layer_configs = cfg_list
            
        cache = {}

        def hook_fn(activations, hook, layer):
            if layer not in cache:
                cache[layer] = []
            cache[layer].append(activations.detach().cpu())
            return activations

        def add_layer_hook(model, layer_name):
            hook = partial(hook_fn, layer=layer_name)
            model.add_hook(layer_name, hook)

        for prober_cfg in layer_configs:
            layer_name = f'blocks.{prober_cfg.layer}.hook_{prober_cfg.position}'
            add_layer_hook(model, layer_name)
        
    model.eval()
        
    if args.extract_sep:
        dataset = dataset[args.sep_number:]
        save_data_name = f'after{args.sep_number}'
        
    questions, answers = [], []
    for value in tqdm(dataset):
        question, answer = value['question'], value[f'{answer_name}']
        questions.append(question)
        answers.append(answer)
    df = pd.DataFrame([questions, answers]).T
    df.columns = ['query', 'answer']

    class CustomDataset(Dataset):
        def __init__(self, dataset, prompt, tokenizer):
            self.dataset = dataset
            self.prompt = prompt
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.dataset)
                    
        def __getitem__(self, index):
            item = self.dataset[index]
            prompt_text = self.prompt(item)
            token = self.tokenizer(prompt_text,return_tensors='pt').to(device)
            return {
                'input_ids': token['input_ids'].squeeze(),
                'attention_mask': token['attention_mask'].squeeze(),
                'prompt_text': prompt_text,
                'text': item,
            }
    df = preprocessing(df, args)
    if (retr_method == 'flare') or (retr_method == 'linguistic'):
        pass
    else:
        dataloader=DataLoader(CustomDataset(df['query'], prompt = prompt_function_data, tokenizer=tokenizer))

    def return_evidences(retrieved_passages, is_sparse = args.is_sparse):
        evidences = ''
        def return_evidence(evidence, is_sparse):
            if is_sparse: return evidence.text
            else: return evidence
        for num, evidence in enumerate(retrieved_passages):
            if (num+1) == len(retrieved_passages):
                evidences+= f'passage {num+1}: {return_evidence(evidence,is_sparse)}'
            else:
                evidences+= f'passage {num+1}: {return_evidence(evidence,is_sparse)}'+'\n'
        return evidences
    
    def return_mean_output(prober_cfg, prober):
        layer_name = f'blocks.{prober_cfg.layer}.hook_{prober_cfg.position}'
        with torch.no_grad():
            # import pdb;pdb.set_trace()
            input=torch.concat(cache[layer_name][1:], dim = 1).to(device)
            input = torch.sum(input, dim = 1)
            logit=prober(input)
        # import pdb;pdb.set_trace()
        return logit

    retr_count_list, pred_list = [], []
    steps = 0
    softmax_f = torch.nn.Softmax(dim = 1)
    if retr_method == 'probing':
        start = time.time()
        for value in tqdm(dataloader):
            cache={}
            
            retr_count = 0
            with torch.no_grad():
                output = model.generate(value['input_ids'], do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
                if steps % 10 == 0:
                    print(model.to_string(output)[0])
                
            if 'google/gemma-2b' == model_id:
                logits = return_prober_logit_gemma_2b(return_mean_output, cfg_list, probers)
                for_set_threshold = torch.zeros_like(logits[0].squeeze())
                
                for num in range(args.ablation, len(logits)):
                    for_set_threshold += (softmax_f(logits[num])).squeeze() 
                
            else: assert 'model id error...'
            
            if for_set_threshold[0].item() + threshold < for_set_threshold[1].item(): prediction_do_more_retriever = 0 # + args.threshold
            else: prediction_do_more_retriever=1
            
            if prediction_do_more_retriever == 0:
                # print(model.to_string(output))
                pred_list.append(model.to_string(output)[0])
                print(for_set_threshold[0].item() + threshold,for_set_threshold[1].item())
            else:
                while prediction_do_more_retriever == 1:
                    cache={}
                    if is_sparse:
                        if retr_count == 0:
                            retrieved_passages = bm25.retrieve(value['text'][0])
                        else:
                            retrieved_passages = bm25.retrieve(search_input_new[0])
                        evidences = return_evidences(retrieved_passages)
                    else:
                        if retr_count == 0:
                            D, I = batch_topk_sim(model_retr, value['text'], index, k = 5)
                        else:
                            
                            D, I = batch_topk_sim(model_retr, search_input_new, index, k = 5)
                        retrieved_passages = list(corpus.iloc[I[0].tolist(),0])
                        
                        evidences = return_evidences(retrieved_passages)
                    new_input = prompt_function_retr(value['text'][0], evidences)
                    
                    with torch.no_grad():
                        
                        output = model.generate(tokenizer(new_input, return_tensors='pt')['input_ids'].to(device), do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
                        output.to('cpu')
                        
                        if 'google/gemma-2b' == model_id:
                            logits = return_prober_logit_gemma_2b(return_mean_output, cfg_list, probers)
                            for_set_threshold = torch.zeros_like(logits[0].squeeze())
                            for num in range(args.ablation, len(logits)):
                                for_set_threshold += (softmax_f(logits[num])).squeeze() 
                            
                        else: assert 'model id error...'
                        
                        if for_set_threshold[0].item() + threshold < for_set_threshold[1].item(): prediction_do_more_retriever = 0 #  + args.threshold
                        else: prediction_do_more_retriever=1
                        
                        search_input_new=model.to_string(output)
                        if (steps + 1) % 3 == 0:
                            print(search_input_new[0])
                        print(for_set_threshold[0].item() + threshold,for_set_threshold[1].item())
                        
                        if retr_count > 2:
                            break
                        else:
                            retr_count += 1
                pred_list.append(search_input_new[0])
                
            retr_count_list.append(retr_count)
            steps += 1
            print(steps)
                
            if steps > steps_limit:   
                end = time.time()
                break

    if retr_method == 'none':
        start = time.time()
        for value in tqdm(dataloader):
            
            with torch.no_grad():
                output = model.generate(value['input_ids'], do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
            pred_list.append(model.to_string(output)[0])
            steps +=1
            if steps > steps_limit:   
                end = time.time()
                break
            
    if retr_method =='simple':
        start = time.time()
        for value in tqdm(dataloader):
            if is_sparse:
                retrieved_passages = bm25.retrieve(value['text'][0])
                evidences = return_evidences(retrieved_passages)
                    
            else:
                D, I = batch_topk_sim(model_retr, value['text'], index, k = 5)
                retrieved_passages = list(corpus.iloc[I[0].tolist(),0])
                evidences = return_evidences(retrieved_passages)
                    
            new_input = prompt_function_retr(value['text'][0], evidences)

            output = model.generate(tokenizer(new_input, return_tensors='pt')['input_ids'].to(device), do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
            output.to('cpu')

            search_input_new=model.to_string(output)[0]
            pred_list.append(search_input_new)
            steps += 1
            if steps > steps_limit:
                end = time.time()
                break
      
    
    acc, metric, pred_to_train=evaluator(df, metric, pred_list,args)
    
    print('time: ',end-start)
    print('acc: ', sum(acc)/len(acc))
    
    if args.extracting_cot_qa:
        if '7' in model_id:
            _save_path = '7b'
        if '2' in model_id:
            _save_path = '2b'
        dfdf=pd.DataFrame([pred_list, pred_to_train, df['answer'][:steps_limit+1], acc]).T
        dfdf.columns= ['pred_with_prompt','pred','answer','acc']
        
        
        dfdf.to_csv(f"dataset/{_save_path}/retrieval_qa_{model_id.split('/')[1]}_{dataset_name}_{retr_method}_{tr_or_dev}_{save_data_name}_{steps_limit}.csv", index=False)
        
        print('making retrieval dataset is end !!!')
    
    else:
        if (args.dataset_name == 'hotpotqa') or (args.dataset_name == '2wikimultihopqa') or (args.dataset_name == 'musique') or (args.dataset_name == 'iirc'):
            df = pd.DataFrame([[retr_method], [end-start],[sum(acc)/len(acc)], [metric.get_metric()['title_em']], [metric.get_metric()['title_f1']]]).T
            if retr_method == 'probing':
                dfdf_clf_pred = pd.DataFrame([str(retr_count_list)])    
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_clf_pred, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'clf_pred', 'acc.1']
            else:
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'acc.1']
            
        else:
            df = pd.DataFrame([[retr_method], [end-start],[sum(acc)/len(acc)], [metric.get_metric()['em']], [metric.get_metric()['f1']]]).T
            if retr_method == 'probing':
                dfdf_clf_pred = pd.DataFrame([str(retr_count_list)])    
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_clf_pred, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'clf_pred', 'acc.1']
            else:
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'acc.1']
        df.to_csv(f'result/{args.ablation}_{_ds}_{retr_type}_{dataset_name}_{threshold}_{retr_method}_{savename_is_cot}_{tr_or_dev}_{steps_limit}.csv', index=False)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retr_method', type=str, default='') # probing, none, adaptive, simple, flare, dragin, fix-length-retrieval, fix-sentence, linguistic
    parser.add_argument('--position', type=str, default='resid_post') # attn_out, resid_mid, mlp_out, resid_post
    parser.add_argument('--dataset_name', type=str, default='hotpotqa') # hotpotqa, nq, musique, 2wikimultihopqa, squad, trivia
    
    # dnese - squad, hotptoqa 메모리 부족 이슈
    # sparse - squad, hotptoqa 
    parser.add_argument('--model_id', type=str, default='google/gemma-2b') # google/gemma-2b mistralai/Mistral-7B-v0.1
    parser.add_argument('--tr_or_dev', type=str, default='dev') # train
    
    parser.add_argument('--ds', type=int, default=777) # 25,5, 75, 1000, else 0
    parser.add_argument('--ablation', type=int, default=0) # 0-> 0 이후 모든 값 더하기
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--steps_limit', type=int, default=10000) # 1500 - 3 
    
    parser.add_argument('--is_sparse', action='store_true')
    parser.add_argument('--is_cot', action='store_true')
    parser.add_argument('--extracting_cot_qa', action='store_true')
    parser.add_argument('--extract_sep', action='store_true')
    parser.add_argument('--sep_number', type=int, default=3200)
    
    args = parser.parse_args()
    main(args)
    
#%%
    #%%
'''
###################################### make_dataset ##########################################################
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name trivia --is_cot --sep_number 0
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name hotpotqa --is_cot --sep_number 0
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name nq --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name trivia --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name hotpotqa --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name nq --is_cot --sep_number 0

###################################### exp ##########################################################

python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name nq --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name musique --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name hotpotqa --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name trivia --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name 2wikimultihopqa --is_cot --is_sparse --model_id google/gemma-2b --ds 3
'''