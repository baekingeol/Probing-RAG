
from typing import Union
from transformer_lens import HookedTransformer
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, OneCycleLR, ExponentialLR
from torch.optim import AdamW
import argparse

from tqdm import tqdm#
import numpy as np
import random
import wandb

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def main(args):
    wandb.init(
        project='probing_train_gemma2b_final_in3', # 'probing_train_cot_7b_v1'
        entity='ingeol',
        name=f"{args.train_ds_ratio}_{args.model_id.split('/')[1]}_linear999_{args.method}_{args.layer}_{args.batch_size}"
    )

    set_seed(args.layer)    
    train_ratio = args.train_ds_ratio
    model_id = args.model_id
    device = args.device
    model = HookedTransformer.from_pretrained(model_id, device = device, cache_dir="/root/data/")
    
    train_data_path = 'dataset/2b/retrieval_qa_gemma-2b_all_train_in3_.csv'
    train_df=pd.read_csv(train_data_path).dropna(axis=0).reset_index(drop=True)
    train_df = train_df[:int(len(train_df) * train_ratio)]
    dev_data_path ='dataset/2b/retrieval_qa_gemma-2b_all_zeroshot_test_500.csv'
    
    
    dev_df=pd.read_csv(dev_data_path)
    
    # train_df = train_df[:1000]
    class Probe(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)
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
        def __init__(self, df, model, max_length = 1536):
            self.df = df
            self.model = model
            self.pad_id = model.tokenizer.pad_token_id
            self.max_length = max_length
        def __len__(self):
            return len(self.df)
        def __getitem__(self, index):
            
            tokens1=self.model.to_tokens(self.df['question_with_prompt'][index]).squeeze().to('cpu')
            tokens2 = self.model.to_tokens(f"{self.df['question_with_prompt'][index]+ ' '+self.df['pred'][index]}").squeeze().to('cpu')
        
            tp = torch.tensor([self.pad_id]) 
            tensor_padding = tp.repeat((self.max_length - tokens2.shape[0]))
            return_tokens = torch.cat((tensor_padding, tokens2))
            
            pred_len = tokens2.shape[-1] - tokens1.shape[-1]        
            acc = self.df['acc'][index]      
        
            return {
                'input_tokens': return_tokens,
                'label': acc,
                'pred_len': pred_len
            }
    batch_size = args.batch_size # 32 -method 1 - 41vram
    num_classes = args.num_classes
    lr = args.lr
    d_model = model.cfg.d_model
    softmax= nn.Softmax(dim = -1)
    sigmoid = nn.Sigmoid()
    layer = args.layer
    epochs = args.epochs
    method = args.method
    
    train_dataset = CustomDataset(train_df, model)
    dev_dataset = CustomDataset(dev_df, model)
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    dev_dataloader=DataLoader(dev_dataset, batch_size=batch_size, shuffle = True)
    
    # probe_bi_l16_attn_out = ImprovedProbe(d_model, num_classes).to(device)
    probe_bi_l16_resid_mid = ImprovedProbe(d_model, num_classes).to(device)
    # probe_bi_l16_mlp_out = ImprovedProbe(d_model, num_classes).to(device)
    probe_bi_l16_resid_post = ImprovedProbe(d_model, num_classes).to(device)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()
    
    optimizer_resid_mid = AdamW(probe_bi_l16_resid_mid.parameters(), lr = lr)
    optimizer_resid_post = AdamW(probe_bi_l16_resid_post.parameters(), lr = lr)

    scheduler_resid_mid = ExponentialLR(optimizer_resid_mid, gamma=0.995)
    scheduler_resid_post = ExponentialLR(optimizer_resid_post, gamma=0.995)

    model.eval()
    
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

    def input_tensor_method1(cache_activation, labels, pred_lens):
        result = []
        new_labels = torch.repeat_interleave(labels, pred_lens).to(device)
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
    def method_1_train(model, optim, scheduler, activations, labels, pred_lens):
        optim.zero_grad()
        input_tensor, new_labels = input_tensor_method1(activations, labels, pred_lens)
        loss, _ = make_loss(model = model, input_tensor=input_tensor, labels=new_labels)
        # optim_scheduler_loss(loss, optim, scheduler)
        loss.backward()
        
        optim.step()
        scheduler.step()
        return round(loss.item(),4), optim.param_groups[0]['lr']

    def method_1_eval(model, activations, labels, pred_lens):
        input_tensor, new_labels = input_tensor_method1(activations, labels, pred_lens)
        loss, logit = make_loss(model = model, input_tensor=input_tensor, labels=new_labels)
        accuracy = return_acc(logit, new_labels)
        return round(accuracy, 4), len(labels), loss

    def _method_2_util(model, activations, labels, pred_lens):
        input_tensor, new_labels = input_tensor_method1(activations, labels, pred_lens)
        
        result = torch.split(input_tensor, pred_lens.tolist())
        averages = [torch.mean(t, dim=0, keepdim=True) for t in result]
        # TODO this line debug
        logit = torch.cat(averages, dim=0)
        
        loss, logit = make_loss(model, logit, labels.to(device))
        return loss, logit

    def method_2_train(model, optim, scheduler, activations, labels, pred_lens,num, accumulation_step):
        loss, logit = _method_2_util(model, activations, labels, pred_lens)
        # optim_scheduler_loss(loss, optim, scheduler)
        # accumulation_step = accumulation_step
        # loss = loss/accumulation_step
        loss.backward()
        # if (num+1) % accumulation_step == 0:
        optim.step()
        scheduler.step()
        optim.zero_grad()
        return round((loss).item(),4), optim.param_groups[0]['lr']

    def method_2_eval(model, activations, labels, pred_lens):
        loss, logit = _method_2_util(model, activations, labels, pred_lens)
        accuracy = return_acc(logit, labels)
        return round(accuracy, 4), len(labels), loss

    def _method_3_util(model, activations, labels, pred_lens):
        input_ids = activations[:,-1,:].squeeze()
        logit=model(input_ids)
        logit = softmax(logit)
        loss = criterion_ce(logit, labels.to(device))
    
        return loss, logit

    def method_3_train(model, optim, scheduler, activations, labels, pred_lens):
        optim.zero_grad()
        loss, _ = _method_3_util(model, activations, labels, pred_lens)
        # optim_scheduler_loss(loss, optim, scheduler)
        
        loss.backward()
        optim.step()
        scheduler.step()
        return round(loss.item(), 4), optim.param_groups[0]['lr']

    def method_3_eval(model, activations, labels, pred_lens):
        loss, logit = _method_3_util(model, activations, labels, pred_lens)
        accuracy=return_acc(logit, labels)
        
        return round(accuracy, 4), len(labels), loss 
    max_acc2 = 0
    max_acc4 = 0
    for epoch in range(epochs):
        probe_bi_l16_resid_mid.to(f'{device}')
        probe_bi_l16_resid_post.to(f'{device}')
        probe_bi_l16_resid_mid.train()
        
        probe_bi_l16_resid_post.train()
        loss1s,loss2s,loss3s,loss4s = [],[],[],[]
        for num, batch in enumerate(tqdm(train_dataloader)):
            with torch.no_grad():
                _, cache = model.run_with_cache(batch['input_tokens'], names_filter = lambda name: name.startswith(f"blocks.{layer}")) # , do_sample=False
            
                activations_attn_out = cache['attn_out', layer]
                activations_resid_mid = cache['resid_mid', layer]
                activations_mlp_out = cache["mlp_out", layer] 
                activations_resid_post = cache['resid_post', layer]
            
            pred_lens = batch['pred_len']
            labels = batch['label']
            
            if method == 'each_token':
                function_fn = method_1_train
                
            if method == 'tokens_mean':
                function_fn = method_2_train
                
            if method == 'last_token':
                function_fn = method_3_train
                
            loss2, lr2=function_fn(model = probe_bi_l16_resid_mid, optim = optimizer_resid_mid, scheduler =scheduler_resid_mid,
                                activations=activations_resid_mid, labels=labels, pred_lens=pred_lens, num = num, accumulation_step=3)
            
            loss4, lr4=function_fn(model = probe_bi_l16_resid_post, optim = optimizer_resid_post, scheduler =scheduler_resid_post,
                                activations=activations_resid_post, labels=labels, pred_lens=pred_lens, num = num, accumulation_step=3)
            wandb.log({
                "loss2": loss2,
                "loss4": loss4,
                "learning_rate": lr2
            })
            loss2s.append(loss2)
            loss4s.append(loss4)
            if num % 200 == 0:
                print('loss: ',sum(loss2s)/len(loss2s), sum(loss4s)/len(loss4s))
                loss1s,loss2s,loss3s,loss4s = [],[],[],[]


        avg_acc1, avg_acc2, avg_acc3, avg_acc4 = [], [], [], []
        total_len = 0
        probe_bi_l16_resid_mid.eval()
        probe_bi_l16_resid_post.eval()
        for num, batch in enumerate(tqdm(dev_dataloader)):
            with torch.no_grad():

                _, cache = model.run_with_cache(batch['input_tokens'], names_filter = lambda name: name.startswith(f"blocks.{layer}"))

                activations_resid_mid = cache['resid_mid', layer]
                activations_resid_post = cache['resid_post', layer]

                pred_lens = batch['pred_len']
                labels = batch['label']

                if method == 'each_token':
                    funtion_fn_eval = method_1_eval

                elif method == 'tokens_mean':
                    funtion_fn_eval = method_2_eval

                elif method == 'last_token':
                    funtion_fn_eval = method_3_eval

                acc2, len_label2, loss2 = funtion_fn_eval(probe_bi_l16_resid_mid, activations_resid_mid, labels, pred_lens)
                acc4, len_label4, loss4 = funtion_fn_eval(probe_bi_l16_resid_post, activations_resid_post, labels, pred_lens)    

                total_len += len_label2

                avg_acc2.append(acc2 * len_label2)
                avg_acc4.append(acc4 * len_label4)
                wandb.log({
                    "eval_loss2": loss2,
                    "eval_loss4": loss4,
                })
            if num > 499:
                break
        total_acc2=round(sum(avg_acc2)/total_len, 4)
        total_acc4=round(sum(avg_acc4)/total_len, 4)
        wandb.log({
            "total_acc2": total_acc2,
            "total_acc4": total_acc4,
        })
        # print('total_acc: ', total_acc1, total_acc2, total_acc3, total_acc4)
        print(f'training info: layer:{args.layer}, method: {args.method}, lr: {args.lr}, batch: {args.batch_size}')


        torch.save(probe_bi_l16_resid_mid.to('cpu').state_dict(), f"pckpt/_3/in3_{train_ratio}_{args.model_id.split('/')[1]}_{method}_{num_classes}_l{layer}_resid_mid_ep{epoch}.pt")
        torch.save(probe_bi_l16_resid_post.to('cpu').state_dict(), f"ckpt/_3/in3_{train_ratio}_{args.model_id.split('/')[1]}_{method}_{num_classes}_l{layer}_resid_post_ep{epoch}.pt")

        if max_acc2 < total_acc2:
            max_acc2 = total_acc2
        if max_acc4 < total_acc4:
            max_acc4 = total_acc4
            
if __name__ == '__main__':
    parser=argparse.ArgumentParser()        
    parser.add_argument('--method', type=str, default='each_token',
                        help="method: 'each_token', 'tokens_mean', 'last_token'")
    parser.add_argument('--train_ds_ratio', type=float,default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--layer', type=int, default=16, help='mistral layer: 32')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_id', default = 'google/gemma-2b')
    args = parser.parse_args()
    main(args)
'''
python train.py --method tokens_mean --batch_size 8 --lr 1e-4 --layer 16 --device cuda:0 --epochs 2
python train.py --method tokens_mean --batch_size 8 --lr 1e-4 --layer 14 --device cuda:0 --epochs 2
python train.py --method tokens_mean --batch_size 8 --lr 1e-4 --layer 12 --device cuda:0 --epochs 2
python train.py --method tokens_mean --batch_size 8 --lr 1e-4 --layer 10 --device cuda:0 --epochs 2
python train.py --method tokens_mean --batch_size 8 --lr 1e-4 --layer 8 --device cuda:0 --epochs 2
python train.py --method tokens_mean --batch_size 8 --lr 1e-4 --layer 6 --device cuda:0 --epochs 2
'''

'''
pip install torch
pip install einops
pip install datasets
pip install tqdm
pip install wandb
pip install faiss-cpu
pip install ir-datasets
pip install -U sentence-transformers
pip install nltk
pip install llama-index
pip install ftfy
pip install llama-index-retrievers-bm25
pip install base58
pip install beautifulsoup4
'''