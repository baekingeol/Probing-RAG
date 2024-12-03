#%%
from typing import Union
import argparse
from functools import partial
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, OneCycleLR, ExponentialLR
from torch.optim import AdamW
# from utils import CustomDataset, ImprovedProbe, Config_Maker, load_prober

from transformer_lens import HookedTransformer

from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import Config_Maker, load_prober_config_gemma_2b, load_probers_gemma_2b, return_prober_logits_gemma_2b
from prompts import dummy_prompt

import os
def main(args):
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    set_seed(22)

    model_id = args.model_id
    device = 'cuda' # args.device
    model = HookedTransformer.from_pretrained(model_id, device = device)
    # 'simple_qa_dataset/odqa_dataset/2b/retrieval_qa_gemma-2b_all_zeroshot_dev_2241.csv'

    dev_data_path = 'simple_qa_dataset/odqa_dataset/2b/retrieval_qa_gemma-2b_all_zeroshot_dev_2241.csv'
    dev_df=pd.read_csv(dev_data_path)
    #%%
    softmax= nn.Softmax(dim = -1)

    
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
    dev_df['pred_with_prompt'] = dev_df['pred_with_prompt'].apply(lambda x: x.split('\n\n')[4])
    dataloader=DataLoader(CustomDataset(dev_df['pred_with_prompt'], prompt = dummy_prompt, tokenizer=model.tokenizer), shuffle=False)
    #%%
    position ='resid_post'
    prober6_cfg, prober8_cfg, prober10_cfg, prober12_cfg, prober14_cfg, prober16_cfg= load_prober_config_gemma_2b(model, Config_Maker, position, device)
    prober_6, prober_8, prober_10, prober_12, prober_14, prober_16 = load_probers_gemma_2b(prober6_cfg, prober8_cfg, prober10_cfg, prober12_cfg, prober14_cfg, prober16_cfg)
    # prober20_et = load_prober(prober20_et_cfg)
    softmax = nn.Softmax(dim = -1)
    #%%
    # shapes= [i.shape for i in cache[layer_name]]
    #%%
    cache = {}
    model.reset_hooks()
    def return_mean_output(model, dev_df, num, prober_cfg, prober):
        layer_name = f'blocks.{prober_cfg.layer}.hook_{prober_cfg.position}'
        tok1 = model.to_tokens(dev_df['question_with_prompt'][num]).to('cpu')
        tok2 = model.to_tokens(dev_df['pred'][num]).to('cpu')
        # import pdb;pdb.set_trace()
        # input=torch.concat(cache[layer_name][0][:,-tok2.shape[1]:,:], dim = 1).to(device)
        
        input = torch.sum(cache[layer_name][0][:,-tok2.shape[1]:,:].to(device), dim = 1)
        logit=prober(input)
        return logit

    def hook_fn(activations, hook, layer):
        if layer not in cache:
            cache[layer] = []
        cache[layer].append(activations.detach().cpu())
        return activations

    def add_layer_hook(model, layer_name):
        hook = partial(hook_fn, layer=layer_name)
        model.add_hook(layer_name, hook)

    layer_configs = [prober6_cfg, prober8_cfg, prober10_cfg, prober12_cfg, prober14_cfg, prober16_cfg]

    for prober_cfg in layer_configs:
        layer_name = f'blocks.{prober_cfg.layer}.hook_{prober_cfg.position}'
        add_layer_hook(model, layer_name)
    
    # predictions12,predictions14,predictions16,predictions18,predictions20,predictions22, predictions24,predictions26,predictions28, predictions30= [],[],[],[],[],[],[],[],[],[]
    
    predictions6, predictions8, predictions10, predictions12, predictions14, predictions16, = [], [], [], [], [], [] 
    logit6_list, logit8_list, logit10_list, logit12_list, logit14_list, logit16_list= [], [], [], [], [], [] 
    count = 0
    # limit_num = 5000
    def predictions_function(model, dev_df, num, prober_cfg, prober):
        
        logit_mean_token = return_mean_output(model, dev_df, num, prober_cfg, prober)
        prediction = torch.argmax(softmax(logit_mean_token),dim = 1)
        return int(prediction.to('cpu')), logit_mean_token
        # predictions.append(int(prediction.to('cpu')))
    acc_list = list(dev_df['acc'])
    cherry_pick_devset = []
    for num in tqdm(range(len(dev_df))):
        cache={}
        retr_count = 0
        with torch.no_grad():
            # output = model.generate(value['input_ids'], use_past_kv_cache=True, do_sample=False)
            model.run_with_cache(dev_df['pred_with_prompt'][num])
            pred_6, logit_6=predictions_function(model, dev_df, num, prober6_cfg, prober_6)
            pred_8, logit_8=predictions_function(model, dev_df, num, prober8_cfg, prober_8)
            pred_10, logit_10=predictions_function(model, dev_df, num, prober10_cfg, prober_10)
            pred_12, logit_12=predictions_function(model, dev_df, num, prober12_cfg, prober_12)
            pred_14, logit_14=predictions_function(model, dev_df, num, prober14_cfg, prober_14)
            pred_16, logit_16=predictions_function(model, dev_df, num, prober16_cfg, prober_16)
            # import pdb;pdb.set_trace()
            logit6_list.append(logit_6.to('cpu').tolist()[0])
            logit8_list.append(logit_8.to('cpu').tolist()[0])
            logit10_list.append(logit_10.to('cpu').tolist()[0])
            logit12_list.append(logit_12.to('cpu').tolist()[0])
            logit14_list.append(logit_14.to('cpu').tolist()[0])
            logit16_list.append(logit_16.to('cpu').tolist()[0])
            # print(pred_6, pred_8, pred_10, pred_12, pred_14, pred_16)
            # import pdb;pdb.set_trace()
            predictions6.append(pred_6)
            predictions8.append(pred_8)
            predictions10.append(pred_10)
            predictions12.append(pred_12)
            predictions14.append(pred_14)
            predictions16.append(pred_16)

        count += 1
    
    from sklearn.metrics import accuracy_score
    acc6 = accuracy_score(predictions6,acc_list)
    acc8 = accuracy_score(predictions8,acc_list)
    acc10 = accuracy_score(predictions10,acc_list)
    acc12 = accuracy_score(predictions12,acc_list)
    acc14 = accuracy_score(predictions14,acc_list)
    acc16 = accuracy_score(predictions16,acc_list)
    
    df = pd.DataFrame([acc6,acc8,acc10,acc12,acc14,acc16]).T
    df.columns = ['acc6','acc8','acc10','acc12','acc14','acc16']

    path = os.getcwd()
    if os.path.isdir(path + '/result'): pass
    else: os.mkdir(path + '/result')
    if os.path.isdir(path + '/result/probing_evaluation'): pass
    else: os.mkdir(path + '/result/probing_evaluation')
    import pdb;pdb.set_trace()
    # dfdfd = pd.DataFrame([str(cherry_pick_devset)])
    # dfdfd.to_csv('simple_qa_dataset/odqa_dataset/2b/zero.csv', index=False)
    if args.is_kde:
        kde_path = path + '/result/kde'
        if os.path.isdir(kde_path): pass
        else: os.mkdir(kde_path)
        dfdf=pd.DataFrame([str(logit6_list),str(logit8_list),str(logit10_list),str(logit12_list),str(logit14_list),str(logit16_list)]).T
        import pdb;pdb.set_trace()
        dfdf.columns = ['6','8','10','12','14','16']
        dfdf.to_csv(kde_path+f'/prob_kde_{args.position}.csv', index=False)
    else:
        df.to_csv(f"result/probing_evaluation/{model_id.split('/')[1]}_{args.position}_acc.csv", index=False)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/gemma-2b')
    parser.add_argument('--position', type=str, default='resid_mid') # attn_out, resid_mid, mlp_out, resid_post
    parser.add_argument('--is_kde', default=False, action='store_true')
    # parser.add_argument('--num_', type=str, default='resid_mid')
    args = parser.parse_args()
    main(args)
'''
python exp_evaluation_probing.py --position resid_mid
python exp_evaluation_probing.py --position resid_post --is_kde


'''