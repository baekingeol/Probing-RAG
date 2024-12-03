# adaptive ragvs probing clf 성능비교
#%%
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import ast
#%%
dataset_name=['nq', 'trivia','musique', '2wikimultihopqa']
df_concat = pd.DataFrame()
for num, j in enumerate(dataset_name):
    if j == '2wikimultihopqa':
        path_none =f'result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_none_cot_dev_500.csv'
        path_linguistic = f"result/qa_evaluation/gemma-2b/trained_ds_len/dr_len_75_sparse_{j}_0.0_linguistic_cot_dev_500.csv"
        path_prober = f"result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_probing_cot_dev_500.csv"
        path_adaptive = f"result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_adaptive_cot_dev_500.csv"
        path_simple = f"result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_simple_cot_dev_500.csv"
        path_dragin = f"result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_dragin_cot_dev_500.csv"
        path_flare = f"result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_flare_cot_dev_500.csv"
    else:
        path_none = f'result/qa_evaluation/gemma-2b/none_{j}_prober_method2_12-30_none_cot_dev_500.csv'
        path_linguistic = f"result/qa_evaluation/gemma-2b/trained_ds_len/dr_len_75_sparse_{j}_0.0_linguistic_cot_dev_500.csv"
        path_prober = f'result/qa_evaluation/gemma-2b/sparse_{j}_prober_method2_12-30_probing_cot_dev_500.csv'
        path_adaptive = f'result/qa_evaluation/gemma-2b/sparse_{j}_prober_method2_12-30_adaptive_cot_dev_500.csv'
        path_simple = f'result/qa_evaluation/gemma-2b/sparse_{j}_prober_method2_12-30_simple_cot_dev_500.csv'
        path_dragin = f"result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_dragin_cot_dev_500.csv"
        path_flare = f"result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_flare_cot_dev_500.csv"
    path_prober = f'/home/baekig/probing_rag/result/qa_evaluation/gemma-2b/trained_ds_len/len_75_sparse_{j}_0.0_probing_cot_dev_500.csv'
    
    df_none=pd.read_csv(path_none)[:500]
    df_none['dataset_name'] = f'{j}'
    df_linguistic=pd.read_csv(path_linguistic)[:500]
    df_linguistic['dataset_name'] = f'{j}'
    df_prober=pd.read_csv(path_prober)[:500]
    df_prober['dataset_name'] = f'{j}'
    df_adaptive=pd.read_csv(path_adaptive)[:500]
    df_adaptive['dataset_name'] = f'{j}'
    df_simple=pd.read_csv(path_simple)[:500]
    df_simple['dataset_name'] = f'{j}'
    df_dragin=pd.read_csv(path_dragin)[:500]
    df_dragin['dataset_name'] = f'{j}'
    df_flare=pd.read_csv(path_flare)[:500]
    df_flare['dataset_name'] = f'{j}'

    df = pd.concat([df_none, df_linguistic, df_prober, df_adaptive, df_simple, df_dragin, df_flare], axis = 0)
    df_concat = pd.concat([df_concat, df], axis = 0)
    df_swap = df
# %%

# %%
''' exp1
none 에서 맞췄던걸 simple에서 틀릴 때 (외부지식의 영향으로 맞추던걸 틀림). 
none에서 맞췄던걸 prober, adaptive rag 에서 맞추거나 틀릴 때 비교  → (motivation 2에 대한 뒷받침 근거 )
'''
# dataset_name=['hotpotqa', 'musique', 'nq', 'squad', 'trivia']
prob_correct, prob_incorrect, adaptive_correct, adaptive_incorrect, simple_correct, simple_incorrect, linguistic_correct, linguistic_incorrect = [],[], [], [], [], [], [], []
flare_correct, flare_incorrect, dragin_correct, dragin_incorrect = [], [], [], []
for j in dataset_name:
    _df=df_concat[df_concat['dataset_name'] ==f'{j}']
    true_df = _df[_df['retr_method']=='none']
    
    pred_prob_df = _df[_df['retr_method']=='probing']
    
    pred_adaptive_df = _df[_df['retr_method']=='adaptive']
    pred_simple_df = _df[_df['retr_method']=='simple']
    pred_linguistic_df = _df[_df['retr_method']==df_linguistic['retr_method'].item()]
    pred_dragin_df = _df[_df['retr_method']=='dragin']
    pred_flare_df = _df[_df['retr_method']=='flare']
    
    true = ast.literal_eval(true_df['acc.1'][0])
    pred_prob = ast.literal_eval(pred_prob_df['acc.1'][0])
    pred_adaptive = ast.literal_eval(pred_adaptive_df['acc.1'][0])
    pred_linguistic = ast.literal_eval(pred_linguistic_df['acc.1'][0])
    pred_simple = ast.literal_eval(pred_simple_df['acc.1'][0])
    pred_dragin = ast.literal_eval(pred_dragin_df['acc.1'][0])
    pred_flare = ast.literal_eval(pred_flare_df['acc.1'][0])
    
    cm_1 = confusion_matrix(true, pred_prob)
    cm_2 = confusion_matrix(true, pred_adaptive)
    cm_3 = confusion_matrix(true, pred_simple)
    cm_4 = confusion_matrix(true, pred_dragin)
    cm_5 = confusion_matrix(true, pred_flare)
    cm_6 = confusion_matrix(true, pred_linguistic)
    
    prob_correct.append(sum(cm_1.diagonal()))
    prob_incorrect.append(500 - sum(cm_1.diagonal()))    
    adaptive_correct.append(sum(cm_2.diagonal()))
    adaptive_incorrect.append(500 - sum(cm_2.diagonal()))    
    simple_correct.append(sum(cm_3.diagonal()))
    simple_incorrect.append(500 - sum(cm_3.diagonal()))  
    dragin_correct.append(sum(cm_4.diagonal()))
    dragin_incorrect.append(500 - sum(cm_4.diagonal()))    
    flare_correct.append(sum(cm_5.diagonal()))
    flare_incorrect.append(500 - sum(cm_5.diagonal()))    
    linguistic_correct.append(sum(cm_6.diagonal()))
    linguistic_incorrect.append(500 - sum(cm_6.diagonal()))    
    
# %%
print('prober:',prob_correct, prob_incorrect)
print('adaptive:',adaptive_correct, adaptive_incorrect)
print('simple:',simple_correct, simple_incorrect)
print('dragin:',dragin_correct, dragin_incorrect)
print('flare:',flare_correct, flare_incorrect)
print('linguistic:',linguistic_correct, linguistic_incorrect)
#%%
# [(j/500) *100 for j in flare_correct]
# %%
'''exp2
adaptive rag vs probing clf 성능비교
'''
# %%
dataset_name=['hotpotqa', 'musique', 'nq', 'trivia', '2wikimultihopqa', 'iirc']
prob_correct, prob_incorrect, adaptive_correct, adaptive_incorrect = [], [], [], []
#%%
for k in dataset_name:
# k = 'hotpotqa'
    _df=df_concat[df_concat['dataset_name'] ==f'{k}']

    true_df = _df[_df['retr_method']=='none']
    pred_prob_df = _df[_df['retr_method']=='probing']
    pred_adaptive_df = _df[_df['retr_method']=='adaptive']

    true = ast.literal_eval(true_df['acc.1'][0])
    pred_prob = ast.literal_eval(pred_prob_df['clf_pred'][0])
    pred_adaptive = ast.literal_eval(pred_adaptive_df['clf_pred'][0])
    true = [1 if j == 0 else 0 for j in true]
    pred_prob = [0 if j == 0 else 1 for j in pred_prob]
    pred_adaptive = [0 if j == 0 else 1 for j in pred_adaptive]
    
    df_dev = pd.DataFrame([true, pred_prob, pred_adaptive]).T
    df_dev.columns = ['true','pred_prob','pred_adaptive']
    
    num_of_one = sum(df_dev['true'])
    num_of_zero = 500 - sum(df_dev['true'])
    if num_of_one > num_of_zero:
        index1 = df_dev[df_dev['true'] == 1].sample(n=num_of_zero).index
        index2 = df_dev[df_dev['true'] == 0].index
    else:
        index1 = df_dev[df_dev['true'] == 0].sample(n=num_of_one).index
        index2 = df_dev[df_dev['true'] == 1].index
    
    df_dev =df_dev.iloc[list(index1) + list(index2), :].reset_index(drop=True)
    
    cm_1 = confusion_matrix(df_dev['true'], df_dev['pred_prob'])
    cm_2 = confusion_matrix(df_dev['true'], df_dev['pred_adaptive'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(cm_1, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    ds_name=k.upper()
    axes[0].set_title(f'{ds_name} Probing Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_2, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title(f'{ds_name} Adaptive RAG Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    prob_correct.append(sum(cm_1.diagonal()))
    prob_incorrect.append(len(df_dev) - sum(cm_1.diagonal()))    
    adaptive_correct.append(sum(cm_2.diagonal()))
    adaptive_incorrect.append(len(df_dev) - sum(cm_2.diagonal()))    
    plt.tight_layout()

    plt.show()
# %%
print(prob_correct, prob_incorrect)
print(adaptive_correct, adaptive_incorrect)
#%%
sum(prob_correct)/(sum(prob_correct) + sum(prob_incorrect))
#%%
sum(adaptive_correct)/(sum(adaptive_correct) + sum(adaptive_incorrect))
#%%
prober_clf_acc, adaptive_clf_acc = [], []
for i in range(len(prob_correct)):
    prober_clf_acc.append(prob_correct[i]/(prob_correct[i] + prob_incorrect[i]))
    adaptive_clf_acc.append(adaptive_correct[i]/(adaptive_correct[i] + adaptive_incorrect[i]))
#%%
prober_clf_acc, adaptive_clf_acc
#%%
'''exp5
no, single, multi retrieval percentage 
'''
#%%
#%%
import ast
# %%
# dataset_name=['hotpotqa', 'musique', 'nq', 'squad', 'trivia']
methods = ['probing', 'adaptive', 'dragin', 'flare','linguistic']
df = pd.DataFrame()
df_retr_call = pd.DataFrame()
df_concat = df_concat[:500]
for j in dataset_name:
    for i in methods:
        _df=df_concat[df_concat['dataset_name'] ==f'{j}']
        dfdf = _df[_df['retr_method']==i]
            
        clf = ast.literal_eval(dfdf['clf_pred'][0])
        acc_1 = ast.literal_eval(dfdf['acc.1'][0])
        ac=pd.DataFrame([acc_1, clf]).T
        ac.columns = ['acc_value', 'clf_pred']
        
        clf_none = ac[ac['clf_pred'] == 0]
        clf_single = ac[ac['clf_pred'] == 1]
        clf_multi = ac[ac['clf_pred'] >1]
        
        devide_values = [len(clf_none['acc_value']), len(clf_single['acc_value']), len(clf_multi['acc_value'])]
        for k in range(len(devide_values)):
            if devide_values[k] == 0:
                devide_values[k]= 1
        
        df_acc = pd.DataFrame([sum(clf_none['acc_value'])/devide_values[0], sum(clf_single['acc_value'])/devide_values[1], sum(clf_multi['acc_value'])/devide_values[2]])
            
        # if len(clf_none['acc_value']) ==0:
        #     df_acc = pd.DataFrame([sum(clf_none['acc_value'])/1, sum(clf_single['acc_value'])/len(clf_single['acc_value']),sum(clf_multi['acc_value'])/len(clf_multi['acc_value'])])    
        # elif len(clf_single['acc_value']) == 0:
        #     df_acc = pd.DataFrame([sum(clf_none['acc_value'])/len(clf_none['acc_value']), sum(clf_single['acc_value'])/1,sum(clf_multi['acc_value'])/len(clf_multi['acc_value'])])    
        # else:
        #     df_acc = pd.DataFrame([sum(clf_none['acc_value'])/len(clf_none['acc_value']), sum(clf_single['acc_value'])/len(clf_single['acc_value']),sum(clf_multi['acc_value'])/len(clf_multi['acc_value'])])
        total = int(len(clf_none['acc_value'])) + int(len(clf_single['acc_value'])) +int(len(clf_multi['acc_value']))
        df_percentage = pd.DataFrame([int(len(clf_none['acc_value']))/total, int(len(clf_single['acc_value']))/total, int(len(clf_multi['acc_value']))/total])
        df_num_of_retr_call = pd.DataFrame([len(clf_none),len(clf_single),len(clf_multi),sum(clf_none['clf_pred']), sum(clf_single['clf_pred']), sum(clf_multi['clf_pred']), sum(clf)])
        
        df_acc.columns = [f"{j}_{i}_acc"]
        df_percentage.columns = [f"{j}_{i}_clf_percentage"]
        df_num_of_retr_call.columns = [f"{j}_{i}"]
        
        df = pd.concat([df,df_acc, df_percentage],axis = 1)
        df_retr_call = pd.concat([df_retr_call, df_num_of_retr_call], axis = 1)
# %%
df_save = df.T
df_save.columns = ['none', 'one', 'multi']
#%%
df_save
#%%
df_retr_save = df_retr_call.T
df_retr_save.columns = ['num_none','num_one','num_multi','none', 'one', 'multi', 'retr_call']
# %% avg
probing_table = df_retr_save.iloc[0::2,:]
adaptive_table = df_retr_save.iloc[1::2,:]
probing_total = sum(probing_table['num_none'])+ sum(probing_table['num_one'])+ sum(probing_table['num_multi'])
adaptive_total = sum(adaptive_table['num_none']) + sum(adaptive_table['num_one']) + sum(adaptive_table['num_multi'])
print(sum(probing_table['num_none'])/probing_total, sum(probing_table['num_one'])/probing_total, sum(probing_table['num_multi'])/probing_total)
print(sum(adaptive_table['num_none'])/adaptive_total, sum(adaptive_table['num_one'])/adaptive_total, sum(adaptive_table['num_multi'])/adaptive_total)
print(sum(probing_table['retr_call']), sum(adaptive_table['retr_call']))
# %%
probing_table = df_retr_save.iloc[0::5,:]
adaptive_table = df_retr_save.iloc[1::5,:]
dragin_table = df_retr_save.iloc[2::5,:]
flare_table = df_retr_save.iloc[3::5,:]
linguistic_table = df_retr_save.iloc[4::5,:]
#%%
2345/1988
#%%
2500*sum(linguistic_table['num_one'])/2000
# %%
'''exp6
ablation
'''
# %%
import os
import pandas as pd
# %%
path=os.getcwd() + '/result/qa_evaluation/gemma-2b/ablation'
files = os.listdir(path)
# %%
dataset_name=['hotpotqa', 'musique', 'nq', 'squad', 'trivia']
df = pd.DataFrame()
for k in dataset_name:
    dataset_path = [j for j in sorted(files) if k in j]

    for j in dataset_path:
        _df = pd.read_csv(path + '/' + j)
        df = pd.concat([df, _df], axis = 0)
# %%
df
# %%
'''exp4: experiment kde'''
#%%
import pandas as pd
import ast
path = 'result/kde/prob_kde_resid_post.csv'
df = pd.read_csv(path)
# %%
list_6 = ast.literal_eval(df['8'][0])
list_8 = ast.literal_eval(df['10'][0])
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataA = np.array(list_6)
dataB = np.array(list_8)
df = pd.DataFrame({
    'x': np.concatenate([dataA[:, 0], dataA[:, 1]]),
    'y': np.concatenate([dataB[:, 0], dataB[:, 1]]),
    'type': ['Pass Retrieval'] * len(dataA) + ['Call Retrieval'] * len(dataA)
})

colors = {'Pass Retrieval': '#1266FF', 'Call Retrieval': '#FF8224'}

g = sns.jointplot(
    data=df,
    x='x', y='y',
    hue='type',
    kind='kde',
    height=6, aspect=0.7,
    fill=False, 
    legend=False,
    palette=colors
)

g.set_axis_labels('Projection on the 1st Probing Layer Direction', 
                  'Projection on the 2nd Probing Layer Direction', 
                  fontsize=14)

# legend 추가 및 위치 조정
for name, color in colors.items():
    g.ax_joint.plot([], [], color=color, label=name)

g.ax_joint.legend(title='type', loc='upper left', bbox_to_anchor=(0, 1.02), fontsize=10)

plt.tight_layout()
# plt.savefig('result/kde/kde2.pdf')
plt.show()
# g.figure.savefig("result/kde/kde_projection.pdf")

# g.legend(loc='upper left')
#%%
'''exp5: experiment kde & heatmap'''
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame with the data provided
data = {
    'Layer': [6, 8, 10, 12, 14, 16],
    'resid_mid(ep=1)': [0.668, 0.69, 0.704, 0.696, 0.704, 0.664],
    'resid_mid(ep=2)': [0.662, 0.692, 0.708, 0.706, 0.682, 0.668],
    'resid_post(ep=1)': [0.674, 0.706, 0.694, 0.71, 0.688, 0.688],
    'resid_post(ep=2)': [0.692, 0.7, 0.718, 0.71, 0.674, 0.688]
}

df = pd.DataFrame(data)

# Melt the DataFrame to long format for easier plotting
df_melted = pd.melt(df, id_vars=['Layer'], var_name='Position', value_name='Accuracy')

# Create a heatmap
plt.figure(figsize=(8, 6))
heatmap_data = df_melted.pivot(index="Layer", columns="Position", values="Accuracy")

# Sort index in descending order so that the heatmap starts from 6 at the bottom
heatmap_data = heatmap_data.iloc[::-1]

# Create the heatmap with yticklabels in reverse order
swarm_plot = sns.heatmap(heatmap_data, annot=True, fmt=".3f",cmap="YlGnBu", cbar=True, yticklabels=heatmap_data.index)
fig = swarm_plot.get_figure()
plt.xlabel("Position", fontsize=15)
plt.ylabel("Layer", fontsize=15)
plt.title('Heatmap of Accuracy by Layer and Position', fontsize=18)
# plt.savefig("result/kde/layer_clf_acc_headmap.pdf", format='pdf')
plt.show()
# fig.savefig('result/kde/layer_clf_acc_headmap2.pdf')
# %%
swarm_plot = sns.swarmplot(...)
fig = swarm_plot.get_figure()
fig.savefig("out.png") 
# coolwarm, YlGnBu
# %%
######################### Appendix - correlation between prober classification and performance analysis #########################
#%%
import ast
from sklearn.metrics import confusion_matrix
none_datas = {}
abls = [6,8,10,12,14]
thresholds = [-2.0, -1.0, 0.0, 1.0, 2.0]
datasets = ['nq', 'trivia','hotpotqa', 'musique', '2wikimultihopqa', 'iirc']

for num, j in enumerate(datasets):
    if (j == 'iirc') or (j == '2wikimultihopqa'):
        path_none =f'result/qa_evaluation/gemma-2b/main/sparse_{j}_0.0_none_cot_dev_500.csv'
    else:
        path_none = f'result/qa_evaluation/gemma-2b/none_{j}_prober_method2_12-30_none_cot_dev_500.csv'
    _df_none = pd.read_csv(path_none)
    _samp = ast.literal_eval(_df_none.to_json(orient = 'records'))[0]
    _samp['acc.1'] = ast.literal_eval(_samp['acc.1'])
    none_datas[f"{j}"] = _samp
#%%
datas = {}
# inversed performance weighting

for abl in abls:
# for threshold in thresholds:
    avg_acc = []
    weights = []
    inv_avg_acc = []
    avg_clf_performance = 0
    for data_name in datasets:
        _path = f'/home/baekig/probing_rag/result/qa_evaluation/gemma-2b/ablation/len_75_abl_{abl}_sparse_{data_name}_0.0_probing_cot_dev_500.csv'
        # _path = f'result/qa_evaluation/gemma-2b/trained_ds_len/len_75_sparse_{data_name}_{str(threshold)}_probing_cot_dev_500.csv'    
        _df = pd.read_csv(_path)
        _sample = ast.literal_eval(_df.to_json(orient='records'))[0]
        _sample['clf_pred'] = ast.literal_eval(_sample['clf_pred'])
        _sample['acc.1'] = ast.literal_eval(_sample['acc.1'])
        
        true = [1 if j == 0 else 0 for j in none_datas[f"{data_name}"]['acc.1']]
        pred_prob = [1 if i > 0 else 0 for i in _sample['clf_pred']]
        # true = [1 if j == 0 else 0 for j in true]
        # pred_prob = [0 if j == 0 else 1 for j in pred_prob]
        
        cm_1 = confusion_matrix(true, pred_prob)

        correct = sum(cm_1.diagonal())
        incorrect = len(true) - sum(cm_1.diagonal())
        performance = correct /(correct + incorrect)    
        _sample['clf_performance'] = performance
        avg_clf_performance += performance
        
        datas[f"{data_name}_{str(abl)}"] = _sample
        avg_acc.append(_sample['acc'] * 100)
        inv_avg_acc.append(1/_sample['acc'])
    
    for inv_acc in avg_acc:
        weights.append(inv_acc/sum(avg_acc))
    
    weighted_average = sum([value * weight for value, weight in zip(avg_acc, weights)])
    
    datas[f"avg_acc_{str(abl)}"]=weighted_average
    datas[f"avg_clf_{str(abl)}"]=avg_clf_performance/len(datasets)
# for abl in abls:
for threshold in thresholds:
    avg_acc = []
    weights = []
    inv_avg_acc = []
    avg_clf_performance = 0
    for data_name in datasets:
        # _path = f'/home/baekig/probing_rag/result/qa_evaluation/gemma-2b/ablation/len_75_abl_{abl}_sparse_{data_name}_0.0_probing_cot_dev_500.csv'
        _path = f'result/qa_evaluation/gemma-2b/trained_ds_len/len_75_sparse_{data_name}_{str(threshold)}_probing_cot_dev_500.csv'    
        _df = pd.read_csv(_path)
        _sample = ast.literal_eval(_df.to_json(orient='records'))[0]
        _sample['clf_pred'] = ast.literal_eval(_sample['clf_pred'])
        _sample['acc.1'] = ast.literal_eval(_sample['acc.1'])
        
        true = [1 if j == 0 else 0 for j in none_datas[f"{data_name}"]['acc.1']]
        pred_prob = [1 if i > 0 else 0 for i in _sample['clf_pred']]
        # true = [1 if j == 0 else 0 for j in true]
        # pred_prob = [0 if j == 0 else 1 for j in pred_prob]
        
        cm_1 = confusion_matrix(true, pred_prob)

        correct = sum(cm_1.diagonal())
        incorrect = len(true) - sum(cm_1.diagonal())
        performance = correct /(correct + incorrect)    
        _sample['clf_performance'] = performance
        avg_clf_performance += performance
        
        datas[f"{data_name}_{str(threshold)}"] = _sample
        avg_acc.append(_sample['acc'] * 100)
        inv_avg_acc.append(1/_sample['acc'])
    
    for inv_acc in avg_acc:
        weights.append(inv_acc/sum(avg_acc))
    
    weighted_average = sum([value * weight for value, weight in zip(avg_acc, weights)])
    
    datas[f"avg_acc_{str(threshold)}"]=weighted_average
    datas[f"avg_clf_{str(threshold)}"]=avg_clf_performance/len(datasets)
#%%

# %%
clfs = []
accs = []
for abl in abls:
# for threshold in thresholds:
    clfs.append(datas[f'avg_clf_{str(abl)}'])
    accs.append(datas[f"avg_acc_{str(abl)}"])
for threshold in thresholds:
    clfs.append(datas[f'avg_clf_{str(threshold)}'])
    accs.append(datas[f"avg_acc_{str(threshold)}"])
    # print()
#%%
#%%
import numpy as np

data_A = np.array(clfs)
data_B = np.array(accs)

correlation = np.corrcoef(data_A, data_B)[0, 1]
correlation
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Given data
classification_performance = data_A
qa_performance = data_B

# Calculate Pearson correlation
correlation, _ = pearsonr(classification_performance, qa_performance)

# Plotting with correlation line
plt.figure(figsize=(6, 6))
plt.scatter(classification_performance, qa_performance, color='blue', label=f'Data Points')

# Fit a line to the data
m, b = np.polyfit(classification_performance, qa_performance, 1)
plt.plot(classification_performance, m * classification_performance + b, color='red', label=f'Correlation: {correlation:.2f}')

plt.xlabel('Classification Performance', fontsize=15)
plt.ylabel('Weighted Average QA Performance', fontsize=15)
plt.legend()
plt.grid(True)
plt.savefig('result/kde/correlation2.pdf')
# plt.show()
#%% ############################################ inversed performance weighting ################################
import numpy as np
def inverse_weighted_average(values):
    return len(values) / np.sum(1 / values)
#%%
data_probing_layer = {
    "Probing Layer": ["6", "6, 8", "6, 8, 10", "6, 8, 10, 12", "6, 8, 10, 12, 14"],
    "HotpotQA": [38.12, 38.12, 38.12, 38.92, 39.12],
    "NQ": [36.72, 33.13, 32.34, 33.73, 34.53],
    "TriviaQA": [50.70, 48.90, 48.70, 49.30, 49.70],
    "MuSiQue": [8.18, 8.78, 8.98, 9.38, 9.98],
    "2Wiki": [42.32, 43.51, 44.31, 44.31, 44.31],
    "IIRC": [24.35, 23.35, 23.95, 24.55, 24.75]
}

# Create DataFrame
df_probing_layer = pd.DataFrame(data_probing_layer)

# Calculating inverse performance weighted average for each probing layer
df_probing_layer['Inverse Weighted Average'] = df_probing_layer.apply(lambda row: inverse_weighted_average(row[1:]), axis=1)
df_probing_layer
#%%
data_new = {
    "Training Dataset Size": ["1000 data points", "1/3 of the total dataset", "2/3 of the total dataset", "The total dataset"],
    "HotpotQA": [36.92, 37.72, 38.92, 39.12],
    "NQ": [33.73, 32.73, 33.93, 35.53],
    "TriviaQA": [48.10, 48.90, 50.30, 50.50],
    "MuSiQue": [9.78, 9.38, 9.98, 9.98],
    "2Wiki": [41.72, 43.51, 44.11, 43.71],
    "IIRC": [23.35, 23.35, 24.95, 24.95]
}

# Create new DataFrame
df_new = pd.DataFrame(data_new)

# Calculating inverse performance weighted average for each training dataset size
df_new['Inverse Weighted Average'] = df_new.apply(lambda row: inverse_weighted_average(row[1:]), axis=1)

df_new[['Training Dataset Size', 'Inverse Weighted Average']]
#%%
data_threshold = {
    "Threshold (θ)": [-2, -1, 0, 1, 2],
    "HotpotQA": [37.72, 39.12, 39.12, 40.12, 40.72],
    "NQ": [32.33, 34.53, 35.53, 36.53, 38.92],
    "TriviaQA": [47.90, 49.70, 50.50, 51.50, 50.70],
    "MuSiQue": [8.78, 9.98, 9.98, 9.58, 9.38],
    "2Wiki": [44.31, 44.31, 43.71, 42.32, 41.31],
    "IIRC": [23.95, 24.75, 24.95, 25.15, 26.95]
}

df_threshold = pd.DataFrame(data_threshold)
df_threshold['Inverse Weighted Average'] = df_threshold.apply(lambda row: inverse_weighted_average(row[1:]), axis=1)
df_threshold[['Threshold (θ)', 'Inverse Weighted Average']]
#%%

data_em = {
    "Methods": ["No Retrieval", "Single-step Approach", "FLARE", "DRAGIN", "Adaptive-RAG", "Probing-RAG(Ours)"],
    "HotpotQA": [16.8, 14.6, 13.2, 19.8, 13.3, 22.2],
    "NQ": [15.0, 11.4, 9.0, 18.8, 11.4, 21.2],
    "TriviaQA": [37.5, 19.6, 13.8, 42.7, 22.8, 40.7],
    "MuSiQue": [3.20, 1.80, 1.20, 4.20, 1.60, 5.00],
    "2Wiki": [22.6, 22.8, 21.6, 26.5, 21.6, 24.2],
    "IIRC": [11.6, 14.2, 21.6, 14.4, 14.6, 13.8]
}

data_acc = {
    "Methods": ["No Retrieval", "Single-step Approach", "FLARE", "DRAGIN", "Adaptive-RAG", "Probing-RAG(Ours)"],
    "HotpotQA": [27.94, 28.34, 20.96, 22.55, 23.55, 39.12],
    "NQ": [24.55, 25.95, 21.76, 22.16, 25.95, 35.53],
    "TriviaQA": [45.51, 38.72, 30.94, 47.11, 40.72, 50.50],
    "MuSiQue": [4.79, 5.79, 1.50, 4.40, 2.80, 9.98],
    "2Wiki": [43.11, 38.32, 27.74, 27.84, 27.84, 43.71],
    "IIRC": [23.15, 25.54, 23.15, 19.16, 23.15, 24.95]
}

# Create DataFrames for EM and ACC
df_em = pd.DataFrame(data_em)
df_acc = pd.DataFrame(data_acc)

# Calculating inverse performance weighted average for EM
df_em['Inverse Weighted Average EM'] = df_em.apply(lambda row: inverse_weighted_average(row[1:]), axis=1)

# Calculating inverse performance weighted average for ACC
df_acc['Inverse Weighted Average ACC'] = df_acc.apply(lambda row: inverse_weighted_average(row[1:]), axis=1)

# Combine EM and ACC results
df_combined = df_em[['Methods', 'Inverse Weighted Average EM']].merge(df_acc[['Methods', 'Inverse Weighted Average ACC']], on='Methods')

df_combined
# %%
