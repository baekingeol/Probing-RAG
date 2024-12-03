#%%
from torchsummary import summary
# %%
from utils import ImprovedProbe

# %%
prober=ImprovedProbe(input_size=2048, output_size=2)
# %%
summary(prober, (1, 2048))
# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_id2='/home/baekig/probing_rag/paper_code/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xxl/epoch/35/2024_08_26/11_12_11'
                
model_clf =AutoModelForSeq2SeqLM.from_pretrained(model_id2)
tokenizer = AutoTokenizer.from_pretrained(model_id2)
#%%
import torch
summary(model_clf, (1024,), dtypes = [torch.long])
# %%
# dummy_input = torch.ones(input_size, dtype=torch.long).unsqueeze(0).to(device)

# 요약 정보 출력 (torchsummary 사용)
summary(model_clf, input_size=(1024,), dtypes=[torch.long])
# %%
batch_size = 1  # 요약 정보 출력용 배치 크기
seq_length = 1024  # 입력 시퀀스 길이

dummy_input = torch.ones((batch_size, seq_length), dtype=torch.long).to('cpu')

# 요약 정보 출력 (torchsummary 사용)
summary(model_clf, input_size=(1024,))
# %%
# 각 파라미터의 크기와 총 파라미터 개수 계산
total_params = 0
param_size_in_bytes = 0

for param in model_clf.parameters():
    num_params = param.numel()  # 파라미터 개수
    total_params += num_params
    param_size_in_bytes += num_params * 4  # float32(4 bytes)로 계산

# MB 단위로 변환 (1MB = 1024 * 1024 bytes)
param_size_in_mb = param_size_in_bytes / (1024 ** 2)

print(f"Total number of parameters: {total_params}")
print(f"Total parameter size: {param_size_in_mb:.2f} MB")
# %%
'''
Total number of parameters: 737,668,096
Total parameter size: 2813.98 MB

Total number of parameters: 1,318,914
Total parameter size: 5.03 MB
'''
#%%

'asdf answr asss'.split('answer')[:1]
# %%
