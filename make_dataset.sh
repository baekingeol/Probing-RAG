python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name trivia --is_cot --sep_number 0
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name hotpotqa --is_cot --sep_number 0
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name nq --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name trivia --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name hotpotqa --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --steps_limit 3200 --dataset_name nq --is_cot --sep_number 0