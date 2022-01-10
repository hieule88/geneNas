python main_ner_multi_obj.py\
  --task_name "ner" \
  --max_epochs 1 \
  --gpus 0 \
  --popsize 10 --num_iter 100 \
  --train_batch_size 1024 \
  --eval_batch_size 1024 \
  --h_main 6 --h_adf 4 \
  --save_dict_path ner.gene_nas.h_main6_h_adf_4_addDistribute.pkl
