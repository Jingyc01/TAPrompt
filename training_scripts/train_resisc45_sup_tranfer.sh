# for seed in 42 
# for delta_weight_value in 0.8 0.6 0.4 0.2 0.1

# do
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
# 	--nproc_per_node=1 \
# 	--master_port='29413' \
# 	--use_env main.py \
# 	resisc45_taprompt_transfer \
# 	--model vit_base_patch16_224 \
# 	--batch-size 24 \
# 	--epochs 50 \
# 	--data-path /data2/jingyc/clproject/l2p-pytorch/local_datasets/ \
# 	--ca_lr 0.005 \
# 	--crct_epochs 30 \
# 	--seed $seed \
# 	--length 5 \
# 	--sched step \
# 	--e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
# 	--larger_prompt_lr \
# 	--ca_storage_efficient_method covariance \
# 	--penalty_weight 0 \
# 	--delta_weight $delta_weight_value \
# 	--output_dir ./output/resisc45_sup_transfer_dwv$delta_weight_value_epoch50$seed  \
# 	--trained_taprompt_model ./output/resisc45_sup_transfer_warmup5_epoch5042 \
# 	--leep_hard_selection True \
# 	--leep_warmup_batches 5 \
# 	--num_tasks 5\
# 	--size 5\
# 	# --eval \
# 	# --num_tasks 20\
# 	# --size 20\
# 	# --use_task_embedding_bias \

# 	# --lr 0.025 \

# done
for seed in 42; do
# for delta_weight_value in 0.8 0.6 0.4 0.2 0.1; do
for delta_weight_value in 1.0 0.2 0.1; do
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port='29414' \
    --use_env main.py \
    resisc45_taprompt_transfer \
    --model vit_base_patch16_224 \
    --batch-size 24 \
    --epochs 50 \
    --data-path /data2/jingyc/clproject/l2p-pytorch/local_datasets/ \
    --ca_lr 0.005 \
    --crct_epochs 30 \
    --seed $seed \
    --length 5 \
    --sched step \
    --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
    --larger_prompt_lr \
    --ca_storage_efficient_method covariance \
    --penalty_weight 0.2 \
    --delta_weight $delta_weight_value \
    --output_dir ./output/resisc45_sup_transfer2_pw0.2_dwv${delta_weight_value}_epoch50_seed$seed \
    --trained_taprompt_model ./output/resisc45_sup_transfer2_warmup5_epoch5042 \
    --leep_hard_selection True \
    --leep_warmup_batches 5 \
    --num_tasks 5 \
    --size 5

done
done