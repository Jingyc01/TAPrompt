for seed in 40 
#epochs 50-》1
#crct_epochs 30-》1
# --penalty_weight 0.2 \
# for seed in 42 
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29411' \
	--use_env main.py \
	cifar100_taprompt \
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
	--penalty_weight 0 \
	--delta_weight 1.0 \
	--output_dir ./output/cifar100_sup_test$seed  \
	--trained_taprompt_model ./output/cifar100_sup_v9-3_linear0_seed42 \
	--leep_hard_selection True \
	--num_tasks 5\
	--size 10\
	# --eval \
	# --num_tasks 20\
	# --size 20\
	# --use_task_embedding_bias \

	# --lr 0.025 \

done