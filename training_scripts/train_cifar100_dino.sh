for seed in 42
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29402' \
	--use_env main.py \
	cifar100_taprompt \
	--model vit_base_patch16_224_dino \
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
	--delta_weight 0.1 \
	--output_dir ./output/cifar100_dino_seed$seed  \
	--trained_taprompt_model ./output/cifar100_dino_seed42 \
	# --eval
	
done