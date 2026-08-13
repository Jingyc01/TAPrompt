# for seed in 42 40 44
for seed in 42 
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29451' \
	--use_env main.py \
	cub_taprompt \
	--model vit_base_patch16_224 \
	--batch-size 24 \
	--epochs 50 \
	--data-path /data2/jingyc/clproject/l2p-pytorch/local_datasets/ \
	--lr 0.06 \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--seed $seed \
	--larger_prompt_lr \
	--ca_storage_efficient_method covariance \
	--e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 \
	--penalty_weight 0.2 \
	--delta_weight 5 \
	--length 5 \
	--output_dir ./output/cub_sup_analysis_seed$seed \
	--trained_taprompt_model ./output/cub_sup_seed42 \
	--leep_hard_selection True \
	--eval \
	# --num_tasks 15 \
	# --size 15
	
done

