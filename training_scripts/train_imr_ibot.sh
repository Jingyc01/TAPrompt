#100 30

for seed in 42
do
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29505' \
	--use_env main.py \
        imr_taprompt \
        --model vit_base_patch16_224_ibot \
        --batch-size 24 \
        --epochs 100 \
        --data-path /data2/jingyc/clproject/l2p-pytorch/local_datasets/  \
        --ca_lr 0.005 \
        --crct_epochs 30 \
	--sched constant \
        --seed $seed \
	--length 10 \
        --prompt_momentum 0.0 \
        --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 \
        --penalty_weight 0.2 \
        --larger_prompt_lr \
        --ca_storage_efficient_method covariance \
        --delta_weight 1 \
	--output_dir ./output/imr_ibot_seed$seed  \
        --trained_taprompt_model ./output/imr_ibot_seed42 \
        --eval
        # --prompt_constran linear \
done





