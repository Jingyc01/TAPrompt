
for seed in 42
do
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29502' \
	--use_env main.py \
        imr_taprompt \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 1 \
        --data-path /data2/jingyc/clproject/l2p-pytorch/local_datasets/ \
        --ca_lr 0.005 \
        --crct_epochs 1 \
	--sched constant \
        --seed $seed \
	--length 10 \
        --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 \
        --penalty_weight 0.2 \
        --larger_prompt_lr \
        --ca_storage_efficient_method covariance \
        --delta_weight 5 \
	--output_dir ./output/imr_sup_v9-3_seed$seed  \
        --trained_taprompt_model ./output/imr_sup_v9-3_seed42 \
        --num_tasks 10 \
        --size 10 \
        --eval 
done



