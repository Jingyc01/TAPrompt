from re import T
import torch
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import time, datetime, os, sys, random, numpy as np
from datasets import build_continual_dataloader
from TAPrompt.engines.tapromtp_engine import train_and_evaluate, evaluate_till_now, analyze_three_weights, visualize_weight_analysis
import vits.taprompt_vision_transformer as taprompt_vision_transformer
import utils
import json
import cv2 
import matplotlib.pyplot as plt 
import torch.nn.functional as F 
from visualization_utils import TaskSimilarityVisualizer, plot_similarity_matrix_from_array
def train(args):
    device = torch.device(args.device)
    
    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)

    if not hasattr(args, 'class_per_task_list'):
        args.class_per_task_list = [len(class_mask[t]) for t in range(args.num_tasks)]
        args.class_mask = class_mask
    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        args = args,
        use_task_embedding=True,
        use_task_embedding_bias=args.use_task_embedding_bias,
        # use_task_embedding_bias=True,
        num_tasks=args.num_tasks,
    )
    model.to(device)

    reference_model = create_model(
            args.original_model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            args = args,
            num_tasks=args.num_tasks,
        )
    reference_model.to(device)
    reference_model.eval()

    if args.freeze:

        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)
    
    if args.eval:
        # 在评估开始时生成一个唯一的文件名
        import random

        random_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        log_filename = f'test_stats_{random_id}.txt'
        log_filepath = os.path.join(args.output_dir, log_filename)
        
        with open(log_filepath, 'w') as f:
            f.write(f"Evaluation started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random ID: {random_id}\n")
            f.write("="*80 + "\n\n")
        
        reference_model = create_model(
                args.original_model,
                pretrained=args.pretrained,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                num_tasks=args.num_tasks,
            )
        reference_model.to(device)
        reference_model.eval()
        acc_ref_matrix = np.zeros((args.num_tasks, args.num_tasks))

        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        acc_matrix_cyc1 = np.zeros((args.num_tasks, args.num_tasks))
        acc_matrix_accn = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            if args.ckpt_num > 0 :
                task_id = args.ckpt_num-1
            checkpoint_path = os.path.join(args.trained_taprompt_model, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if task_id == 0:
                ref_checkpoint_path = checkpoint_path
            else:
                ref_checkpoint_path = os.path.join(args.trained_taprompt_model, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            if os.path.exists(ref_checkpoint_path):
                print('Loading ref_checkpoint from:', ref_checkpoint_path)
                ref_checkpoint = torch.load(ref_checkpoint_path, map_location=device)
                reference_model.load_state_dict(ref_checkpoint['model'],strict=False)
            else:
                print('No ref_checkpoint found at:', ref_checkpoint_path)
                return
          
            test_stats, result_str = evaluate_till_now(model, data_loader, device,
                                  task_id, class_mask, target_task_map, acc_matrix, acc_matrix_cyc1, acc_ref_matrix, args,acc_matrix_accn, reference_model=reference_model)
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                        }
            
            if args.output_dir and utils.is_main_process():
                
                with open(log_filepath, 'a') as f:
                    f.write(f"\n--- Task {task_id + 1} ---\n")
                    f.write(json.dumps(log_stats) + '\n')
                    f.write(result_str + '\n')  
        if args.output_dir and utils.is_main_process():
            with open(log_filepath, 'a') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Evaluation completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print(f'Number of params: {n_parameters / 1e6}M')
    


    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    if args.larger_prompt_lr:
        base_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' in name and p.requires_grad == True]
        base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' not in name and p.requires_grad == True]
        base_params_name = [name for name, p in model_without_ddp.named_parameters() if 'prompt' in name and p.requires_grad == True]
        base_fc_params_name = [name for name, p in model_without_ddp.named_parameters() if 'prompt' not in name and p.requires_grad == True]
        base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
        network_params = [base_params, base_fc_params]
        # import ipdb; ipdb.set_trace()
        optimizer = create_optimizer(args, network_params)
    else:
        optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp,
                       criterion, data_loader, data_loader_per_cls,
                       optimizer, lr_scheduler, device, class_mask, target_task_map, args,reference_model=reference_model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    

