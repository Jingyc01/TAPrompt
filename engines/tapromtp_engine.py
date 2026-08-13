"""
Train and eval functions used in main.py
"""
import math
from re import T
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import copy
import torch
import torch.distributed as dist
import numpy as np
from torch.autograd import Variable
from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import math
import time
import cv2
import matplotlib.pyplot as plt



def linear_constraint(prompt,task_id,args):
    layer_num,kv,pool_size,length,head,dim = prompt.shape
    prompt = prompt.reshape(layer_num,kv,pool_size,length,-1)
    prompt_0 = prompt[:,:,0].detach().clone()
    prompt_before = prompt[:,:,task_id-1].detach().clone()
    prompt_this = prompt[:,:,task_id]
    delta_1 = prompt_before - prompt_0
    delta_2 = prompt_this - prompt_0
    sim = F.cosine_similarity(delta_1,delta_2,dim=-1)
    loss_sim = (-sim +1) 
    return torch.mean(loss_sim)

def l2_normalize(x, dim=None, epsilon=1e-12):
    """Normalizes a given vector or matrix."""
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm

def train_one_epoch(model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None,reference_model: torch.nn.Module=None ):
    model.train(set_training_mode)
    if args.dataset == 'Split-RESISC45-transfer':
        class_per_task = 3
    else:
        class_per_task = args.nb_classes // args.num_tasks
    
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    loss_con = torch.zeros(1)
    proto_ortho_loss = torch.zeros(1)
    Loss_sim = torch.zeros(1).to(device) 
    warmup_batches = args.leep_warmup_batches if hasattr(args, 'leep_warmup_batches') else 5
    batch_count = 0
    leep_accumulator = None
    # ✅ 读取硬选择配置
    use_hard_selection = args.leep_hard_selection if hasattr(args, 'leep_hard_selection') else True
    use_hard_selection = True
    leep_expanded = None

    for batch_idx, (input, target) in enumerate(data_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if task_id>0:
            with torch.no_grad(): 

                prompt_weight_0 = torch.ones(input.shape[0],args.num_tasks,class_per_task).cuda() #24,10,10
                prompt_weight_0 = torch.mean(prompt_weight_0,dim=-1)
                prompt_weight_0[:,task_id+1:] = 0

                prompt_weight_0 = norm(prompt_weight_0)
                output = model(input, task_id=task_id, train=set_training_mode,prompt_weight=prompt_weight_0, compute_similarity=False)                                 
                logits = output['logits'] # 24,100
                # ✅ 关键修改：立即 detach 并 clone
                # prompt_weight = F.softmax(logits.detach(), dim=-1)  # ← detach 断开计算图
                prompt_weight_0 = F.softmax(logits, dim=-1)  # ← 不要 detach 断开计算图
                prompt_weight_0 = prompt_weight_0.reshape(input.shape[0], -1, class_per_task)
                prompt_weight_0 = torch.mean(prompt_weight_0, dim=-1)
                prompt_weight_0[:, task_id+1:] = 0
                prompt_weight_0 = norm(prompt_weight_0)
                # import ipdb;ipdb.set_trace()
                prompt_weight_logit = prompt_weight_0.clone()
                # ✅ 使用双阶段累积策略
                # import ipdb;ipdb.set_trace()
                leep_scores, leep_weights, leep_accumulator, batch_count, is_stable = compute_leep_with_two_stages(
                    logits, target, task_id, class_per_task, device,
                    leep_accumulator=leep_accumulator,
                    batch_count=batch_count,
                    warmup_batches=warmup_batches,
                    update_interval=1
                )
                # ✅ 关键修改：在这里对 leep_weights 进行硬选择转换
                prompt_weight_leep = leep_weights.clone()  # 先克隆原始权重
                # import ipdb;ipdb.set_trace()
                if is_stable and use_hard_selection:
                    # 硬选择：找到 LEEP 最高的任务，只保留它的权重为 1，其余为 0
                    max_leep_task = torch.argmax(prompt_weight_leep).item()
                    prompt_weight_leep_hard = torch.zeros_like(prompt_weight_leep)
                    prompt_weight_leep_hard[max_leep_task] = 1.0
                    prompt_weight_leep = prompt_weight_leep_hard  # 使用硬选择后的权重
                # 组合权重
                if is_stable:
                    if use_hard_selection:
                        # ✅ 硬选择模式：对 LEEP 权重进行硬选择，然后叠加到 logits 权重上
                        # 第一步：对 LEEP 权重进行硬选择（只保留最大值，其余为 0）
                        max_leep_task = torch.argmax(prompt_weight_leep).item()
                        prompt_weight_leep_hard = torch.zeros_like(prompt_weight_leep)
                        prompt_weight_leep_hard[max_leep_task] = 1.0  # 只有最高 LEEP 的任务权重为 1

                        leep_expanded = torch.zeros(input.shape[0], args.num_tasks, device=prompt_weight_leep_hard.device)
                        # 填充前 task_id+1 个位置
                        leep_expanded[:, :task_id+1] = prompt_weight_leep_hard.unsqueeze(0)       

                
                prompt_weight_new = prompt_weight_logit.clone()
                prompt_weight_new[:, :task_id] = 0
                prompt_weight_new[:, task_id+1:] = 0
                prompt_weight_new = norm(prompt_weight_new)


            #第二次推理
            if reference_model is not None:
                output_ref = reference_model(input)
                cls_features = output_ref['pre_logits'] #cls_token or avg pooled features    
            # import ipdb;ipdb.set_trace()
            output_new = model(input, task_id=task_id,prompt_weight=prompt_weight_new,train=set_training_mode,compute_similarity=False)#仅当前任务prompt 的输出
            output = model(input, task_id=task_id, train=set_training_mode,prompt_weight=prompt_weight_logit,cls_features=cls_features,compute_similarity=True,target=target,leep_expanded=leep_expanded) #混合当前任务+过去任务prompt 的输出

            with torch.no_grad(): 
                # 融合prototype 和leep
                if 'task_similarity_loss' in output:
                    # import ipdb;ipdb.set_trace()
                    prompt_weight = output['prompt_weight_sum']
                else:
                    prompt_weight = output['prompt_weight']
                prompt_weight_old_sum = torch.sum(prompt_weight[:, :task_id], dim=-1)
                prompt_weight_new_sum = prompt_weight[:, task_id] 
                prompt_weight_old = prompt_weight.clone()
                prompt_weight_old[:, task_id:] = 0
                prompt_weight_old = norm(prompt_weight_old)
                output_old = model(input, task_id=task_id, train=set_training_mode, prompt_weight=prompt_weight_old)
                logits_old = output_old['logits']

               # ✅ 打印权重对比
                # if batch_idx % 50 == 0:
                #     print(f"\n--- Batch {batch_idx} Weights ---")
                #     print(f"LEEP weights: {prompt_weight_leep.tolist()}")
                #     print(f"Logits weights (sample): {prompt_weight_logit[0, :task_id+1].tolist()}")
                #     print(f"Final weights (sample): {prompt_weight[0, :task_id+1].tolist()}")
                # '''-----''' 
            #混合当前任务+过去任务prompt
            logits_mix = output['logits_detach']
            logits = output['logits']

            #仅当前任务prompt_t
            logits_new_mix = output_new['logits_detach']
            logits_new = output_new['logits']
            #print(prompt_weight_new_sum+prompt_weight_old_sum)
            logits_mix[:,:task_id*class_per_task] = logits_mix[:,:task_id*class_per_task].detach().clone()
            logits_new_mix[:,:task_id*class_per_task] = logits_new_mix[:,:task_id*class_per_task].detach().clone()
            
            prob = F.softmax(logits_mix,dim=-1)#混合当前任务+过去任务prompt
            prob_old = F.softmax(logits_old,dim=-1).detach().clone()#仅过去任务prompt
            prob_new = F.softmax(logits_new_mix,dim=-1)#.detach().clone()#仅当前任务prompt
            select_index = torch.arange(prob_new.shape[0])
            delta = prob_new[select_index,target] * prompt_weight_new_sum \
                + prob_old[select_index,target] * prompt_weight_old_sum \
                - prob[select_index,target]
            delta[delta<0]=0
            loss_con = torch.mean(delta)*args.delta_weight
            # import ipdb;ipdb.set_trace()

            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                logits_new = logits_new.index_fill(dim=1, index=not_mask, value=float('-inf'))
            
            loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
            # 添加任务原型损失
            if 'task_similarity_loss' in output:
                
                Loss_sim = 0.2 * (1 - output['task_similarity_loss']) #v8-1
                # Loss_sim = 0.2 * output['task_similarity_loss'] #v8-2
                loss = loss + Loss_sim
            loss = loss + loss_con
            
        else : 

            prompt_weight = torch.ones(input.shape[0],args.num_tasks,class_per_task).cuda()
            prompt_weight = torch.mean(prompt_weight,dim=-1)
            prompt_weight[:,task_id+1:] = 0

            prompt_weight = norm(prompt_weight)
            
            output = model(input, task_id=task_id, train=set_training_mode,prompt_weight=prompt_weight)
            logits = output['logits']
            
            # ========== Mask 处理 ==========
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)

                logits = logits.clone()
                logits.index_fill_(dim=1, index=not_mask, value=float('-inf'))  # 使用 index_fill_
                
            # ========== 损失计算 ==========
            loss = criterion(logits, target)
            if 'task_similarity_loss' in output:
                Loss_sim = 0.2*(1 - output['task_similarity_loss'])#v8-1
                # Loss_sim = 0.2 * output['task_similarity_loss'] #v8-2
                loss = loss + Loss_sim

            

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        if task_id >= 2 :

            #loss_linear =  linear_constraint(model.e_prompt.prompt,task_id,args) * args.penalty_weight
            if args.distributed:
                loss_linear =  model.module.linear_constrain(task_id,args) * args.penalty_weight
            else :
                loss_linear =  model.linear_constrain(task_id,args) * args.penalty_weight
            loss += loss_linear
        else :
            loss_linear = torch.zeros_like(loss)
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        # import ipdb;ipdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Loss_sim=Loss_sim)
        # metric_logger.update(Loss_ortho=proto_ortho_loss.item())
        metric_logger.update(Loss_l=loss_linear.item())
        metric_logger.update(Loss_con=loss_con.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def norm(feature):
    """
    归一化函数，确保不破坏计算图
    """
    feature_norm = torch.sum(feature, dim=-1, keepdim=True)
    feature_norm = torch.clamp(feature_norm, min=1e-10)
    
    # ✅ 显式创建新 tensor
    normalized = feature / feature_norm
    
    return normalized

def prompt_id_from_logit (logits,args,task_id,class_mask,device,target_task_map):
    if args.train_mask and class_mask is not None:
        mask = []
        for id in range(task_id + 1):
            mask.extend(class_mask[id])
        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
    prompt_id = torch.max(logits, dim=1)[1]
    # translate cls to task_id
    prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
        -1)
    return prompt_id

def compute_leep_alignment_loss(logits, leep_weights, target, task_id, class_per_task):
    """
    计算 LEEP 对齐损失：鼓励模型在高 LEEP 任务上有更好的预测
    
    核心思想：如果 LEEP 说任务 i 相关性高，那么任务 i 的 logits 应该有更高的置信度
    
    ⚠️ 修改：移除 alignment_loss 参数，内部初始化
    """
    device = logits.device
    batch_size = logits.shape[0]
    current_task_start = task_id * class_per_task
    
    # ✅ 在函数内部初始化，而不是接收外部参数
    alignment_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 获取每个旧任务的预测概率
    for old_task_id in range(task_id + 1):
        old_start = old_task_id * class_per_task
        old_end = (old_task_id + 1) * class_per_task
        old_logits = logits[:, old_start:old_end]
        old_probs = F.softmax(old_logits, dim=-1)
        
        # 使用 LEEP 权重作为"软标签"的置信度
        leep_weight = leep_weights[old_task_id]
        
        # 计算旧任务预测的不确定性（熵）
        entropy = -(old_probs * torch.log(old_probs + 1e-10)).sum(dim=-1).mean()
        
        # LEEP 高 → 鼓励低熵；LEEP 低 → 不惩罚
        alignment_loss = alignment_loss + leep_weight * entropy
    
    return alignment_loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader,
             device, task_id=-1, eval_task_id=-1, reference_model: torch.nn.Module=None, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    model.eval()

    
    if reference_model is not None:
        reference_model.eval()
    class_per_task = args.nb_classes // args.num_tasks


    with torch.no_grad():

        for batch_idx, (input, target) in enumerate(data_loader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            start_time = time.time()

            if reference_model is not None:
                reference_output = reference_model(input)
                reference_logits = reference_output['logits']  
                cls_features = reference_output['pre_logits'] #cls_token or avg pooled features                    

            prompt_weight = torch.ones(input.shape[0],args.num_tasks,class_per_task).cuda()
            prompt_weight = torch.mean(prompt_weight,dim=-1)
            prompt_weight[:,task_id+1:] = 0
            prompt_weight = norm(prompt_weight)

            output = model(input,prompt_weight=prompt_weight)
            logits = output['logits']
            prompt_id2 = prompt_id_from_logit(logits,args,task_id,class_mask,device,target_task_map)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            metric_logger.meters['Acc@cyc1'].update(acc1.item(), n=input.shape[0])
            task_inference_acc = utils.task_inference_accuracy(prompt_id2, target, target_task_map)
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])
            
            prompt_weight2 = F.softmax(logits,dim=-1)
            prompt_weight2 = prompt_weight2.reshape(input.shape[0],-1,class_per_task)
            prompt_weight2 = torch.mean(prompt_weight2,dim=-1)
            prompt_weight2[:,task_id+1:] = 0

            prompt_weight2 = norm(prompt_weight2)

            output2 = model(input, prompt_weight=prompt_weight2, task_id = task_id, cls_features=cls_features,compute_similarity=True)

            logits2 = output2['logits']
            acc1, acc5 = accuracy(logits2, target, topk=(1, 5))
            metric_logger.meters['Acc@cyc2'].update(acc1.item(), n=input.shape[0])
            if args.cycle_num > 2:
                logitsn = logits2
                for _ in range(args.cycle_num-2) :  
                    prompt_weight = F.softmax(logitsn,dim=-1)
                    prompt_weight = prompt_weight.reshape(input.shape[0],-1,class_per_task)
                    prompt_weight = torch.mean(prompt_weight,dim=-1)
                    prompt_weight[:,task_id+1:] = 0

                    prompt_weight = norm(prompt_weight)
                    outputn = model(input, prompt_weight=prompt_weight)
                    logitsn = outputn['logits']
                acc1, acc5 = accuracy(logitsn, target, topk=(1, 5))
            metric_logger.meters['Acc@cycn'].update(acc1.item(), n=input.shape[0])
            #add by jyc for FWT
            if reference_model is not None:
                acc1_ref, acc5_ref = accuracy(reference_logits, target, topk=(1, 5))
                metric_logger.meters['Acc_ref@1'].update(acc1_ref.item(), n=input.shape[0])
                metric_logger.meters['Acc_ref@5'].update(acc5_ref.item(), n=input.shape[0])


    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task.global_avg:.3f} Acc@cyc1 {cyc1.global_avg:.3f} Acc@pw2 {cyc2.global_avg:.3f} Acc@pwn {cycn.global_avg:.3f}'
        .format(task=metric_logger.meters['Acc@task'],cyc1=metric_logger.meters['Acc@cyc1'], cyc2=metric_logger.meters['Acc@cyc2'], cycn=metric_logger.meters['Acc@cycn']))
    if reference_model is not None:
        print(
            '* Acc_ref@1 {top1_ref.global_avg:.3f} Acc_ref@5 {top5_ref.global_avg:.3f}'
            .format(top1_ref=metric_logger.meters['Acc_ref@1'], top5_ref=metric_logger.meters['Acc_ref@5']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None,acc_matrix_cyc1=None, acc_matrix_ref=None, args=None, acc_matrix_accn=None, reference_model: torch.nn.Module=None,):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
    start_time = time.time()

    for i in range(task_id + 1):

        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'],
                              device=device,  task_id=task_id, reference_model=reference_model, eval_task_id=i, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)


        stat_matrix[0, i] = test_stats['Acc@cyc1']
        stat_matrix[1, i] = test_stats['Acc@cyc2']
        stat_matrix[2, i] = test_stats['Acc@task']
        stat_matrix[3, i] = test_stats['Acc@cycn']

        acc_matrix[i, task_id] = test_stats['Acc@cyc2'] #上三角
        acc_matrix_cyc1[i, task_id] = test_stats['Acc@cyc1'] #上三角

        acc_matrix_accn[i, task_id] = test_stats['Acc@cycn']
        
        if acc_matrix_ref is not None and reference_model is not None:
            acc_matrix_ref[i, task_id] = test_stats['Acc_ref@1']
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=(total_time)))
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    diagonal = np.diag(acc_matrix)
    diagonal_cyc1 = np.diag(acc_matrix_cyc1)
    diagonal_cycn = np.diag(acc_matrix_accn)
    if acc_matrix_ref is not None and reference_model is not None:

        diagonal_ref = np.diag(acc_matrix_ref)
    
    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@cyc1: {:.4f}\tAcc@cyc2: {:.4f}\tAcc@cycn: {:.4f}".format(
        task_id + 1,
        avg_stat[2],
        avg_stat[0],
        avg_stat[1],
        avg_stat[3],)
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        forgetting_accn = np.mean((np.max(acc_matrix_accn, axis=1) -
                              acc_matrix_accn[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id]) #负向迁移 
        acc_all = 0 
        for t in range(task_id+1) :
            acc_all += (acc_matrix[:, t][:t+1]).mean()
        acc_all = acc_all / (task_id+1)
        if acc_matrix_ref is not None:
            diagonal_cyc2 = diagonal
            forwardcyc2 = np.mean((diagonal_cyc2 - diagonal_ref)[:task_id])# 新增正向迁移
            forwardcyc1 = np.mean((diagonal_cyc1 - diagonal_ref)[:task_id])# 新增正向迁移
            forwardcycn = np.mean((diagonal_cycn - diagonal_ref)[:task_id])# 新增正向迁移
            result_str += "\tForward_cycn: {:.4f}".format(forwardcycn)
            result_str += "\tForward_cyc2: {:.4f}".format(forwardcyc2)
            result_str += "\tForward_cyc1: {:.4f}".format(forwardcyc1)
        result_str += "\tForgetting: {:.4f}\tAFn: {:.4f}\tBackward: {:.4f}\tAAC: {:.4f}".format(forgetting,forgetting_accn ,backward,acc_all)
    print(result_str)
    # import ipdb;ipdb.set_trace()

    return test_stats, result_str


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, 
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, reference_model: torch.nn.Module=None,):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    acc_matrix_ref = np.zeros((args.num_tasks, args.num_tasks))
    acc_matrix_cyc1 = np.zeros((args.num_tasks, args.num_tasks))
    acc_matrix_accn = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix_accn = np.zeros((args.num_tasks, args.num_tasks))
    #存储每个类的统计信息
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()
    # 在评估开始时生成一个唯一的文件名
    import random
    # random_id = random.randint(10000, 99999)  # 5位随机数
    # 或者使用时间戳 + 随机数
    
    random_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    log_filename = f'train_stats_{random_id}.txt'
    log_filepath = os.path.join(args.output_dir, log_filename)
    # 写入文件头信息（可选）
    with open(log_filepath, 'w') as f:
        f.write(f"Evaluation started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random ID: {random_id}\n")
        f.write("="*80 + "\n\n")
    for task_id in range(args.num_tasks):
        # task_id = 9
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            if args.larger_prompt_lr:
                # This is a simple yet effective trick that helps to learn task-specific prompt better.
                base_params = [p for name, p in model_without_ddp.named_parameters() if
                            'prompt' in name and p.requires_grad == True]
                base_fc_params = [p for name, p in model_without_ddp.named_parameters() if
                                'prompt' not in name and p.requires_grad == True]
                base_fc_params_name = [name for name, p in model_without_ddp.named_parameters() if
                                'prompt' not in name and p.requires_grad == True]
                base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
                base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
                network_params = [base_params, base_fc_params]
                optimizer = create_optimizer(args, network_params)
            else:
                optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # if model already trained
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        if task_id < args.ckpt_num :
            if args.ckpt_num>0 and task_id < args.ckpt_num-1 :
                continue 
            resume = True
            load_path = os.path.join(args.trained_taprompt_model, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))

            if os.path.exists(load_path):
                print('Loading checkpoint from:', load_path)
                checkpoint = torch.load(load_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', load_path)
                return
            
        else :
            resume = False

            if args.prompt_pool and args.shared_prompt_pool :
                if task_id > 0:
                    prev_start = (task_id - 1) * args.top_k
                    prev_end = task_id * args.top_k

                    cur_start = prev_end
                    cur_end = (task_id + 1) * args.top_k

                    if (prev_end > args.size) or (cur_end > args.size):
                        pass
                    else:
                        cur_idx = (
                            slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(cur_start, cur_end))
                        prev_idx = (
                            slice(None), slice(None),
                            slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                            slice(None), slice(prev_start, prev_end))

                        with torch.no_grad():
                            if args.distributed:
                                model.module.e_prompt.prompt.grad.zero_()
                                model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                                
                                # optimizer.param_groups[0]['params'] = model.module.parameters()
                            else:
                                if model.e_prompt.prompt.grad != None :
                                    model.e_prompt.prompt.grad.zero_()
                                model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                                # optimizer.param_groups[0]['params'] = model.parameters()
                    init_method = args.task_embedding_init_method if hasattr(args, 'task_embedding_init_method') else 'copy'
                    if args.distributed:
                        model.module.initialize_task_embedding_from_previous(task_id, init_method=init_method)
                    else:
                        model.initialize_task_embedding_from_previous(task_id, init_method=init_method)
            # ✅ 冻结之前任务的原型
            if task_id > 0:
                if args.distributed:
                    model.module.freeze_previous_task_embeddings(task_id)
                else:
                    model.freeze_previous_task_embeddings(task_id)
            for epoch in range(args.epochs):
                # import ipdb;ipdb.set_trace()
                start_time = time.time()
                train_stats = train_one_epoch(model=model, criterion=criterion,
                                                data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                                device=device, epoch=epoch, max_norm=args.clip_grad,
                                                set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                                target_task_map=target_task_map, args=args, reference_model=reference_model, )
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                # print(f"Total training one epoch time: {total_time_str}")
                # import ipdb;ipdb.set_trace()
                if lr_scheduler:
                    lr_scheduler.step(epoch)

            if args.prompt_momentum > 0 and task_id > 0: #false
                if args.use_prefix_tune_for_e_prompt:
                    with torch.no_grad():
                        print(model.e_prompt.prompt[:, :, task_id].shape)
                        print(
                            model.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True).shape)
                        model.e_prompt.prompt[:, :, task_id].copy_(
                            (1 - args.prompt_momentum) * model.e_prompt.prompt[:, :, task_id].detach().clone()
                            + args.prompt_momentum * model.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(
                                dim=2))


        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args,reference_model=reference_model)

        if task_id > 0 and not args.not_train_ca and not resume:

            train_task_adaptive_prediction(model, args, device, class_mask, task_id)

        if reference_model is not None:
            print("Loading model parameters to reference_model...")

            reference_model.load_state_dict(model_without_ddp.state_dict(), strict=False)
            reference_model.eval()  # 设置为评估模式

            for param in reference_model.parameters():
                param.requires_grad = False
            print("Reference model loaded and frozen.")
        test_stats,result_str = evaluate_till_now(model=model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args,acc_matrix_ref=acc_matrix_ref, acc_matrix_cyc1=acc_matrix_cyc1,acc_matrix_accn=acc_matrix_accn, reference_model=reference_model,)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)
        if not resume:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
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

@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, reference_model: torch.nn.Module=None,):
    model.eval()

    class_per_task = args.nb_classes // args.num_tasks
    

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            prompt_weight = torch.ones(inputs.shape[0],args.num_tasks,class_per_task).cuda()
            prompt_weight = torch.mean(prompt_weight,dim=-1)
            prompt_weight[:,:task_id] = 0
            prompt_weight[:,task_id+1:] = 0
            prompt_weight = norm(prompt_weight)#已知task id 
            features = model(inputs, task_id=task_id,prompt_weight=prompt_weight, train=True)['pre_logits']
       
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]
        try :
            dist.barrier()
            dist.all_gather(features_per_cls_list, features_per_cls)
        except Exception as e :
            features_per_cls_list = [features_per_cls]

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars

def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n] #训练prompt以外的所有可训练参数
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                       sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * num_sampled_pcls)
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        #print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()



def get_leep_score(predictions, target_labels, num_target_classes=None):
    """
    计算 LEEP (Log Expected Empirical Prediction) 分数
    
    Args:
        predictions: tensor [N, S] 源任务模型在目标数据上的预测概率
                    N = 样本数, S = 源任务类别数
        target_labels: tensor [N] 目标数据的真实标签
        num_target_classes: int, 目标任务的类别数（如果为None则自动推断）
    
    Returns:
        leep: float, 可迁移性分数（越高越好）
    """
    device = predictions.device
    N, S = predictions.shape
    
    # 获取目标类别数
    if num_target_classes is None:
        num_target_classes = int(target_labels.max().item()) + 1
    
    # One-hot 编码目标标签 [N, T]
    one_hot_target = F.one_hot(target_labels.long(), num_classes=num_target_classes).float()  # [N, T]
    
    # 计算联合出现次数 [S, T]
    # occurrences_s_t[s, t] = sum_n(predictions[n, s] * one_hot_target[n, t])
    occurrences_s_t = torch.einsum('ns,nt->st', predictions, one_hot_target)  # [S, T]
    
    # 计算每个源类别的总预测量 [S, 1]
    occurrences_s = occurrences_s_t.sum(dim=-1, keepdim=True)  # [S, 1]
    
    # 计算条件概率 P(target | source) [S, T]
    # 避免除零
    probability_t_given_s = occurrences_s_t / (occurrences_s + 1e-10)  # [S, T]
    
    # 计算目标预测 [N, T]
    target_predictions = torch.matmul(predictions, probability_t_given_s)  # [N, T]
    
    # 计算 EEP (Expected Empirical Prediction)
    eep = (target_predictions * one_hot_target).sum(dim=-1)  # [N]
    
    # 计算 LEEP
    # 对于 eep > 0 的样本取 log，否则使用随机猜测的对数
    random_baseline = -torch.log(torch.tensor(float(num_target_classes), device=device))
    leep = torch.where(
        eep > 0,
        torch.log(eep + 1e-10),
        random_baseline
    ).mean()
    
    return leep.item()


def compute_leep_with_two_stages(logits, target, task_id, class_per_task, device,
                                  leep_accumulator=None, batch_count=0, 
                                  warmup_batches=5, update_interval=1):
    """
    双阶段 LEEP 计算策略
    
    阶段 1 (Warmup): 前 warmup_batches 个 batch，只累积不计算权重，使用均匀权重
    阶段 2 (Stable): 之后每 update_interval 个 batch 更新一次权重
    
    Args:
        logits: [batch_size, num_classes]
        target: [batch_size]
        task_id: 当前任务 ID
        class_per_task: 每个任务的类别数
        device: 设备
        leep_accumulator: 累积器（同方案1）
        batch_count: 当前 batch 计数
        warmup_batches: 预热阶段的 batch 数量
        update_interval: 更新间隔
    
    Returns:
        leep_scores: dict
        leep_weights: tensor
        updated_accumulator: dict
        updated_batch_count: int
        is_stable: bool  # 是否已进入稳定阶段
    """
    current_task_start = task_id * class_per_task
    local_target = target - current_task_start
    
    if leep_accumulator is None:
        leep_accumulator = {}
    
    # 累积统计量（与方案1相同）
    one_hot_target = F.one_hot(local_target.long(), num_classes=class_per_task).float()
    
    for old_task_id in range(task_id + 1):
        old_start = old_task_id * class_per_task
        old_end = (old_task_id + 1) * class_per_task
        old_logits = logits[:, old_start:old_end]
        old_probs = F.softmax(old_logits, dim=-1)
        
        if old_task_id not in leep_accumulator:
            leep_accumulator[old_task_id] = {
                'leep_scores': [],  # 存储每个 batch 的 LEEP 分数
                'total_samples': 0
            }
        
        # 计算当前 batch 的 LEEP
        batch_leep = get_leep_score(old_probs, local_target, class_per_task)
        leep_accumulator[old_task_id]['leep_scores'].append(batch_leep)
        leep_accumulator[old_task_id]['total_samples'] += logits.size(0)
    
    batch_count += 1
    is_stable = batch_count >= warmup_batches
    
    # 计算权重
    if is_stable and (batch_count - warmup_batches) % update_interval == 0:
        # 使用所有累积的 LEEP 分数计算加权平均
        leep_scores = {}
        for old_task_id in range(task_id + 1):
            scores = leep_accumulator[old_task_id]['leep_scores']
            # 可以使用加权平均（最近的 batch 权重更高）
            weights = np.exp(np.linspace(-1, 0, len(scores)))  # 指数衰减权重
            weights = weights / weights.sum()
            leep_scores[old_task_id] = np.average(scores, weights=weights)
    else:
        # Warmup 阶段：返回均匀权重
        leep_scores = {tid: -np.log(class_per_task) for tid in range(task_id + 1)}
    
    leep_values = torch.tensor([leep_scores[i] for i in range(task_id + 1)], device=device)
    leep_weights = F.softmax(leep_values, dim=0)
    
    return leep_scores, leep_weights, leep_accumulator, batch_count, is_stable



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


def analyze_three_weights(model, data_loader, device, task_id, args, reference_model=None):
    """
    分析三种任务相关性权重的分布和对比:
    1. prompt_weight_logit: 第一次推理的logits权重 (基于分类器输出)
    2. prompt_weight_leep: LEEP分数计算的权重 (基于标签转移概率)
    3. prompt_weight_sim: 原型相似度权重 (基于特征-原型余弦相似度)
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器列表 [task_0, task_1, ..., task_n]
                    每个元素是字典 {'train': loader, 'val': loader}
        device: 设备
        task_id: 当前训练到的任务ID
        args: 参数配置
        reference_model: 参考模型 (用于提取cls特征)
    
    Returns:
        analysis_results: 包含统计数据和可视化数据的字典
    """
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    
    class_per_task = args.nb_classes // args.num_tasks
    
    # 存储每个任务的权重数据
    weight_data = {
        'logit': defaultdict(list),    # 按任务分组存储
        'leep': defaultdict(list),
        'sim': defaultdict(list),
        'final': defaultdict(list),    # 最终融合权重
    }
    
    # 存储样本级别的权重对比
    sample_weights = {
        'logit': [],
        'leep': [],  
        'sim': [],
        'final': [],
        'true_task': [],  # 样本的真实任务ID
    }
    
    # LEEP累积器
    warmup_batches = args.leep_warmup_batches if hasattr(args, 'leep_warmup_batches') else 1
    batch_count = 0
    leep_accumulator = None
    
    print(f"\n{'='*70}")
    print(f"Analyzing weights for tasks 0-{task_id}...")
    print(f"{'='*70}")
    
    total_samples = 0
    max_samples_per_task = float('inf')  # 不限制，分析所有样本  # 每个任务最多分析100个样本
    
    with torch.no_grad():
        # 遍历所有已见任务的测试数据
        for eval_task_id in range(task_id + 1):
            print(f"\nProcessing Task {eval_task_id} test data...")
            
            # 获取当前任务的测试数据加载器
            current_loader = data_loader[eval_task_id]['val']
            
            task_sample_count = 0
            
            for batch_idx, (input, target) in enumerate(current_loader):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # 计算真实任务ID
                true_task_ids = target // class_per_task
                
                # ========== 1. 获取 prompt_weight_logit (第一次推理) ==========
                prompt_weight_0 = torch.ones(input.shape[0], args.num_tasks, class_per_task).to(device)
                prompt_weight_0 = torch.mean(prompt_weight_0, dim=-1)
                prompt_weight_0[:, task_id+1:] = 0
                prompt_weight_0 = norm(prompt_weight_0)
                
                output_first = model(input, task_id=task_id, train=False, 
                                    prompt_weight=prompt_weight_0, compute_similarity=False)
                logits = output_first['logits']
                
                # 转换为任务级权重
                prompt_weight_0 = F.softmax(logits, dim=-1)
                prompt_weight_0 = prompt_weight_0.reshape(input.shape[0], -1, class_per_task)
                prompt_weight_0 = torch.mean(prompt_weight_0, dim=-1)
                prompt_weight_0[:, task_id+1:] = 0
                prompt_weight_logit = norm(prompt_weight_0)
                
                # ========== 2. 计算 prompt_weight_leep (LEEP分数) ==========
                leep_scores, leep_weights, leep_accumulator, batch_count, is_stable = compute_leep_with_two_stages(
                    logits, target, eval_task_id, class_per_task, device,
                    leep_accumulator=leep_accumulator,
                    batch_count=batch_count,
                    warmup_batches=warmup_batches,
                    update_interval=1
                )
                # ✅ LEEP权重维度是 [eval_task_id+1]，需要补齐到 [task_id+1]
                if leep_weights.shape[0] < task_id + 1:
                    # 补齐维度：在末尾填充0
                    padding_size = task_id + 1 - leep_weights.shape[0]
                    leep_weights_padded = torch.cat([
                        leep_weights,
                        torch.zeros(padding_size, device=device)
                    ], dim=0)
                else:
                    leep_weights_padded = leep_weights[:task_id+1]
                # 扩展LEEP权重到batch维度
                prompt_weight_leep = leep_weights_padded.unsqueeze(0).expand(input.shape[0], -1)
                
                # ========== 3. 获取 prompt_weight_sim (原型相似度) ==========
                if reference_model is not None:
                    output_ref = reference_model(input)
                    cls_features = output_ref['pre_logits']
                else:
                    cls_features = None
                
                if cls_features is not None and hasattr(model, 'task_embedding'):
                    cls_features_norm = F.normalize(cls_features, p=2, dim=-1)
                    
                    # 计算与所有已见任务原型的相似度
                    all_seen_prototypes = model.task_embedding[:task_id+1].reshape(-1, model.embed_dim)
                    all_prototypes_norm = F.normalize(all_seen_prototypes, p=2, dim=-1)
                    
                    all_class_sim = torch.matmul(cls_features_norm, all_prototypes_norm.t())
                    all_class_sim_reshaped = all_class_sim.view(-1, task_id+1, class_per_task)
                    
                    # 任务级相似度 (取最大类相似度)
                    task_sim, _ = all_class_sim_reshaped.max(dim=2)
                    
                    # 填充到完整维度
                    full_similarity = torch.zeros(input.shape[0], args.num_tasks, device=device)
                    full_similarity[:, :task_id+1] = task_sim
                    
                    # Mask未见任务
                    mask = torch.ones_like(full_similarity)
                    mask[:, task_id+1:] = 0
                    
                    prompt_weight_raw = F.softmax(full_similarity, dim=-1)
                    prompt_weight_sim = prompt_weight_raw * mask
                    prompt_weight_sim = norm(prompt_weight_sim)
                else:
                    prompt_weight_sim = prompt_weight_logit.clone()
                
                # ========== 4. 计算最终融合权重 ==========
                # 乘性融合
                prompt_weight_final = prompt_weight_sim * prompt_weight_logit
                prompt_weight_final = norm(prompt_weight_final)
                
                # 可选: 加入LEEP
                if is_stable:
                    leep_expanded = torch.zeros(input.shape[0], args.num_tasks, device=device)
                    leep_expanded[:, :task_id+1] = prompt_weight_leep[:, :task_id+1]
                    prompt_weight_final = 0.5 * prompt_weight_final + 0.5 * leep_expanded
                    prompt_weight_final = norm(prompt_weight_final)
                
                # ========== 5. 收集数据 ==========
                for i in range(input.shape[0]):
                    true_task = true_task_ids[i].item()
                    
                    # 按任务分组统计
                    for t in range(task_id + 1):
                        weight_data['logit'][t].append(prompt_weight_logit[i, t].item())
                        weight_data['leep'][t].append(prompt_weight_leep[i, t].item())
                        weight_data['sim'][t].append(prompt_weight_sim[i, t].item())
                        weight_data['final'][t].append(prompt_weight_final[i, t].item())
                    
                    # 样本级别存储
                    sample_weights['logit'].append(prompt_weight_logit[i, :task_id+1].cpu().numpy())
                    sample_weights['leep'].append(prompt_weight_leep[i, :task_id+1].cpu().numpy())
                    sample_weights['sim'].append(prompt_weight_sim[i, :task_id+1].cpu().numpy())
                    sample_weights['final'].append(prompt_weight_final[i, :task_id+1].cpu().numpy())
                    sample_weights['true_task'].append(true_task)
                    
                    task_sample_count += 1
                    total_samples += 1
                
                # 限制每个任务的样本数
                if task_sample_count >= max_samples_per_task:
                    break
            
            print(f"  Task {eval_task_id}: collected {task_sample_count} samples")
    
    print(f"\nTotal samples analyzed: {total_samples}")
    
    # ========== 6. 计算统计数据 ==========
    statistics = {}
    for weight_type in ['logit', 'leep', 'sim', 'final']:
        statistics[weight_type] = {}
        for t in range(task_id + 1):
            data = weight_data[weight_type][t]
            if len(data) > 0:
                statistics[weight_type][f'task_{t}'] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'median': np.median(data),
                    'max': np.max(data),
                    'min': np.min(data),
                }
            else:
                statistics[weight_type][f'task_{t}'] = {
                    'mean': 0, 'std': 0, 'median': 0, 'max': 0, 'min': 0
                }
    
    # 计算任务选择准确率 (权重最大的任务是否为真实任务)
    selection_accuracy = {}
    for weight_type in ['logit', 'leep', 'sim', 'final']:
        if len(sample_weights[weight_type]) > 0:
            weights_array = np.array(sample_weights[weight_type])
            predicted_tasks = np.argmax(weights_array, axis=1)
            true_tasks = np.array(sample_weights['true_task'])
            accuracy = np.mean(predicted_tasks == true_tasks)
            selection_accuracy[weight_type] = accuracy
        else:
            selection_accuracy[weight_type] = 0.0
    
    print("\nTask Selection Accuracy:")
    for wt in ['logit', 'leep', 'sim', 'final']:
        print(f"  {wt.capitalize()}: {selection_accuracy[wt]:.4f}")
    
    return {
        'weight_data': weight_data,
        'sample_weights': sample_weights,
        'statistics': statistics,
        'selection_accuracy': selection_accuracy,
        'task_id': task_id,
    }
def visualize_weight_analysis(analysis_results, save_dir):
    """
    可视化三种权重的分布和对比
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    task_id = analysis_results['task_id']
    weight_data = analysis_results['weight_data']
    sample_weights = analysis_results['sample_weights']
    statistics = analysis_results['statistics']
    selection_accuracy = analysis_results['selection_accuracy']
    
    # ========== 1. 任务级权重分布箱线图 ==========
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    weight_types = ['logit', 'leep', 'sim', 'final']
    titles = ['Logits Weight', 'LEEP Weight', 'Prototype Similarity Weight', 'Final Fused Weight']
    
    for idx, (wt, title) in enumerate(zip(weight_types, titles)):
        data_to_plot = [weight_data[wt][t] for t in range(task_id + 1)]
        axes[idx].boxplot(data_to_plot, labels=[f'T{t}' for t in range(task_id + 1)])
        axes[idx].set_title(f'{title}\nAcc: {selection_accuracy[wt]:.3f}')
        axes[idx].set_xlabel('Task ID')
        axes[idx].set_ylabel('Weight')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_distribution_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 2. 权重热力图对比 (随机选择50个样本) ==========
    num_samples = min(50, len(sample_weights['true_task']))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (wt, title) in enumerate(zip(weight_types, titles)):
        ax = axes[idx // 2, idx % 2]
        weights_matrix = np.array(sample_weights[wt][:num_samples])
        true_tasks = np.array(sample_weights['true_task'][:num_samples])
        
        # 按真实任务排序
        sort_idx = np.argsort(true_tasks)
        weights_sorted = weights_matrix[sort_idx]
        true_tasks_sorted = true_tasks[sort_idx]
        
        sns.heatmap(weights_sorted, ax=ax, cmap='YlOrRd', 
                   xticklabels=[f'T{t}' for t in range(task_id + 1)],
                   yticklabels=[f'S{i}(T{true_tasks_sorted[i]})' for i in range(num_samples)],
                   cbar_kws={'label': 'Weight'}, vmin=0, vmax=1)
        ax.set_title(f'{title}')
        ax.set_xlabel('Task ID')
        ax.set_ylabel('Sample (True Task)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_heatmap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 3. 权重相关性分析 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    comparisons = [('logit', 'leep'), ('logit', 'sim'), ('leep', 'sim')]
    
    for idx, (wt1, wt2) in enumerate(comparisons):
        weights1 = np.array(sample_weights[wt1])
        weights2 = np.array(sample_weights[wt2])
        
        # 展平所有任务的权重
        weights1_flat = weights1.flatten()
        weights2_flat = weights2.flatten()
        
        axes[idx].scatter(weights1_flat, weights2_flat, alpha=0.3, s=10)
        axes[idx].set_xlabel(f'{wt1.capitalize()} Weight')
        axes[idx].set_ylabel(f'{wt2.capitalize()} Weight')
        axes[idx].set_title(f'{wt1.capitalize()} vs {wt2.capitalize()}\nCorr: {np.corrcoef(weights1_flat, weights2_flat)[0,1]:.3f}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 4. 统计报告 ==========
    with open(os.path.join(save_dir, 'weight_statistics.txt'), 'w') as f:
        f.write(f"=== Weight Analysis Report (Task {task_id}) ===\n\n")
        
        f.write("Task Selection Accuracy:\n")
        for wt in weight_types:
            f.write(f"  {wt.capitalize()}: {selection_accuracy[wt]:.4f}\n")
        f.write("\n")
        
        for wt in weight_types:
            f.write(f"--- {wt.capitalize()} Weight Statistics ---\n")
            for t in range(task_id + 1):
                stats = statistics[wt][f'task_{t}']
                f.write(f"  Task {t}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                       f"median={stats['median']:.4f}, max={stats['max']:.4f}, min={stats['min']:.4f}\n")
            f.write("\n")
    
    print(f"✅ Visualizations saved to {save_dir}")
    print(f"   - weight_distribution_boxplot.png")
    print(f"   - weight_heatmap_comparison.png")
    print(f"   - weight_correlation.png")
    print(f"   - weight_statistics.txt")


def norm(feature):
    """归一化到和为1"""
    feature_norm = torch.sum(feature, dim=-1, keepdim=True)
    feature_norm = torch.clamp(feature_norm, min=1e-10)
    return feature / feature_norm


