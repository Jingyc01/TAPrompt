import os.path
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader

import utils
import warnings


warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
warnings.filterwarnings('ignore', category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_taprompt':
        from configs.cifar100_taprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_taprompt', help='Split-CIFAR100 taprompt configs')
    elif config == 'imr_taprompt':
        from configs.imr_taprompt import get_args_parser
        config_parser = subparser.add_parser('imr_taprompt', help='Split-ImageNet-R taprompt configs')
    elif config == 'cub_taprompt':
        from configs.cub_taprompt import get_args_parser
        config_parser = subparser.add_parser('cub_taprompt', help='Split-CUB taprompt configs')
    elif config == 'resisc45_taprompt':
        from configs.resisc45_taprompt import get_args_parser
        config_parser = subparser.add_parser('resisc45_taprompt', help='Split-RESISC45 taprompt configs')
    elif config == 'resisc45_taprompt_transfer':
        from configs.resisc45_taprompt_transfer import get_args_parser
        config_parser = subparser.add_parser('resisc45_taprompt_transfer', help='Split-RESISC45-transfer taprompt configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    args = parser.parse_args()
    args.config = config
    return args

def main(args):
    utils.init_distributed_mode(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if 'taprompt' in args.config :
        import trainers.taprompt_trainer as taprompt_trainer
        taprompt_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    args = get_args()
    print(args)
    main(args)
