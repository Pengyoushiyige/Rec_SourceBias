import argparse
import utils
import torch
import random
import numpy as np
from pathlib import Path
from parameters import parse_args
from test import Tester
from train_loop import Trainer
from train_human import Trainer_main

import os
import logging

def seed_torch(seed=2023):  # 固定随机种子以确保结果可复现
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #固定 Python 内置哈希算法的随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 关闭 CUDNN benchmark 避免算法动态寻找引入差异
    torch.backends.cudnn.deterministic = True  # 强制使用确定性算法

if __name__ == '__main__':
    utils.setuplogger()  # 1. 初始化日志系统
    args = parse_args()  # 2. 解析命令行参数
    seed_torch(seed=args.seed)  # 3. 设置全局随机种子
    logging.info(f"args: {args}")  # 4. 日志记录参数配置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 5. 指定 GPU 计算设备
    if 'loop' == args.mode:  # 6. 分支一：迭代训练模式
        trainer = Trainer(args)
        trainer.run()
    elif 'test' == args.mode:  # 分支二：模型测试模式
        tester = Tester(args)
        tester.run()
    elif 'train' == args.mode:  # 分支三：主训练模式
        tester = Trainer_main(args)
        tester.run()