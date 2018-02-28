from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
from datasets import Cifar10Folder
import config
import data

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.config import cfg, cfg_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/birds_proGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    num_gpu = len(cfg.GPU_ID.split(','))
    print("num_gpu:{0}".format(num_gpu))

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'image_train', True
    # if not cfg.TRAIN.FLAG:
    #     if cfg.DATASET_NAME == 'birds':
    #         bshuffle = False
    #         split_dir = 'image_test'

    # Get data loader imsize=256 BASE_SIZE=64
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    print(cfg.TREE.BASE_SIZE)

    print("imsize:{0}".format(imsize))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])


    if cfg.DATA_DIR.find('cifar10') != -1:
        label_dataset = Cifar10Folder(cfg.DATA_DIR, "cifar10_label", transform=image_transform)
        unlabel_dataset = Cifar10Folder(cfg.DATA_DIR, "cifar10_unlabel", transform=image_transform)

    elif cfg.DATA_DIR.find('imagenet') != -1:
       pass
    elif cfg.GAN.B_CONDITION:
        pass

    label_loader = torch.utils.data.DataLoader(
        label_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
    unlabel_loader = torch.utils.data.DataLoader(
        label_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))



    from trainer import condGANTrainer as trainer
    algo = trainer(output_dir, label_loader, unlabel_loader, imsize)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        db_dataset = Cifar10Folder(cfg.DATA_DIR, "image_db",
                                   base_size=cfg.TREE.BASE_SIZE,
                                   transform=image_transform)
        test_dataset = Cifar10Folder(cfg.DATA_DIR, "image_test",
                                     base_size=cfg.TREE.BASE_SIZE,
                                     transform=image_transform)

        db_dataloader = torch.utils.data.DataLoader(
            db_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,  # * num_gpu,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,  # * num_gpu,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        algo.evaluate_MAP(db_dataloader, test_dataloader, "eval")

    end_t = time.time()
    print('Total time for training:', end_t - start_t)
    ''' Running time comparison for 10epoch with batch_size 24 on birds dataset
        T(1gpu) = 1.383 T(2gpus)
            - gpu 2: 2426.228544 -> 4min/epoch
            - gpu 2 & 3: 1754.12295008 -> 2.9min/epoch
            - gpu 3: 2514.02744293
    '''
