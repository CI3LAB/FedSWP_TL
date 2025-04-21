# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Joy
# Date              : 2024/06/09
# Last Modified Date: 2024/08/29
# Last Modified By  : Joy
# Reference         : siamese-triplet
# Description       : main
# ******************************************************
import os
import time
import json
import torch
import argparse
import torch.distributed as dist


if __name__ == '__main__':
    # dl setting
    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    wsize = dist.get_world_size()  # number of processes = num_workers + 1
    server_rank = -1 # this process is the server

    # setup
    parser = argparse.ArgumentParser(description='FedSWP TL')
    parser.add_argument('--data', type=str, default='yawdd', choices=("conppfm", "yawdd", "drozy"),
                        help='Define the data used for training')
    parser.add_argument('--use_cuda', action="store_true", help='Use CUDA if available')
    args = parser.parse_args()

    # Setting device to local rank (torch.distributed.launch)
    if args.use_cuda:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
        print(devices,flush=True)
        per_device_ranks = int(wsize/len(devices)) + 1
        print('Device assignment: %s , %s'%(args.local_rank, int(args.local_rank/per_device_ranks)),flush=True)
        torch.cuda.set_device(int(args.local_rank/per_device_ranks))

    torch.manual_seed(66)
    device = torch.device("cuda" if args.use_cuda else "cpu")

    #############################################################################################################################
    #                                                 swp setup code
    #############################################################################################################################
    argdict = {
        'batch_size' : 512,
        'shuffle' : True,
        'epochs': 10,
        'log_interval': 3
    }
    triplet_train_loader = torch.utils.data.DataLoader(**argdict)
    triplet_test_loader = torch.utils.data.DataLoader(**argdict)

    from trainer import SWP
    trainer = SWP()
    trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler,**argdict)
    print("*" * 20)
    hist = {"acc": trainer.worker_testloader_acc_hist, "loss": trainer.worker_train_loss_hist}
    print(hist)
