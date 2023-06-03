import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

import lib.utils as utl
from configs.jaad import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.jaadpie_train_utils_cvae import train, val, test

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', model_name, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))


    model = build_model(args)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-10, verbose=1)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']

    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train')
    val_gen = utl.build_data_loader(args, 'val')
    test_gen = utl.build_data_loader(args, 'test')
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())



    # train
    min_loss = 1e6
    min_MSE_15 = 10e5
    best_model = None
    best_model_metric = None

    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):

        # print("Number of training samples:", len(train_gen))

        print('\n                                    \tEpoch {}\t                                       \n'.format(epoch))
        # train
        print('************************************\tTraining\t**************************************')
        train_goal_loss, train_cvae_loss, train_KLD_loss = train(model, train_gen, criterion, optimizer, device, args.steps_per_epoch)        
        print('Goal loss: {:.4f}\t CVAE loss: {:.4f}\t KLD loss: {:.4f}\n'.format(train_goal_loss, train_cvae_loss, train_KLD_loss))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_goal_loss + train_cvae_loss + train_KLD_loss,
            }, 
            'checkpoints/model.pt')

        # val
        print('************************************\tValidation\t************************************')
        val_loss = val(model, val_gen, criterion, device, args.steps_validation)
        print("Validation Loss: {:.4f}\n".format(val_loss))
        lr_scheduler.step(val_loss)

        # test
        print('************************************\tTesting\t***************************************')
        test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE = test(model, test_gen, criterion, device, args.steps_testing)
        print("Test Loss: {:.4f}".format(test_loss))
        print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f\n" % (MSE_05, MSE_10, MSE_15))

        # print('\n************************************Epoch {} End************************************\n'.format(epoch))



if __name__ == '__main__':
    main(parse_args())
