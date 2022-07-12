import argparse
import time
import pandas as pd
import torch
import torch.optim as optim
import torchnet as tnt
from torch import nn
from utils import util
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import dataset
from models.C3D import C3D
from models.R2Plus1D import R2Plus1D

torch.backends.cudnn.benchmark = True



class trainer(object):
    def __init__(self, opt):
        self.train_csv = opt.train_csv
        self.test_csv = opt.test_csv

        self.batch_size = opt.batch_size
        self.max_num_epoch = opt.num_epochs

        self.model_type, self.PRE_TRAIN, self.gpu_id= opt.model_type, opt.pre_train, opt.gpu_ids

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get dataloader
        train_data = dataset.VideoDataset(data_csv=self.train_csv, split='train')
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        # val_data = VideoDataset(dataset=dataset, split='val')
        # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
        test_data = dataset.VideoDataset(data_csv=self.test_csv, split='test')
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=16)
        self.val_loader = self.test_loader


    def train(self):

        # Initiate model
        NUM_CLASS = max(self.train_loader.dataset.emotion_num)
        if self.model_type == 'r2plus1d':
            model = R2Plus1D(NUM_CLASS, (2, 2, 2, 2))
        else:
            model = C3D(NUM_CLASS)

        loss_criterion = nn.CrossEntropyLoss()

        # optim_configs = [{'params': model.parameters(), 'lr': 1e-4}]
        optim_configs = [{'params': model.feature.parameters(), 'lr': 0.001},
                         {'params': model.fc.parameters(), 'lr': 0.001 * 10}]
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience= 20, factor=0.9, verbose=True)
        print('Number of parameters:', sum(param.numel() for param in model.parameters()))

        model = model.to(self.device)

        for epoch in range(0, self.max_num_epoch):

            # Get logger for experiment
            batch_time, losses, top1, top5, lr = self.create_logger(split='Training')
            iter_progress = self.display_logger(self.train_loader, "{}: [{}]".format('batch: ', epoch), [batch_time, losses, top1, top5, lr])
            end = time.time()
            model.train()
            for it, (videos, labels) in enumerate(self.train_loader):

                iteration = epoch * len(self.train_loader) + it

                videos = videos.to(self.device , non_blocking=True)
                labels = labels.to(self.device , non_blocking=True)
                outputs = model(videos)
                loss = loss_criterion(outputs, labels)

                # measure accuracy and record loss
                acc1, acc5 = util.accuracy(outputs, labels, topk=(1, 5))
                losses.update(loss.item(), videos.size(0))
                top1.update(acc1[0], videos.size(0))
                top5.update(acc5[0], videos.size(0))
                lr.update(optimizer.param_groups[0]['lr'])
                # compute gradient and do SGD step
                loss.backward()
                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if it % opt.print_freq == 0:
                    iter_progress.display(it)
        # epoch_progress = self.display_logger(self.train_loader, epoch, [lr])
        # epoch_progress.display()
            test_loss = self.validate(self.val_loader, model, loss_criterion)
            scheduler.step(losses.avg)


    def validate(self, val_loader, model, criterion, epoch=0):
        batch_time, losses, top1, top5, lr = self.create_logger(split='testing')
        iter_progress = self.display_logger(self.test_loader, "{}: [{}]".format('Testing: ', epoch), [batch_time, losses, top1, top5])

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for it, (videos, labels) in enumerate(val_loader):

                videos = videos.cuda(self.device, non_blocking=True)
                labels = labels.cuda(self.device, non_blocking=True)

                # compute output
                output = model(videos)
                loss = criterion(output, labels)

                # measure accuracy and record loss
                acc1, acc5 = util.accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), videos.size(0))
                top1.update(acc1[0], videos.size(0))
                top5.update(acc5[0], videos.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            # TODO: this should also be done with the ProgressMeter
            iter_progress.display(it)

        return losses.avg

    def create_logger(self, split):
        batch_time = util.AverageMeter('Time:', ':.3f')
        losses = util.AverageMeter('{} Loss:'.format(split), ':.4f')
        top1 = util.AverageMeter('{}-Top1 Accuracy: '.format(split), ':.2f')
        top5 = util.AverageMeter('{}-Top5 Accuracy: '.format(split), ':.2f')
        lr = util.AverageMeter('Learning rate: ', ':.6f')

        return batch_time, losses, top1, top5, lr

    def display_logger(self, data_loader, prefixs, output_list):
        progress = util.ProgressMeter(
            len(data_loader),
            output_list,
            prefix=prefixs)

        return progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Activity Recognition Model')
    parser.add_argument('--data_type', default='ucf101', type=str, choices=['ucf101', 'hmdb51', 'kinetics600'],
                        help='dataset type')
    parser.add_argument('--gpu_ids', default='0', type=str, help='selected gpu')
    parser.add_argument('--model_type', default='r2plus1d', type=str, choices=['r2plus1d', 'c3d'], help='model type')
    parser.add_argument('--batch_size', default=12, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=1000, type=int, help='training epoch number')
    parser.add_argument('--pre_train', default=None, type=str, help='used pre-trained model epoch name')

    opt = parser.parse_args()
    opt.train_csv = '/home/fz/1-Dataset/RAVDESS/RAVDESS_video_train.csv'
    opt.test_csv = '/home/fz/1-Dataset/RAVDESS/RAVDESS_video_test.csv'
    opt.print_freq = 10

    R3D_net = trainer(opt)
    R3D_net.train()
