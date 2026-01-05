from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer
import evaluate_FlowFormer as evaluate
import evaluate_FlowFormer_tile as evaluate_tile
import core.datasets as datasets
from core.loss import sequence_loss, sequence_loss_nighttime, sequence_loss_daytime, flow_consis_loss, loss_KL_div, cost_memory_consist, sequence_event_loss, flow_attention_loss
from core.optimizer import fetch_optimizer, build_scheduler
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
import tqdm
# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger
from core.utils.gradients import Sobel
from PIL import Image
from torch.autograd import Variable

import torchvision.transforms as transforms
from glob import glob
from core.RetinexNet.retinex_net import Network as RetinexNet

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NighttimeFlow:
    def __init__(self, cfg):
        # if split == 'testing':
        #     self.is_test = True
        self.cfg = cfg
        
    # train daytime optical flow network: unsupervised and supervised.
    def train_stage_1(self):
        cfg = self.cfg
        model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(model))

        if cfg.restore_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
            model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

        model.cuda()
        model.train()

        train_loader = datasets.fetch_dataloader(cfg)
        optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(model, scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, flow, valid = [x.cuda() for x in data_blob]
                

                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                
                output = {}
                flow_predictions, _ = model(image1, image2)
                
                loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                metrics.update(output)
                logger.push(metrics)

                ### change evaluate to functions

                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                    torch.save(model.state_dict(), PATH)
                
                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = cfg.log_dir + '/final'
        torch.save(model.state_dict(), PATH)

        PATH = f'checkpoints/{cfg.stage}.pth'
        torch.save(model.state_dict(), PATH)

        return PATH
    
    # train nighttime optical flow network: unsupervised and supervised.
    def train_stage_2(self):
        cfg = self.cfg
        model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(model))

        if cfg.restore_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
            model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

        model.cuda()
        model.train()

        train_loader = datasets.fetch_dataloader(cfg)
        optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(model, scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, flow, valid = [x.cuda() for x in data_blob]
                

                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                
                output = {}
                flow_predictions, _ = model(image1, image2)
                
                loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                metrics.update(output)
                logger.push(metrics)

                ### change evaluate to functions

                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                    torch.save(model.state_dict(), PATH)
                
                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = cfg.log_dir + '/final'
        torch.save(model.state_dict(), PATH)

        PATH = f'checkpoints/{cfg.stage}.pth'
        torch.save(model.state_dict(), PATH)

        return PATH



    # train nighttime optical flow network: unsupervised and supervised.
    def generate_reflectance(self):
               
        cfg = self.cfg
        
        retinex_model = RetinexNet()
        retinex_model = retinex_model.cuda()
        retinex_model.eval()
        retinex_model.load_state_dict(torch.load(cfg.retinex_model))

        for p in retinex_model.parameters():
            p.requires_grad = False


        daytime_img_path = '/mnt/data/zhouhanyu/NIPS2023/dataset/DayKITTI/train/Image'
        nighttime_img_path = '/mnt/data/zhouhanyu/NIPS2023/dataset/NightKITTI/train/Image'

        daytime_R_path = '/mnt/data/zhouhanyu/NIPS2023/dataset/DayKITTI/train/R'
        if not os.path.exists(daytime_R_path):
            os.makedirs(daytime_R_path)
        
        nighttime_R_path = '/mnt/data/zhouhanyu/NIPS2023/dataset/NightKITTI/train/R'
        if not os.path.exists(nighttime_R_path):
            os.makedirs(nighttime_R_path)

        # daytime_img_path = 'F:/Research/Experiment_Code/2023NIPS/Dataset/DayKITTI/train/Image'
        # nighttime_img_path = 'F:/Research/Experiment_Code/2023NIPS/Dataset/NightKITTI/train/Image'
        # # nighttime_img_path = 'F:/Dateset/NIPS_Data/000001/left'

        # daytime_R_path = 'F:/Research/Experiment_Code/2023NIPS/Dataset/DayKITTI/train/R'
        # if not os.path.exists(daytime_R_path):
        #     os.makedirs(daytime_R_path)
        
        # nighttime_R_path = 'F:/Research/Experiment_Code/2023NIPS/Dataset/NightKITTI/train/R'
        # # nighttime_R_path = 'F:/Dateset/NIPS_Data/000001/R'
        # if not os.path.exists(nighttime_R_path):
        #     os.makedirs(nighttime_R_path)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        transform = transforms.Compose(transform_list)
        
        daytime_img_names = glob(daytime_img_path + '/*.png')
        # daytime_img_names = glob(daytime_img_path + '/*.bmp')
        daytime_img_names.sort()
        count_daytime = len(daytime_img_names)

        nighttime_img_names = glob(nighttime_img_path + '/*.png')
        # nighttime_img_names = glob(nighttime_img_path + '/*.bmp')
        nighttime_img_names.sort()
        count_nighttime = len(nighttime_img_names)
        print(count_daytime)
        # 继续遍历
        if False:
            with torch.no_grad():
                for idx in range(count_daytime):
                    print("generate daytime " + str(idx) + " img.")
                    img = Image.open(daytime_img_names[idx]).convert('RGB')
                    img_norm = transform(img).numpy()
                    img_norm = np.transpose(img_norm, (1, 2, 0))
                    
                    save_img_name = daytime_img_names[idx].split('\\')[-1]
                    # print(save_img_name)
                    low = np.asarray(img_norm, dtype=np.float32)
                    low = np.transpose(low[:, :, :], (2, 0, 1))
                    daytime_img = torch.from_numpy(low)
                    daytime_img = daytime_img.unsqueeze(0)
                    input = Variable(daytime_img, volatile=True).cuda()
                    u_list, r_list = retinex_model(input)

                    u_path = daytime_R_path + '/' + save_img_name
                
                    image_numpy = u_list[0].cpu().float().numpy()[0]
                    print(image_numpy.shape)
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
                    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
                    im.save(u_path)
        


        print(count_nighttime)
        if True:
            with torch.no_grad():
                for idx in range(count_nighttime):
                    print("generate nighttime " + str(idx) + " img.")
                    img = Image.open(nighttime_img_names[idx]).convert('RGB')
                    img_norm = transform(img).numpy()
                    img_norm = np.transpose(img_norm, (1, 2, 0))
                    
                    save_img_name = nighttime_img_names[idx].split('\\')[-1]
                    # print(save_img_name)
                    low = np.asarray(img_norm, dtype=np.float32)
                    low = np.transpose(low[:, :, :], (2, 0, 1))
                    nighttime_img = torch.from_numpy(low)
                    nighttime_img = nighttime_img.unsqueeze(0)
                    input = Variable(nighttime_img, volatile=True).cuda()
                    u_list, r_list = retinex_model(input)

                    u_path = nighttime_R_path + '/' + save_img_name
                
                    image_numpy = u_list[0].cpu().float().numpy()[0]
                    print(image_numpy.shape)
                    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
                    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
                    im.save(u_path)

        
        return 0

    
    # train daytime/nighttime reflectance optical flow network: unsupervised.
    # loss: consis loss and epe loss
    def train_stage_4(self):
        cfg = self.cfg
        dayflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Dayflow_model Parameter Count: %d" % count_parameters(dayflow_model))

        nightflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Nightflow_model Parameter Count: %d" % count_parameters(nightflow_model))

        day_refectflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Refectflow_model Parameter Count: %d" % count_parameters(day_refectflow_model))

        night_refectflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Refectflow_model Parameter Count: %d" % count_parameters(night_refectflow_model))

        # restore_reflectance_ckpt, restore_nighttime_ckpt, restore_daytime_ckpt
        if cfg.restore_daytime_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_daytime_ckpt))
            dayflow_model.load_state_dict(torch.load(cfg.restore_daytime_ckpt), strict=True)
        
        if cfg.restore_nighttime_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_nighttime_ckpt))
            nightflow_model.load_state_dict(torch.load(cfg.restore_nighttime_ckpt), strict=True)
        
        if cfg.restore_dayreflect_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_dayreflect_ckpt))
            day_refectflow_model.load_state_dict(torch.load(cfg.restore_dayreflect_ckpt), strict=True)
        
        if cfg.restore_nightreflect_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_nightreflect_ckpt))
            night_refectflow_model.load_state_dict(torch.load(cfg.restore_nightreflect_ckpt), strict=True)

        dayflow_model.cuda()
        dayflow_model.eval()

        nightflow_model.cuda()
        nightflow_model.train()

        day_refectflow_model.cuda()
        day_refectflow_model.train()

        night_refectflow_model.cuda()
        night_refectflow_model.train()


        # load retinex model
        retinex_model = RetinexNet()
        retinex_model = retinex_model.cuda()
        retinex_model.eval()
        retinex_model.load_state_dict(torch.load(cfg.retinex_model))
        for p in retinex_model.parameters():
            p.requires_grad = False

        # continue
        day_train_loader, night_train_loader = datasets.fetch_dataloader(cfg)
        # night_train_loader = datasets.fetch_dataloader(cfg)
        # nightflow_optimizer, nightflow_scheduler = fetch_optimizer(nightflow_model, cfg.trainer)
        # day_refectflow_optimizer, day_refectflow_scheduler = fetch_optimizer(day_refectflow_model, cfg.trainer)
        # night_refectflow_optimizer, night_refectflow_scheduler = fetch_optimizer(night_refectflow_model, cfg.trainer)

        # set optimizer and scheduler
        params_group = [
            {'params': nightflow_model.parameters()},
            {'params': day_refectflow_model.parameters()},
            {'params': night_refectflow_model.parameters()}
        ]
        optimizer = torch.optim.AdamW(params_group, lr=cfg.trainer.canonical_lr, weight_decay=cfg.trainer.adamw_decay, eps=cfg.trainer.epsilon)
        scheduler = build_scheduler(cfg.trainer, optimizer)


        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(nightflow_model, scheduler, cfg)
        # refectflow_logger = Logger(day_refectflow_model, day_refectflow_scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(day_train_loader):
                optimizer.zero_grad()
                # day_refectflow_optimizer.zero_grad()
                # night_refectflow_optimizer.zero_grad()

                day_image1, day_image2, day_flow, day_valid = [x.cuda() for x in data_blob]

                # night_image1, night_image2, night_flow, night_valid = [x.cuda() for x in night_train_loader.__getitem__(i_batch)]
                
                night_blob = next(iter(night_train_loader))
                night_image1, night_image2, night_flow, night_valid = [x.cuda() for x in night_blob]
                
                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    day_image1 = (day_image1 + stdv * torch.randn(*day_image1.shape).cuda()).clamp(0.0, 255.0)
                    day_image2 = (day_image2 + stdv * torch.randn(*day_image2.shape).cuda()).clamp(0.0, 255.0)
                    night_image1 = (night_image1 + stdv * torch.randn(*night_image1.shape).cuda()).clamp(0.0, 255.0)
                    night_image2 = (night_image2 + stdv * torch.randn(*night_image2.shape).cuda()).clamp(0.0, 255.0)
                else:
                    day_image1 = day_image1.clamp(0.0, 255.0)
                    day_image2 = day_image2.clamp(0.0, 255.0)
                    night_image1 = night_image1.clamp(0.0, 255.0)
                    night_image2 = night_image2.clamp(0.0, 255.0)

                
                output = {}
                
                with torch.no_grad():
                    day_R1_list, day_r1_list = retinex_model(day_image1)
                    day_R2_list, day_r2_list = retinex_model(day_image2)
                    night_R1_list, night_r1_list = retinex_model(night_image1)
                    night_R2_list, night_r2_list = retinex_model(night_image2)
                    day_flow_predictions, _ = dayflow_model(day_image1, day_image2)

                night_flow_predictions, _ = nightflow_model(night_image1, night_image2)
                
                day_R1 = day_R1_list[-2]
                day_R2 = day_R2_list[-2]
                night_R1 = night_R1_list[-2]
                night_R2 = night_R2_list[-2]
                
                # add retinex for reflectance
                dayref_flow_predictions, _ = day_refectflow_model(day_R1, day_R2)
                nightref_flow_predictions, _ = night_refectflow_model(night_R1, night_R2)
                
                # day_loss, day_metrics = sequence_loss_daytime(day_flow_predictions, day_flow, day_valid, cfg)
                night_loss, night_metrics = sequence_loss_nighttime(night_flow_predictions, night_flow, night_valid, cfg)
                
                # consistency loss
                day_ref_loss, day_ref_metrics = flow_consis_loss(dayref_flow_predictions[0], day_flow_predictions[0])
                night_ref_loss, night_ref_metrics = flow_consis_loss(nightref_flow_predictions[0], night_flow_predictions[0])
                
                loss = night_loss + day_ref_loss + night_ref_loss
                
                # gradient backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(nightflow_model.parameters(), cfg.trainer.clip)
                torch.nn.utils.clip_grad_norm_(day_refectflow_model.parameters(), cfg.trainer.clip)
                torch.nn.utils.clip_grad_norm_(night_refectflow_model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                dict_metric = dict(night_metrics, **day_ref_metrics)
                dict_metric = dict(dict_metric, **night_ref_metrics)
                dict_metric.update(output)
                logger.push(dict_metric)

                ### change evaluate to functions

                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'nightkitti')
                    torch.save(nightflow_model.state_dict(), PATH)

                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'day_reflectance')
                    torch.save(day_refectflow_model.state_dict(), PATH)

                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'night_reflectance')
                    torch.save(night_refectflow_model.state_dict(), PATH)
                

                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        # refectflow_logger.close()
        # refectflow_logger.close()

        PATH = f'checkpoints/nightkitti.pth'
        torch.save(nightflow_model.state_dict(), PATH)

        PATH = f'checkpoints/day_reflectance.pth'
        torch.save(day_refectflow_model.state_dict(), PATH)

        PATH = f'checkpoints/night_reflectance.pth'
        torch.save(night_refectflow_model.state_dict(), PATH)

        return PATH


    # 继续添加KL散度、cost memory一致性损失代码
    def train_stage_5(self):
        cfg = self.cfg
        dayflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Dayflow_model Parameter Count: %d" % count_parameters(dayflow_model))

        nightflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Nightflow_model Parameter Count: %d" % count_parameters(nightflow_model))

        day_refectflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Refectflow_model Parameter Count: %d" % count_parameters(day_refectflow_model))

        night_refectflow_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Refectflow_model Parameter Count: %d" % count_parameters(night_refectflow_model))

        # restore_reflectance_ckpt, restore_nighttime_ckpt, restore_daytime_ckpt
        if cfg.restore_daytime_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_daytime_ckpt))
            dayflow_model.load_state_dict(torch.load(cfg.restore_daytime_ckpt), strict=True)
        
        if cfg.restore_nighttime_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_nighttime_ckpt))
            nightflow_model.load_state_dict(torch.load(cfg.restore_nighttime_ckpt), strict=True)
        
        if cfg.restore_dayreflect_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_dayreflect_ckpt))
            day_refectflow_model.load_state_dict(torch.load(cfg.restore_dayreflect_ckpt), strict=True)
        
        if cfg.restore_nightreflect_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_nightreflect_ckpt))
            night_refectflow_model.load_state_dict(torch.load(cfg.restore_nightreflect_ckpt), strict=True)

        dayflow_model.cuda()
        dayflow_model.eval()

        nightflow_model.cuda()
        nightflow_model.train()

        day_refectflow_model.cuda()
        day_refectflow_model.train()

        night_refectflow_model.cuda()
        night_refectflow_model.train()


        # load retinex model
        retinex_model = RetinexNet()
        retinex_model = retinex_model.cuda()
        retinex_model.eval()
        retinex_model.load_state_dict(torch.load(cfg.retinex_model))
        for p in retinex_model.parameters():
            p.requires_grad = False

        # load data
        day_train_loader, night_train_loader = datasets.fetch_dataloader(cfg)

        # set optimizer and scheduler
        params_group = [
            {'params': nightflow_model.parameters()},
            {'params': day_refectflow_model.parameters()},
            {'params': night_refectflow_model.parameters()}
        ]
        optimizer = torch.optim.AdamW(params_group, lr=cfg.trainer.canonical_lr, weight_decay=cfg.trainer.adamw_decay, eps=cfg.trainer.epsilon)
        scheduler = build_scheduler(cfg.trainer, optimizer)


        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(nightflow_model, scheduler, cfg)
        # refectflow_logger = Logger(day_refectflow_model, day_refectflow_scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(day_train_loader):
                optimizer.zero_grad()
                # day_refectflow_optimizer.zero_grad()
                # night_refectflow_optimizer.zero_grad()

                day_image1, day_image2, day_flow, day_valid = [x.cuda() for x in data_blob]

                # night_image1, night_image2, night_flow, night_valid = [x.cuda() for x in night_train_loader.__getitem__(i_batch)]
                
                night_blob = next(iter(night_train_loader))
                night_image1, night_image2, night_flow, night_valid = [x.cuda() for x in night_blob]
                
                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    day_image1 = (day_image1 + stdv * torch.randn(*day_image1.shape).cuda()).clamp(0.0, 255.0)
                    day_image2 = (day_image2 + stdv * torch.randn(*day_image2.shape).cuda()).clamp(0.0, 255.0)
                    night_image1 = (night_image1 + stdv * torch.randn(*night_image1.shape).cuda()).clamp(0.0, 255.0)
                    night_image2 = (night_image2 + stdv * torch.randn(*night_image2.shape).cuda()).clamp(0.0, 255.0)
                else:
                    day_image1 = day_image1.clamp(0.0, 255.0)
                    day_image2 = day_image2.clamp(0.0, 255.0)
                    night_image1 = night_image1.clamp(0.0, 255.0)
                    night_image2 = night_image2.clamp(0.0, 255.0)

                
                output = {}
                
                with torch.no_grad():
                    day_R1_list, day_r1_list = retinex_model(day_image1)
                    day_R2_list, day_r2_list = retinex_model(day_image2)
                    night_R1_list, night_r1_list = retinex_model(night_image1)
                    night_R2_list, night_r2_list = retinex_model(night_image2)
                    day_flow_predictions, day_cost_predictions = dayflow_model(day_image1, day_image2)

                night_flow_predictions, night_cost_predictions = nightflow_model(night_image1, night_image2)
                
                day_R1 = day_R1_list[-2]
                day_R2 = day_R2_list[-2]
                night_R1 = night_R1_list[-2]
                night_R2 = night_R2_list[-2]
                
                # add retinex for reflectance
                dayref_flow_predictions, dayref_cost_predictions = day_refectflow_model(day_R1, day_R2)
                nightref_flow_predictions, nightref_cost_predictions = night_refectflow_model(night_R1, night_R2)
                
                # day_loss, day_metrics = sequence_loss_daytime(day_flow_predictions, day_flow, day_valid, cfg)
                night_loss, night_metrics = sequence_loss_nighttime(night_flow_predictions, night_flow, night_valid, cfg)
                
                # consistency loss
                day_ref_loss, day_ref_metrics = flow_consis_loss(dayref_flow_predictions[0], day_flow_predictions[0])
                night_ref_loss, night_ref_metrics = flow_consis_loss(nightref_flow_predictions[0], night_flow_predictions[0])
                
                loss_KL, kl_metrics = loss_KL_div(nightref_cost_predictions, dayref_cost_predictions)

                cost_loss = cost_memory_consist(day_cost_predictions, dayref_cost_predictions)[0] + cost_memory_consist(night_cost_predictions, nightref_cost_predictions)[0]

                loss = night_loss + day_ref_loss + night_ref_loss + loss_KL + cost_loss
                
                # gradient backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(nightflow_model.parameters(), cfg.trainer.clip)
                torch.nn.utils.clip_grad_norm_(day_refectflow_model.parameters(), cfg.trainer.clip)
                torch.nn.utils.clip_grad_norm_(night_refectflow_model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                dict_metric = dict(night_metrics, **night_ref_metrics)
                dict_metric = dict(dict_metric, **kl_metrics)
                dict_metric.update(output)
                logger.push(dict_metric)

                ### change evaluate to functions

                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'nightkitti')
                    torch.save(nightflow_model.state_dict(), PATH)

                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'day_reflectance')
                    torch.save(day_refectflow_model.state_dict(), PATH)

                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'night_reflectance')
                    torch.save(night_refectflow_model.state_dict(), PATH)
                

                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        # refectflow_logger.close()
        # refectflow_logger.close()

        PATH = f'checkpoints/nightkitti.pth'
        torch.save(nightflow_model.state_dict(), PATH)

        PATH = f'checkpoints/day_reflectance.pth'
        torch.save(day_refectflow_model.state_dict(), PATH)

        PATH = f'checkpoints/night_reflectance.pth'
        torch.save(night_refectflow_model.state_dict(), PATH)


        return PATH

    # 训练事件光流模型
    def train_stage_6(self):
        cfg = self.cfg
        event_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(event_model))

        if cfg.restore_event_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_event_ckpt))
            event_model.load_state_dict(torch.load(cfg.restore_event_ckpt), strict=True)

        event_model.cuda()
        event_model.train()

        train_loader = datasets.fetch_dataloader(cfg)
        optimizer, scheduler = fetch_optimizer(event_model, cfg.trainer)

        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(event_model, scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, event1, event2, flow, valid = [x.cuda() for x in data_blob]
                

                # if cfg.add_noise:
                #     stdv = np.random.uniform(0.0, 5.0)
                #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                
                output = {}
                flow_predictions, _ = event_model(event1, event2)
                
                loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(event_model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                metrics.update(output)
                logger.push(metrics)

                ### change evaluate to functions

                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                    torch.save(event_model.state_dict(), PATH)
                
                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = cfg.log_dir + '/final'
        torch.save(event_model.state_dict(), PATH)

        PATH = f'checkpoints/{cfg.stage}.pth'
        torch.save(event_model.state_dict(), PATH)

        return PATH



    # 用DSEC继续训练事件光流模型和夜间图像光流模型，筛选一下夜间数据
    def train_stage_7(self):
        cfg = self.cfg
        event_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(event_model))

        night_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(night_model))

        if cfg.restore_event_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_event_ckpt))
            event_model.load_state_dict(torch.load(cfg.restore_event_ckpt), strict=True)
        
        if cfg.restore_nighttime_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_nighttime_ckpt))
            night_model.load_state_dict(torch.load(cfg.restore_nighttime_ckpt), strict=True)

        event_model.cuda()
        event_model.train()

        night_model.cuda()
        night_model.train()


        train_loader = datasets.fetch_dataloader(cfg)
        # set optimizer and scheduler
        params_group = [
            {'params': night_model.parameters()},
            {'params': event_model.parameters()}
        ]
        optimizer = torch.optim.AdamW(params_group, lr=cfg.trainer.canonical_lr, weight_decay=cfg.trainer.adamw_decay, eps=cfg.trainer.epsilon)
        scheduler = build_scheduler(cfg.trainer, optimizer)


        
        # optimizer, scheduler = fetch_optimizer(event_model, cfg.trainer)

        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(event_model, scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, event1, event2, flow, valid = [x.cuda() for x in data_blob]
                

                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                else:
                    image1 = image1.clamp(0.0, 255.0)
                    image2 = image2.clamp(0.0, 255.0)

                
                output = {}
                flow_image_predictions, _ = night_model(image1, image2)
                flow_event_predictions, _ = event_model(event1, event2)
                
                loss_image, metrics_image = sequence_loss(flow_image_predictions, flow, valid, cfg)
                loss_event, metrics_event = sequence_event_loss(flow_event_predictions, flow, valid, cfg)
                loss = loss_image + loss_event

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(night_model.parameters(), cfg.trainer.clip)
                torch.nn.utils.clip_grad_norm_(event_model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                dict_metric = dict(metrics_image, **metrics_event)
                # dict_metric = dict(dict_metric, **kl_metrics)
                dict_metric.update(output)
                logger.push(dict_metric)

                ### change evaluate to functions
                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'nightflow')
                    torch.save(night_model.state_dict(), PATH)

                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'eventflow')
                    torch.save(event_model.state_dict(), PATH)
                
                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = f'checkpoints/nightflow.pth'
        torch.save(night_model.state_dict(), PATH)

        PATH = f'checkpoints/eventflow.pth'
        torch.save(event_model.state_dict(), PATH)

        return PATH
    


    # 用DSEC作为数据集，使用一致性作为约束
    def train_stage_8(self):
        cfg = self.cfg
        event_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(event_model))

        night_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(night_model))

        if cfg.restore_event_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_event_ckpt))
            event_model.load_state_dict(torch.load(cfg.restore_event_ckpt), strict=True)
        
        if cfg.restore_nighttime_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_nighttime_ckpt))
            night_model.load_state_dict(torch.load(cfg.restore_nighttime_ckpt), strict=True)

        event_model.cuda()
        event_model.train()

        night_model.cuda()
        night_model.train()


        train_loader = datasets.fetch_dataloader(cfg)
        # set optimizer and scheduler
        params_group = [
            {'params': night_model.parameters()},
            {'params': event_model.parameters()}
        ]
        optimizer = torch.optim.AdamW(params_group, lr=cfg.trainer.canonical_lr, weight_decay=cfg.trainer.adamw_decay, eps=cfg.trainer.epsilon)
        scheduler = build_scheduler(cfg.trainer, optimizer)


        
        # optimizer, scheduler = fetch_optimizer(event_model, cfg.trainer)

        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(event_model, scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, event1, event2, flow, valid = [x.cuda() for x in data_blob]
                

                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                else:
                    image1 = image1.clamp(0.0, 255.0)
                    image2 = image2.clamp(0.0, 255.0)

                
                output = {}
                flow_image_predictions, _ = night_model(image1, image2)
                flow_event_predictions, _ = event_model(event1, event2)
                
                loss_image, metrics_image = sequence_loss(flow_image_predictions, flow, valid, cfg)
                loss_event, metrics_event = sequence_event_loss(flow_event_predictions, flow, valid, cfg)

                loss_flow_consist, metrics_consist = flow_consis_loss(flow_image_predictions[0], flow_event_predictions[0])

                loss = loss_image + loss_event + loss_flow_consist

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(night_model.parameters(), cfg.trainer.clip)
                torch.nn.utils.clip_grad_norm_(event_model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                dict_metric = dict(metrics_image, **metrics_event)
                dict_metric = dict(dict_metric, **metrics_consist)
                dict_metric.update(output)
                logger.push(dict_metric)

                ### change evaluate to functions
                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'nightflow')
                    torch.save(night_model.state_dict(), PATH)

                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'eventflow')
                    torch.save(event_model.state_dict(), PATH)
                
                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = f'checkpoints/nightflow.pth'
        torch.save(night_model.state_dict(), PATH)

        PATH = f'checkpoints/eventflow.pth'
        torch.save(event_model.state_dict(), PATH)

        return PATH
    
    # 用DSEC作为数据集，计算图像的时空梯度场和事件帧（直接读取即可），构建注意力图
    def train_stage_9(self):
        cfg = self.cfg
        sobel = Sobel()
        event_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(event_model))

        night_model = nn.DataParallel(build_flowformer(cfg))
        loguru_logger.info("Parameter Count: %d" % count_parameters(night_model))

        if cfg.restore_event_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_event_ckpt))
            event_model.load_state_dict(torch.load(cfg.restore_event_ckpt), strict=True)
        
        if cfg.restore_nighttime_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.restore_nighttime_ckpt))
            night_model.load_state_dict(torch.load(cfg.restore_nighttime_ckpt), strict=True)

        event_model.cuda()
        event_model.train()

        night_model.cuda()
        night_model.train()


        train_loader = datasets.fetch_dataloader(cfg)
        # set optimizer and scheduler
        params_group = [
            {'params': night_model.parameters()},
            {'params': event_model.parameters()}
        ]
        optimizer = torch.optim.AdamW(params_group, lr=cfg.trainer.canonical_lr, weight_decay=cfg.trainer.adamw_decay, eps=cfg.trainer.epsilon)
        scheduler = build_scheduler(cfg.trainer, optimizer)

        # optimizer, scheduler = fetch_optimizer(event_model, cfg.trainer)

        total_steps = 0
        scaler = GradScaler(enabled=cfg.mixed_precision)
        # scaler = GradScaler()
        logger = Logger(event_model, scheduler, cfg)

        add_noise = False

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, event1, event2, flow, valid = [x.cuda() for x in data_blob]
                
                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                else:
                    image1 = image1.clamp(0.0, 255.0)
                    image2 = image2.clamp(0.0, 255.0)

                
                output = {}
                flow_image_predictions, _ = night_model(image1, image2)
                flow_event_predictions, _ = event_model(event1, event2)

                # 合成注意力图
                img_gradx, img_grady = sobel(image1)
                # warped_img_grady = F.grid_sample(img_grady, grid_pos, mode="bilinear", padding_mode="zeros")
                img_deltaL = img_grady * flow_image_predictions[:, 0:1, :, :] + img_gradx * flow_image_predictions[:, 1:2, :, :]

                # avg_iwe = self.averaged_iwe(flow, event_list, pol_mask)
                # event_deltaL = avg_iwe[:, 0:1, :, :] - avg_iwe[:, 1:2, :, :]  # C == 1
                # 相邻时间戳很短，近似用归一化体素替代
                event_deltaL = torch.mean(event1, dim=1)

                # 训练过程中存储累积事件帧、图像时间梯度场，后面单独加

                # 点乘求相关性值
                # correlation_map = torch.norm(img_deltaL - event_deltaL, dim=1)
                correlation_map = torch.softmax(img_deltaL - event_deltaL, dim=1)
                attention_map = torch.norm(correlation_map, dim=1)

                loss_attention, metrics_attention = flow_attention_loss(flow_event_predictions, flow_image_predictions, attention_map)
                loss_image, metrics_image = sequence_loss(flow_image_predictions, flow, valid, cfg)
                loss_event, metrics_event = sequence_event_loss(flow_event_predictions, flow, valid, cfg)

                # loss_flow_consist, metrics_consist = flow_consis_loss(flow_image_predictions[0], flow_event_predictions[0])

                loss = loss_image + loss_event + loss_attention

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(night_model.parameters(), cfg.trainer.clip)
                torch.nn.utils.clip_grad_norm_(event_model.parameters(), cfg.trainer.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                dict_metric = dict(metrics_image, **metrics_event)
                dict_metric = dict(dict_metric, **metrics_attention)
                dict_metric.update(output)
                logger.push(dict_metric)

                ### change evaluate to functions
                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'nightflow')
                    torch.save(night_model.state_dict(), PATH)

                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, 'eventflow')
                    torch.save(event_model.state_dict(), PATH)
                
                total_steps += 1

                if total_steps > cfg.trainer.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = f'checkpoints/nightflow.pth'
        torch.save(night_model.state_dict(), PATH)

        PATH = f'checkpoints/eventflow.pth'
        torch.save(event_model.state_dict(), PATH)

        return PATH
    