# This file is mostly copied from https://github.com/jasonppy/FaST-VGS-Family/blob/master/datasets/spokencoco_dataset.py
# The visual data processing pipeline is copied from : https://github.com/jasonppy/word-discovery/blob/master/datasets/spokencoco_dataset.py
# All changes to the otiginal file are commented as "kh"
# The changes are manily made to 
# 1. replace Faster-RCNN image features with DINO image features.  
# 2. have a seprate methods for speech self-supervised training and validation (applied on SSL only data).
# 3. have a validation method for obtaining S similarity matrix applied in COCO-Semtest

import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets import spokencoco_dataset, places_dataset, flickr8k_dataset, libri_dataset
from datasets.sampler import StatefulSampler
from models import fast_vgs
from .utils import *
from .trainer_utils import *
from .bert_adam import BertAdam
from logging import getLogger
import logging

import numpy

from models import vit_utils

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()
class Trainer:
    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--exp_dir", type=str)
        parser.add_argument("--trained_weights_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--val_batch_size", type=int)
        parser.add_argument("--val_cross_batch_size", type=int)
        parser.add_argument("--n_epochs", type=int)
        parser.add_argument("--n_print_steps", type=int)
        parser.add_argument("--n_val_steps", type=int)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--warmup_fraction", type=float, default=0.1)
    
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args
        self.args.coarse_to_fine_retrieve = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"number of devices: {torch.cuda.device_count()}")
        self.writer = SummaryWriter(self.args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()
        self.progress, self.total_progress = setup_progress(self)
        self.dual_encoder, self.cross_encoder, self.trainables, self.indices, self.libri_indices, self.optim_states = self._setup_models()
        # now this is used to check if after VGS/VGS+ epoch we want SSL-only one.(Default: False( weight = 0))
        self.use_libri_loss = self.args.libri_w2v2_weight != 0
        
        if self.args.ssl:
            # kh: for ssl pretraining
            self.libri_train_loader, self.libri_valid_loader, self.libri_train_sampler, self.libri_train_data_length = self._setup_dataloader_ssl()
        else:
            # kh: for normal training:
            self.train_loader, self.valid_loader, self.train_sampler, self.libri_train_loader, self.libri_valid_loader, self.libri_train_sampler, self.train_data_length = self._setup_dataloader()
        
        if self.args.ssl:
            # kh: for ssl pretraining
            self.total_num_updates = int(math.floor(self.libri_train_data_length / self.args.batch_size))*self.args.n_epochs
        else:
            # kh: for normal training:
            self.total_num_updates = int(math.floor(self.train_data_length / self.args.batch_size))*self.args.n_epochs
        
        print (' ...here is total number of updates calculated at init ... ')
        print (self.total_num_updates)
        ###
        self.optimizer = self._setup_optimizer()
        


        if torch.cuda.device_count() > 1:
            self.dual_encoder = nn.DataParallel(self.dual_encoder)
            self.cross_encoder = nn.DataParallel(self.cross_encoder)
        self.scheduler = self._setup_scheduler()
        self.criterion = fast_vgs.Margin_InfoNCE_loss
        self.avportion = 16
        logger.info(f"batch size: {self.args.batch_size}")
        
    
    def forward(self, batch):
        audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls, losses = self.dual_encoder(audio_feats = batch['audio'], attention_mask = batch['audio_attention_mask'], images = batch['images'])
        coarse_cross_relationship_score_matrix = visual_cls @ audio_cls.transpose(0,1)
        losses['coarse_matching_loss'] = fast_vgs.Margin_InfoNCE_loss(coarse_cross_relationship_score_matrix, margin=self.args.margin, img_id = batch['img_id']) # [0:self.avportion]
        return losses
    
    def forward_ssl (self, libri_batch):
        losses = self.dual_encoder(audio_feats = libri_batch['audio'].to(self.device), attention_mask = libri_batch['audio_attention_mask'].to(self.device), forward_libri=True)
        return losses

    def train(self):
        print ('############# here is inside train function ###############')
        flag = True
        step_per_epoch = int(self.train_data_length/self.args.batch_size)
        #step_per_epoch_libri = int(self.libri_train_data_length/self.args.batch_size)
        data_start_time = time.time()

        while flag:
            logger.info('epoch starts here ')
            if self.use_libri_loss:
                libri_loader_iterator = iter(self.libri_train_loader)
                
            
            for i, batch in enumerate(self.train_loader):
                
                if self.use_libri_loss:
                    libri_batch = next(libri_loader_iterator)
                      
                data_end_time = time.time()
                self.dual_encoder.train()
                self.cross_encoder.train()
                if self.progress['num_updates'] > self.total_num_updates:
                    flag = False
                    r10, r5, r1 = self.validate_and_save()
                    self.writer.close()
                    break
                
                cur_lr = np.mean(self.optimizer.get_lr())
    
                self.writer.add_scalar("lr", cur_lr, self.progress['num_updates'])
                cur_step = self.progress['num_updates'] % step_per_epoch
    
                
                cur_batch = {
                        "images": batch['images'].to(self.device),
                        "audio": batch['audio'].to(self.device),
                        "audio_attention_mask": batch['audio_attention_mask'].to(self.device),
                        "img_id": batch['img_id']
                        }
                
                losses = self.forward(cur_batch)
                
                if self.use_libri_loss:
                    losses.update(self.dual_encoder(audio_feats = libri_batch['audio'].to(self.device), attention_mask = libri_batch['audio_attention_mask'].to(self.device), forward_libri=True)) 
    
                for key in losses:
                    if key in self.meters:
                        self.meters[key].update(losses[key].mean().cpu().item(), cur_batch['images'].shape[0])
                        self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                
                alpha = 0.5
                beta = 0.5
                weighted_loss = self.weight_loss(losses, alpha, beta) #self.weight_loss(losses)
    
                self.meters['weighted_loss'].update(weighted_loss.item(), cur_batch['images'].shape[0])
                self.writer.add_scalar('weighted_loss', weighted_loss.item(), self.progress['num_updates'])
                
                #########
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainables, 1.)
                self.optimizer.step()
                self.optimizer.zero_grad()
                #########
                
                self.meters['data_time'].update(data_end_time - data_start_time)
                self.meters['train_time'].update(time.time() - data_end_time)
    
                self.writer.add_scalar("data_time", data_end_time - data_start_time, self.progress['num_updates'])
                self.writer.add_scalar("train_time", time.time() - data_end_time, self.progress['num_updates'])
    
                # logging
                if self.progress['num_updates'] % self.args.n_print_steps == 0:
                    
                    log_out = {}
                    log_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
                    log_out['cur_step/steps_per_epoch'] = f"{cur_step}/{step_per_epoch}"
                    log_out['num_updates'] = self.progress['num_updates']
                    log_out['lr'] = f"{cur_lr:.7f}"
                    for key in self.meters:
                        if self.meters[key].val != 0 or self.meters[key].avg != 0:
                            log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                    logger.info(log_out)
                    if np.isnan(self.meters['weighted_loss'].avg):
                        logger.info("training diverged...")
                        return
                    
               
                # validation and save models
                if self.progress['num_updates'] % self.args.n_val_steps == 0:
                    
                    r10, r5, r1 = self.validate_and_save(libri=self.use_libri_loss, places=self.args.places, n_save_ind = self.progress['epoch'])
                ########    
                self.progress['num_updates'] += 1
                self.progress['epoch'] = int(math.ceil(self.progress['num_updates'] / step_per_epoch))
                data_start_time = time.time()
                #print(self.progress['num_updates'])


    def train_ssl(self):
        print ('############# here is inside train_ssl function ###############')
        print ('############# here is size of encoder ###############')
        print(self.args.encoder_layers)
        print ('############# here is size of attention heads ###############')
        print(self.args.encoder_attention_heads)
        print ('############# here is layer use ###############')
        print(self.args.layer_use)
        flag = True     
        # Kh: steps pers epochs based on coco
        # step_per_epoch = int(self.train_data_length/self.args.batch_size)
        # Kh: steps pers epochs based on libri
        step_per_epoch_libri = int(self.libri_train_data_length/self.args.batch_size)
        #step_per_epoch_coco = int(self.train_data_length/self.args.batch_size)
        step_per_epoch = step_per_epoch_libri
        
        
        #khazar
        print ('start of training method')
        print ('...step_per_epoch for libri is....')
        print(step_per_epoch_libri)
        ###
        data_start_time = time.time()
        
        while flag:
            logger.info('epoch starts here ')
            
            # coco_loader_iterator = iter(self.train_loader)
            # libri_loader_iterator = iter(self.libri_train_loader)
            
            # kh: iterate based on libri
            for i, libri_batch in enumerate(self.libri_train_loader): 
                
                # cur_step shows step within one epoch (0,step_per_epoch)
                cur_step = self.progress['num_updates'] % step_per_epoch
                     
                data_end_time = time.time()
                self.dual_encoder.train()
                if self.progress['num_updates'] > self.total_num_updates:
                    flag = False
                    self.validate_and_save_ssl(n_save_ind = self.progress['epoch'])
                    self.writer.close()
                    break
                
                cur_lr = np.mean(self.optimizer.get_lr())

                self.writer.add_scalar("lr", cur_lr, self.progress['num_updates'])                 
                losses = self.forward_ssl (libri_batch)

                for key in losses:
                    if key in self.meters:
                        self.meters[key].update(losses[key].mean().cpu().item(), libri_batch['audio'].shape[0])
                        self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                
                weighted_loss = losses['libri_w2v2_loss'].mean() #* self.args.libri_w2v2_weight

                self.meters['weighted_loss'].update(weighted_loss.item(), libri_batch['audio'].shape[0])
                self.writer.add_scalar('weighted_loss', weighted_loss.item(), self.progress['num_updates'])
                
                #########
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainables, 1.)
                self.optimizer.step()
                self.optimizer.zero_grad()
                #########
                
                self.meters['data_time'].update(data_end_time - data_start_time)
                self.meters['train_time'].update(time.time() - data_end_time)
   
                self.writer.add_scalar("data_time", data_end_time - data_start_time, self.progress['num_updates'])
                self.writer.add_scalar("train_time", time.time() - data_end_time, self.progress['num_updates'])

                # logging
                if self.progress['num_updates'] % self.args.n_print_steps == 0:
                    log_out = {}
                    log_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
                    log_out['cur_step/steps_per_epoch'] = f"{cur_step}/{step_per_epoch}"
                    log_out['num_updates'] = self.progress['num_updates']
                    log_out['lr'] = f"{cur_lr:.7f}"
                    for key in self.meters:
                        if self.meters[key].val != 0 or self.meters[key].avg != 0:
                            log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                    logger.info(log_out)
                    if np.isnan(self.meters['weighted_loss'].avg):
                        logger.info("training diverged...")
                        return
                    
               
                # validation and save models
                if self.progress['num_updates'] % step_per_epoch == 0:
                #if self.progress['num_updates'] % self.args.n_val_steps == 0:
                    self.validate_and_save_ssl(n_save_ind = self.progress['epoch'])
                    
                ########    
                self.progress['num_updates'] += 1
                self.progress['epoch'] = int(math.ceil(self.progress['num_updates'] / step_per_epoch))
                data_start_time = time.time()
                #print(self.progress['num_updates'])
                
    def validate_and_save_ssl(self, n_save_ind = 0):  
        
        best_ssl_loss = self.validate_libri()

        save_progress(self)
        if best_ssl_loss:
            self.progress['best_epoch'] = self.progress['epoch']
            save_path = os.path.join(self.args.exp_dir, "best_bundle.pth")
            torch.save(
                {
                    "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                    "optimizer":  self.optimizer.state_dict(),
                    "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
                },save_path )
            logger.info(f"save *best* models at {save_path} at global step {self.progress['num_updates']}")
        if self.progress['epoch'] <= 5 :
            save_path = os.path.join(self.args.exp_dir, 'E' + str(n_save_ind) + "_bundle.pth")
        elif self.progress['epoch'] > 5  and self.progress['epoch'] % 10 == 0:
            save_path = os.path.join(self.args.exp_dir, 'E' + str(n_save_ind) + "_bundle.pth")          
        else:
            save_path = os.path.join(self.args.exp_dir, "bundle.pth")
        #save_path = os.path.join(self.args.exp_dir,"bundle.pth")
        torch.save(
            {
                "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
            },save_path
        )
        logger.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['num_updates']}")
        
    def validate_and_save(self, libri=False, places=False , n_save_ind = 0):
        # khazar: I added "n_save_ind" argument to save intermediate models 
        self.dual_encoder.eval()
        self.cross_encoder.eval()
        if places:
            r10, r5, r1 = self.validate(self.valid_loader)
            r10_unseen, r5_unseen, r1_unseen = self.validate(self.valid_loader2, unseen=True)
            r10, r5, r1 = (r10+r10_unseen)/2, (r5+r5_unseen)/2, (r1+r1_unseen)/2
        else:
            r10, r5, r1 = self.validate_one_to_many()
        
        if libri:
            best_ssl_loss = self.validate_libri()
        # r1 = 0.1 # ignore validation, for debugging
        if r1 > self.progress['best_acc']:
            self.progress['best_epoch'] = self.progress['epoch']
            self.progress['best_acc'] = r1
            save_path = os.path.join(self.args.exp_dir,"best_bundle.pth")
            torch.save(
                {
                    "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                    # khazar: I commented this to reduce bundle file size
                    #"cross_encoder": self.cross_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.cross_encoder.state_dict(),
                    "optimizer":  self.optimizer.state_dict(),
                    "indices": self.train_sampler.state_dict(),
                    "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
                },save_path
            )
            logger.info(f"save *best* models at {save_path} at global step {self.progress['num_updates']}")
           
        save_progress(self)
        
        #######################################################################
        #Kh: here it saves the model in each call 
        # if self.progress['epoch'] <= 5 :
        #     save_path = os.path.join(self.args.exp_dir, 'E' + str(n_save_ind) + "_bundle.pth")
        # elif self.progress['epoch'] > 5  and self.progress['epoch'] % 25 == 0:
        #     save_path = os.path.join(self.args.exp_dir, 'E' + str(n_save_ind) + "_bundle.pth")          
        # else:
        #     save_path = os.path.join(self.args.exp_dir, "bundle.pth")
        #######################################################################    
        save_path = os.path.join(self.args.exp_dir,"bundle.pth")
        torch.save(
            {
                "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                # khazar: I commented this to reduce bundle file size
                #"cross_encoder": self.cross_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.cross_encoder.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "indices": self.train_sampler.state_dict(),
                "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
            },save_path
        )
        logger.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['num_updates']}")

        return r10, r5, r1


    def validate_one_to_many(self, hide_progress=True):
        print ("kh: it entered validate_one_to_many function ..... ")
        self.dual_encoder.eval()
        self.cross_encoder.eval()
        N_examples = self.valid_loader.dataset.__len__()
        # khazar: N_examples = 25035
        print('kh: below it should print n_example ....')
        print('N_example is ' + str(N_examples))
        with torch.no_grad():
            # get single modal representations
            audio_feats_total = [] 
            extended_audio_attention_mask_total = []
            audio_cls_total = []
            audio_img_id_total = [] # this is same order as audio_cls_total and audio_feats_total
            img_id_to_img_feats = {}
            img_img_id_list = []
            img_cls_list = [] # this is distinct, order is the same as img_img_id_list
            img_feats_list = [] # this is distinct, order is the same as img_img_id_list
            print(' kh: below is the length of valid_loader  ')
            print(len(self.valid_loader))
            for i, batch in enumerate(self.valid_loader):
                # khazar :  here it loads all validation data to batch (i = 0: N_examples/batch_size)
                # print(' i = ' +str(i))
                
                self.dual_encoder.eval()
                self.cross_encoder.eval()
                
                
                ############################################################### khazar: validation loss
                
                cur_batch = {
                        "images": batch['images'].to(self.device),
                        "audio": batch['audio'].to(self.device),
                        "audio_attention_mask": batch['audio_attention_mask'].to(self.device),
                        "img_id": batch['img_id'],
                        #"label": batch['label']
                        }
                
                loss_val = self.forward(cur_batch)
                
                key = 'vloss_av'
                #self.meters[key].update(loss_val["coarse_matching_loss"].mean().cpu().item(), self.avportion)
                self.meters[key].update(loss_val["coarse_matching_loss"].mean().cpu().item(), cur_batch['images'].shape[0])
                self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                key = 'vloss_cap'
                self.meters[key].update(loss_val['caption_w2v2_loss'].mean().cpu().item(), cur_batch['images'].shape[0])
                self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                
                
                ###############################################################
                # khazar :  for high batch sizes below line gives memory related error
                audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device),images = batch['images'].to(self.device), test = True)
                audio_cls_total.append(audio_cls)
                # visual_cls_total.append(visual_cls)
                
                # Kh : I commented below line for appending to "audio_feat_total" and "extended_audio_attention_mask_total"       
                #audio_feats_total.append(audio_feats.detach()) # still on cude after .detach(), just removed from graph, so no gradient
                #extended_audio_attention_mask_total.append(extended_audio_attention_mask.detach())
            
                
                # visual_feats_total.append(visual_feats.detach())
                detached_visual_feats = visual_feats.detach()
                audio_img_id_total.append(batch['img_id'])
                #print(' kh: below is batch[img-id]..... ')
                #print(batch['img_id'])
                
                for j, img_id in enumerate(batch['img_id']):
                    # khazar : j = 0:batch_size
                    #print(img_id)
                    #print(' j = ' + str(j))
                    if img_id not in img_id_to_img_feats:
                        img_id_to_img_feats[img_id] = detached_visual_feats[j]
                        # khazar: I commented below line
                        #img_feats_list.append(detached_visual_feats[j])
                        img_cls_list.append(visual_cls[j].detach())
                        img_img_id_list.append(img_id)
                # if i>= 110:
                #     break
            
            
            
            audio_cls_total = torch.cat(audio_cls_total)  
            #kh: audio_cls_total = audio_cls_total.cuda(device=1)
            
            img_cls_list = torch.stack(img_cls_list)
            #kh: img_cls_list = img_cls_list.cuda(device=1)

            audio_img_id_total = np.concatenate(audio_img_id_total)
            img_img_id_list = np.array(img_img_id_list)

            coarse_cross_relationship_score_matrix = img_cls_list @ audio_cls_total.transpose(0,1)
            recalls = calc_recalls_from_S_one_to_many_coarse(coarse_cross_relationship_score_matrix, row_img_id=img_img_id_list, column_img_id=audio_img_id_total)
            avg_acc_coarse = (recalls['A_r10'] + recalls['I_r10']) / 2
            avg_acc_r1_coarse = (recalls['A_r1'] + recalls['I_r1']) / 2
            self.writer.add_scalar("acc_coarse", avg_acc_coarse, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1_coarse", avg_acc_r1_coarse, self.progress['num_updates'])
            
            ############################################################### khazar: logging validation loss
            l_out = {}
            l_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
            key = 'vloss_av'
            l_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
            key = 'vloss_cap'
            l_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
            logger.info(l_out)
            ###############################################################
            
            logger.info("Coarse Retrieval Accuracy:")
            logger.info('Audio R@100 {A_r100:.3f} Image R@100 {I_r100:.3f} Average R@100 {r100_ave:.3f} over {N:d} validation pairs'.format(A_r100=recalls['A_r100'], I_r100=recalls['I_r100'], r100_ave=(recalls['A_r100']+recalls['I_r100'])/2, N=N_examples))
            logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls['A_r10'], I_r10=recalls['I_r10'], r10_ave=(recalls['A_r10']+recalls['I_r10'])/2, N=N_examples))
            logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls['A_r5'], I_r5=recalls['I_r5'], r5_ave=(recalls['A_r5']+recalls['I_r5'])/2, N=N_examples))
            logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls['A_r1'], I_r1=recalls['I_r1'], ave_r1=(recalls['A_r1']+recalls['I_r1'])/2,  N=N_examples))
            # khazar: I added below printing
            print ("..........  coarse_to_fine_retrieve is.....................")
            print(self.args.coarse_to_fine_retrieve)
            print (" .................................")

        avg_acc_r10 = (recalls['A_r10'] + recalls['I_r10']) / 2
        avg_acc_r5 = (recalls['A_r5'] + recalls['I_r5']) / 2
        avg_acc_r1 = (recalls['A_r1'] + recalls['I_r1']) / 2
        self.writer.add_scalar("acc_r10", avg_acc_r10, self.progress['num_updates'])
        self.writer.add_scalar("acc_r5", avg_acc_r5, self.progress['num_updates'])
        self.writer.add_scalar("acc_r1", avg_acc_r1, self.progress['num_updates'])
        return avg_acc_r10, avg_acc_r5, avg_acc_r1
    
    
    def validate_khazar(self):
        # frame_counts = []
        with torch.no_grad():
            # get single modal representations
            audio_feats_total = [] 
            extended_audio_attention_mask_total = []
            visual_feats_total = [] 
            #img_id_total = []
            audio_cls_total = []
            visual_cls_total = []
            for i, batch in enumerate(self.valid_loader):
                self.dual_encoder.eval()
                self.cross_encoder.eval()
                audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device), images = batch['images'].to(self.device), test = True)
                audio_cls_total.append(audio_cls)
                visual_cls_total.append(visual_cls)
                audio_feats_total.append(audio_feats.detach()) # still on cude after .detach(), just removed from graph, so no gradient
                extended_audio_attention_mask_total.append(extended_audio_attention_mask.detach())
                visual_feats_total.append(visual_feats.detach())
                #img_id_total.append(batch['img_id'])

            audio_feats_total = torch.cat(audio_feats_total)
            extended_audio_attention_mask_total = torch.cat(extended_audio_attention_mask_total)
            visual_feats_total = torch.cat(visual_feats_total)
            #img_id_total = np.concatenate(img_id_total)
    
            visual_cls_total = torch.cat(visual_cls_total)
            audio_cls_total = torch.cat(audio_cls_total)
            coarse_cross_relationship_score_matrix = audio_cls_total @ visual_cls_total.transpose(0,1)
            #recalls = calc_recalls_from_S_coarse(coarse_cross_relationship_score_matrix, img_id=img_id_total)     
            return batch,coarse_cross_relationship_score_matrix
     
    def validate_libri(self):
        with torch.no_grad():
            N = 0
            total_loss = 0
            for batch in self.libri_valid_loader:
                self.dual_encoder.eval()
                n = len(batch['audio'])
                N += n
                losses = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device),  forward_libri=True) # target_list=batch['label'],
                total_loss += losses['libri_w2v2_loss'].mean()*n
        cur_val_loss = (total_loss/N).item()
        self.writer.add_scalar("libri_val_loss", cur_val_loss, self.progress['num_updates'])
        best_ssl_loss = False
        if cur_val_loss < self.progress['best_libri_val_loss']:
            self.progress['best_libri_val_loss'] = cur_val_loss
            logger.info(f"libri validation loss: {cur_val_loss:.3f}*\n")
            best_ssl_loss = True
        else:
            logger.info(f"libri validation loss: {cur_val_loss:.3f}\n")
        return best_ssl_loss
    
    def _setup_meters(self):
        meters = {}
        meter_names = ['vloss_av', 'vloss_cap','weighted_loss', "fine_matching_loss", "coarse_matching_loss", 'caption_w2v2_loss', "libri_w2v2_loss", "caption_m_acc", "libri_m_acc",'data_time', 'train_time']
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters
   
    def _setup_models(self):
        dual_encoder = fast_vgs.DualEncoder(self.args)
        cross_encoder = fast_vgs.CrossEncoder(self.args)     
        # Khazar: change print_model = True if you want to print the whole model and not only the model parameters
        print_model_info(dual_encoder , print_model = False)
        print_model_info(cross_encoder, print_model = False)
        if self.args.trained_weights_dir != None:
            bundle = torch.load(os.path.join(self.args.trained_weights_dir, "best_bundle.pth"))
            # Kh: I change this part from loading from dual encoder to w2v2 layers
            if self.args.ssl:
                # for only CNN feature extractor part
                dual_encoder.carefully_load_state_dict_ssl(bundle['dual_encoder'])   
            else:
                dual_encoder.carefully_load_state_dict_pre(bundle['dual_encoder'])
            indices = None
            libri_indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info(f"Load trained weights from {self.args.trained_weights_dir}")
        elif self.args.validate:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth")) #khazar: change this later to best_bundle
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            #cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
            indices = None
            libri_indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info("Perform Validation")
        # khazar: change below to best_bundle for resume    
        elif self.progress['num_updates'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth"))
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            #cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
            indices = None
            #indices = bundle['indices']
            libri_indices = bundle['libri_indices']
            optim_states = bundle['optimizer']
            logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
        else:
            indices = None
            libri_indices = None
            optim_states = None

        if self.args.fb_w2v2_weights_fn and self.progress['num_updates'] <= 1 and not self.args.validate and self.args.trained_weights_dir == None:           
            b = torch.load(self.args.fb_w2v2_weights_fn)['model']
            dual_encoder.conv1_trm1_trm3.carefully_load_state_dict(b)

        if self.args.feature_grad_mult <= 0.:
            for name, p in dual_encoder.named_parameters():
                if "feature_extractor" in name:
                    p.requires_grad = False
                    
        if self.args.load_pretrained_vit != None and self.progress['num_updates'] <= 1 and not self.args.validate:
            ckpt_root = self.args.load_pretrained_vit
            ckpt_name = f"dino_{self.args.vit_arch.lower()}{str(self.args.vit_patch_size)}_pretrain_full_checkpoint.pth"
            ckpt_fn = os.path.join(ckpt_root, ckpt_name)
            vit_utils.load_pretrained_weights(dual_encoder.trm, ckpt_fn, self.args.vit_checkpoint_key, self.args.vit_arch, self.args.vit_patch_size)
                    
        trainables1 = [p for p in dual_encoder.parameters() if p.requires_grad]
        trainables2 = [p for p in cross_encoder.parameters() if p.requires_grad]
        trainables = trainables1 + trainables2

        dual_encoder.to(self.device)
        cross_encoder.to(self.device)

        return dual_encoder, cross_encoder, trainables, indices, libri_indices, optim_states
    
    def _setup_dataloader(self):
        # SpokenCOCO
        train_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='train')
        val_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='val')
        # Kh: I change use_random to False to avoide shuffling
        train_sampler = StatefulSampler(len(train_dataset), use_random=True)
        if self.progress['num_updates'] > 1 and self.indices is not None:
            train_sampler.load_state_dict(self.indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)

        if self.use_libri_loss:
            # librispeech dataloaders
            # train
            libri_train_dataset = libri_dataset.LibriDataset(self.args, split="train")
            
            # below calculates batch size of libri based on steps per epoch obtained from COCO
            ####
            step_per_epoch = int(np.floor(len(train_dataset)/self.args.batch_size))
            libri_train_bzs = libri_train_dataset.calculate_batch_size(step_per_epoch) 
            libri_train_bzs = self.args.batch_size #min(libri_train_bzs, 64)
            
            logger.info(f"librispeech train batch size: {libri_train_bzs}")
            libri_train_sampler = StatefulSampler(len(libri_train_dataset))
            if self.progress['num_updates'] > 1 and self.libri_indices is not None:
                libri_train_sampler.load_state_dict(self.libri_indices)
            libri_train_loader = torch.utils.data.DataLoader(libri_train_dataset, batch_size=libri_train_bzs, num_workers=self.args.num_workers, pin_memory=True, sampler = libri_train_sampler, collate_fn = libri_train_dataset.collate, drop_last=True)
            
            # val
            # libri_val_dataset = libri_dataset_mm.LibriDataset(self.args, split="val")
            libri_val_dataset = libri_dataset.LibriDataset(self.args, split="val")
            logger.info(f"librispeech val batch size: {self.args.libri_val_bzs}")
            libri_valid_loader = torch.utils.data.DataLoader(libri_val_dataset, batch_size=self.args.libri_val_bzs, num_workers=self.args.num_workers, pin_memory=True, collate_fn = libri_val_dataset.collate, drop_last=True)
        else:
            libri_train_loader = None
            libri_valid_loader = None
            libri_train_sampler = None
           
        return train_loader, valid_loader, train_sampler, libri_train_loader, libri_valid_loader, libri_train_sampler, len(train_dataset)

    def _setup_dataloader_ssl(self):
    
        libri_train_dataset = libri_dataset.LibriDataset(self.args, split="train")
        
        libri_train_bzs = self.args.batch_size 
        
        print ("############# here is inside LS dataloader ##################")
        print('------------- here is the n_per_epoch libri ------------')
        print(int(np.floor(len(libri_train_dataset)/libri_train_bzs)))
        ###
        
        logger.info(f"librispeech train batch size: {libri_train_bzs}")
        libri_train_sampler = StatefulSampler(len(libri_train_dataset))
        if self.progress['num_updates'] > 1 and self.libri_indices is not None:
            libri_train_sampler.load_state_dict(self.libri_indices)
        libri_train_loader = torch.utils.data.DataLoader(libri_train_dataset, batch_size=libri_train_bzs, num_workers=self.args.num_workers, pin_memory=True, sampler = libri_train_sampler, collate_fn = libri_train_dataset.collate, drop_last=True)
        
        # val
        # libri_val_dataset = libri_dataset_mm.LibriDataset(self.args, split="val")
        libri_val_dataset = libri_dataset.LibriDataset(self.args, split="val")
        logger.info(f"librispeech val batch size: {self.args.libri_val_bzs}")
        libri_valid_loader = torch.utils.data.DataLoader(libri_val_dataset, batch_size=self.args.libri_val_bzs, num_workers=self.args.num_workers, pin_memory=True, collate_fn = libri_val_dataset.collate, drop_last=True)
        
        return libri_train_loader, libri_valid_loader, libri_train_sampler, len(libri_train_dataset) # kh: I added the last return item
    
    def _setup_optimizer(self):
        optimizer = BertAdam(self.trainables, lr=self.args.lr, warmup=self.args.warmup_fraction, t_total=self.total_num_updates)
        # KH: I added this
        print('...................... we are inside setup optimizer function .......................')
        print (optimizer)
        print('...................... here printing lr .......................')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        if self.progress['num_updates'] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        optimizer.zero_grad()
        return optimizer
    
    def _setup_scheduler(self):
        pass

    def weight_loss(self, losses, alpha, beta):
        weighted_loss = losses['coarse_matching_loss'] * self.args.coarse_matching_weight * alpha 
        if 'caption_w2v2_loss' in losses:
            weighted_loss += losses['caption_w2v2_loss'].mean() * self.args.caption_w2v2_weight * (beta)           
        if 'libri_w2v2_loss' in losses:
            weighted_loss += losses['libri_w2v2_loss'].mean() * self.args.libri_w2v2_weight
        if 'caption_hubert_loss' in losses:
            weighted_loss += losses['caption_hubert_loss'].mean() * self.args.caption_hubert_weight
        if 'libri_hubert_loss' in losses:
            weighted_loss += losses['libri_hubert_loss'].mean() * self.args.libri_hubert_weight

        
        return weighted_loss
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def save_intermediate_scores(self, recall):
        pass
        


