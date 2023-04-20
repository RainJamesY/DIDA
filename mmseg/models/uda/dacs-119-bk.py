# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

from asyncio import base_tasks
import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, gaussian_blur_tar, color_jitter_tar, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio

# pdb to debug
import pdb

# gc to clean memory
import gc
# einops package
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# csv module
import csv

# pandas module
import pandas as pd

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        
        # Create a feature and a label memory buffer
        self.buffer_size = 300 # can be viewed as a hyperparameter
        dim = 256   
        
        # self.f_buffer = torch.randn(self.buffer_size, dim, dtype=torch.float32) # feature buffer
        # self.f_buffer = F.normalize(self.f_buffer, dim=1) # initiallize as unit random vectors
        # load saved buffer
        
        if self.buffer_size == 300:
            buffer_path = './embedding_cache/balanced avg cache 300'
        elif self.buffer_size == 500:
            buffer_path = './embedding_cache/balanced avg cache 500'
        fb_path = os.path.join(buffer_path, 'f_buffer_s.npz')
        lb_path = os.path.join(buffer_path, 'l_buffer_s.npz')
        self.f_buffer = torch.tensor(np.load(fb_path)['arr_0'], dtype=torch.float32)
        self.l_buffer = torch.tensor(np.load(lb_path)['arr_0'], dtype=torch.long) # label buffer
        
        "if load full buffer, set buffer_idx = buffer_size"
        self.buffer_idx = self.buffer_size

        """
        self.f_buffer = torch.randn(self.buffer_size, dim, dtype=torch.float32) # feature buffer
        self.f_buffer = F.normalize(self.f_buffer, dim=1) # initiallize as unit random vectors
        "else set buffer_idx -> 0"
        self.l_buffer = torch.zeros(self.buffer_size, dtype=torch.long) # label buffer
        self.buffer_idx = 0
        """

        "hyper parameters for further update buffer"
        self.update_idx = 0
        self.if_update = 0
        self.update_interval = 25
        self.momentum = 0.99
        self.update_stride = 5
        self.stop_update = 50
        
        
        # using self.register_buffer
        # self.register_buffer("f_buffer", torch.randn(self.buffer_size, dim, dtype=torch.float32)) # feature buffer
        # self.f_buffer = F.normalize(self.f_buffer, dim=1)
        # self.register_buffer("l_buffer", torch.zeros(self.buffer_size, dtype=torch.long)) # label buffer
        # self.register_buffer("buffer_idx", torch.zeros(1, dtype=torch.long)) # label idx

        "hyper parameters for calibrarion"
        self.if_calibrate = 0
        # set softmax temperature w(eak)t and s(trong)t
        self.wt = 0.1
        self.st = 0.1
        # set smooth parameter α: aggregate smooth
        self.a_smooth = 1


    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log


    @torch.no_grad()
    def instance_feature_avg(self, feat, labels, device):
        """Averaging the x_feat to 19 instance-features
        and update the memory buffer"""
        # Direct return when buffer is full and requires no update
        if self.buffer_idx >=  self.buffer_size and self.if_update == 0:
            return
        # count update_stride; defalut as < len(rare_seq)
        cnt_step = 0
        # pdb.set_trace() # debug
        batch_size1 = feat.shape[0] # batch size
        # channel_size = feat.shape[1] # channel size
        # flatten feature HxW to 1 dim
        feat_flt = rearrange(feat, 'b c h w -> b c (h w)')    # feat_flt.shape: [2, 256, 512*512]
        label_flt = rearrange(labels, 'b c h w -> b c (h w)') # label_flt.shape: [2, 1, 512*512]
        # get index of each class
        # class_num = 19
        # rare class sequence
        rare_seq = [18, 12, 17, 16, 7, 6, 11, 15, 4, 5, 14, 3, 9, 13, 8, 1, 10, 2, 0]
        # extract feature and update buffer 
        # take one image at a time
        for k in range(batch_size1): # normally batch is 2
            # averaging instance(class), with i in rare class sequence
            for i in rare_seq:  
                srch = (label_flt==i).nonzero(as_tuple=False)   # srch: search vector locations with class i
                idx = torch.zeros(srch.shape[0],  dtype=torch.int32, device=device) # remember to add dtype and device
                for j in range(srch.shape[0]):
                    idx[j] = srch[j][2]  # flattened locations at dim=2
                # feature map for batch k, shape[256, 512*512] with channel and index. 
                feat_tmp = feat_flt[k]  
                # select channels with given index: feat_select
                feat_select = torch.index_select(feat_tmp, dim = 1, index = idx) 
                feat_mean = reduce(feat_select, 'a b -> a', reduction='mean')   # take average and normalize to 1 dimension
                # store into memory buffer
                if len(idx) == 0: # no matching class for i
                    continue
                else:
                    if self.if_update == 0: # normally update buffer
                        self.update_buffer(feat_mean, i)
                    if self.if_update == 1: # allow further update
                        unequal_seq =self.is_equally_distrubuted()
                        # if there is no unequal class, update as step
                        if len(unequal_seq) == 0:
                            self.update_buffer_further(feat_mean, i)
                            cnt_step += 1
                            if cnt_step == self.update_stride:
                                return
                        # if there are missing classes, update as full rare seq
                        else:
                            break

            """If there are unequal classes during an update period,
            fill the unequal blank"""
            if len(self.is_equally_distrubuted()) > 0 and self.if_update == 1:
                # pdb.set_trace()
                for i in unequal_seq:  
                    srch = (label_flt==i).nonzero(as_tuple=False)   # srch: search vector locations with class i
                    idx = torch.zeros(srch.shape[0],  dtype=torch.int32, device=device) # remember to add dtype and device
                    for j in range(srch.shape[0]):
                        idx[j] = srch[j][2]  # flattened locations at dim=2
                    # feature map for batch k, shape[256, 512*512] with channel and index. 
                    feat_tmp = feat_flt[k]  
                    # select channels with given index: feat_select
                    feat_select = torch.index_select(feat_tmp, dim = 1, index = idx) 
                    feat_mean = reduce(feat_select, 'a b -> a', reduction='mean')   # take average and normalize to 1 dimension
                    # store into memory buffer
                    if len(idx) == 0: # no matching class for i
                        continue
                    else:
                        self.update_buffer_further(feat_mean, i)

    @torch.no_grad()
    def ema_update_bank(self, feat, labels, device):
        """ ema update the memory buffer """
        # Direct return when buffer is full and requires no update
        if self.buffer_idx >=  self.buffer_size and self.if_update == 0:
            return
        # count update_stride; defalut as < len(rare_seq)
        cnt_step = 0
        # pdb.set_trace() # debug
        batch_size1 = feat.shape[0] # batch size
        # channel_size = feat.shape[1] # channel size
        # flatten feature HxW to 1 dim
        feat_flt = rearrange(feat, 'b c h w -> b c (h w)')    # feat_flt.shape: [2, 256, 512*512]
        label_flt = rearrange(labels, 'b c h w -> b c (h w)') # label_flt.shape: [2, 1, 512*512]
        # get index of each class
        class_num = 19
        # channel size
        channel_size = 256
        # rare class sequence
        rare_seq = [18, 12, 17, 16, 7, 6, 11, 15, 4, 5, 14, 3, 9, 13, 8, 1, 10, 2, 0]
        # extract feature and update buffer 
        # take one image at a time
        for k in range(batch_size1): # normally batch is 2
            # only update with 1 image
            if k == 1:
                break
            # averaging instance(class), with i in rare class sequence
            # initialize a temporary cache for the 19 instances, channels = 256
            cls_cache = torch.zeros(class_num, channel_size,  dtype=torch.float32, device=device) # mind the dtype!
            for i in rare_seq:  
                srch = (label_flt==i).nonzero(as_tuple=False)   # srch: search vector locations with class i
                idx = torch.zeros(srch.shape[0],  dtype=torch.int32, device=device) # remember to add dtype and device
                for j in range(srch.shape[0]):
                    idx[j] = srch[j][2]  # flattened locations at dim=2
                # feature map for batch k, shape[256, 512*512] with channel and index. 
                feat_tmp = feat_flt[k]  
                # select channels with given index: feat_select
                feat_select = torch.index_select(feat_tmp, dim = 1, index = idx) 
                feat_mean = reduce(feat_select, 'a b -> a', reduction='mean')   # take average and normalize to 1 dimension
                # store into memory buffer
                if len(idx) == 0: # no matching class for i
                    continue
                else:
                    cls_cache[i] = feat_mean
            # after storing features in cls_cache [2, 19, 256]
            # Ema update to buffer
            # pdb.set_trace()
            self.l_buffer = self.l_buffer.to(device)
            self.f_buffer = self.f_buffer.to(device)
            self.l_buffer.detach()
            self.f_buffer.detach()

            label_expand = self.l_buffer.expand(256, self.buffer_size)
            label_expand = rearrange(label_expand, 'c n-> n c') # reshape to [K, 256] 
            cash_in = cls_cache.gather(0, label_expand) # cache_in [k, 256]

            # pdb.set_trace()
            # ema update buffer
            self.f_buffer = self.momentum * self.f_buffer + (1-self.momentum) * cash_in

            self.l_buffer = self.l_buffer.to('cpu')
            self.f_buffer = self.f_buffer.to('cpu')


    @torch.no_grad()
    def is_missing(self):
        rare_seq = [18, 12, 17, 16, 7, 6, 11, 15, 4, 5, 14, 3, 9, 13, 8, 1, 10, 2, 0]
        missing_seq = []
        for cls in rare_seq:
            if cls not in self.l_buffer: 
                missing_seq.append(cls)
        return missing_seq

    @torch.no_grad()
    def is_equally_distrubuted(self):
        rare_seq = [18, 12, 17, 16, 7, 6, 11, 15, 4, 5, 14, 3, 9, 13, 8, 1, 10, 2, 0]
        unequal_seq = []
        l_buffer = self.l_buffer.tolist()
        for cls in rare_seq:
            if l_buffer.count(cls) < ((self.buffer_size)/len(rare_seq)-1):
                unequal_seq.append(cls)
        return unequal_seq  

    @torch.no_grad()
    def update_buffer(self, feat, cls):
        """update the memory buffera and prepare for
        further sampling implementation"""
        # self.f_buffer = self.f_buffer.to(device)
        # pdb.set_trace()
        if self.buffer_idx < self.buffer_size:
            self.f_buffer[self.buffer_idx] = feat
            self.l_buffer[self.buffer_idx] = cls
            self.buffer_idx += 1
        else:
            return

    @torch.no_grad()
    def update_buffer_further(self, feat, cls):
        """update the memory buffera and prepare for
        further sampling implementation"""
        if self.update_idx < self.buffer_size:
            self.f_buffer[self.update_idx] = feat
            self.l_buffer[self.update_idx] = cls
            self.update_idx += 1
        else:
            self.update_idx = 0  # reset new pointer
            return



    @torch.no_grad()
    def cal_instance_similarity_w(self, emb_tw):
        """calculate instance similarity
        with labeled souce domain embeddings stored in the buffer"""
        dev = emb_tw.device
        # transpose them into 2 x 512 x 512 x 256
        emb_tw = rearrange(emb_tw, 'b c h w -> b h w c')
        # transpose buffer into 256 x K
        self.f_buffer = rearrange(self.f_buffer , 'a b  -> b a')
        self.f_buffer = self.f_buffer.to(dev)
        self.f_buffer.detach()

        # matrix dot product to calculate similarity(no need to select batch)
        sim_w = emb_tw @ self.f_buffer

        self.f_buffer = rearrange(self.f_buffer , 'b a  -> a b')
        self.f_buffer = self.f_buffer.to('cpu')
        # sim_w /= torch.sum(sim_w, dim=3, keepdim=True)
        sim_w = rearrange(sim_w, 'b h w c  -> b c h w')
        if self.a_smooth < 1 or self.if_calibrate == 0:
            sim_w /= torch.sum(sim_w, dim=1, keepdim=True) 
        # sim_w = torch.softmax(sim_w, dim=1)
        # softmax on K
        # sim_w = self.softmax(sim_w / self.wt, dim=1)
        # sim_w = F.softmax(sim_w / self.wt, dim=1)
        # pdb.set_trace()
        # del temp variables
        del emb_tw
        return sim_w

    def cal_instance_similarity_s(self, emb_ts):
        """calculate instance similarity
        with labeled souce domain embeddings stored in the buffer"""
        dev = emb_ts.device
        # transpose them into 2 x 512 x 512 x 256
        emb_ts = rearrange(emb_ts, 'b c h w -> b h w c')
        # transpose buffer into 256 x K
        self.f_buffer = rearrange(self.f_buffer , 'a b  -> b a')
        self.f_buffer = self.f_buffer.to(dev)
        self.f_buffer.detach()

        # matrix dot product to calculate similarity(no need to select batch)
        sim_s = emb_ts @ self.f_buffer
        
        # transpose back
        self.f_buffer = rearrange(self.f_buffer , 'b a  -> a b')
        self.f_buffer = self.f_buffer.to('cpu')

        # transpose back to original 
        sim_s = rearrange(sim_s, 'b h w c  -> b c h w') # torch.Size([2, K, 512, 512])
        sim_s /= torch.sum(sim_s, dim=1, keepdim=True)
        # softmax on K
        # sim_s = self.softmax(sim_s / self.st, dim=1)
        # del temp variables
        del emb_ts
        return sim_s

    def instance_similarity_loss(self, sim_w, sim_s):
        # sim_w.shape: torch.Size([2, K, 512, 512]); K is buffer size
        losses = dict()
        # calculate instance similarity loss -- L_in
        # sim_s is the predicted logits, sim_w is instance pseudo label
        # add near_0 to avoid NaN
        near_0 = 1e-10
        losses['loss_in'] = torch.sum(-sim_w.detach() * torch.log(sim_s + near_0), dim=1) # cross entropy loss on K
        losses['loss_in'] = losses['loss_in'].mean()
        # free GPU space
        del sim_w, sim_s
        return losses
    
    # Main alteration in this part
    # We first apply weak augmentation for target_img in this fuction
    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        tar_batch_size = target_img.shape[0]
        dev = img.device

        """reset buffer pointer self.buffer_idx"""
        if self.buffer_idx >= self.buffer_size: # when buffer is full
            if self.local_iter % self.update_interval == 0:       # update at every interval iter
                self.if_update = 1
            else:
                self.if_update = 0

        "stop normal update when the bank is equally distributed"
        # if self.local_iter > self.stop_update:
        # if len(self.is_equally_distrubuted()) == 0:
        #     self.if_update = 0

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        # Strong augmentation parameters
        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Weak augmentation parameters
        means_w, stds_w = get_mean_std(target_img_metas, dev)
        weak_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means_w[0].unsqueeze(0),  # assume same normalization
            'std': stds_w[0].unsqueeze(0)
        }   
        

        # Train on source images --> loss L_S
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        
        # Obtaining source image feature embedding_s(source): torch.Size([2, 256, 512, 512]) 
        embedding_s = clean_losses.pop('embeddings')

        # Obtaining source domain seg_logit_s(source): torch size [2, 19, 512, 512]
        seg_logit_s = clean_losses.pop('seg_logits')
        
        # averaging instance features and update memory buffer
        # self.instance_feature_avg(embedding_s, gt_semantic_seg, device=dev)


        if self.if_update:
            self.ema_update_bank(embedding_s, gt_semantic_seg, device=dev)


        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        # pdb.set_trace()
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')


        # ImageNet feature distance --> loss L_FD
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        
        # Applying weak augmentation to target images before generating pseudo labels --> w_target_img
        w_target_img = target_img
        w_target_img_metas = target_img_metas
        for i in range(tar_batch_size):
            # Apply gaussian blur
            w_target_img[i] = gaussian_blur_tar(blur = weak_parameters['blur'], data = target_img[i])
            # target_img[i] = gaussian_blur_tar(blur = weak_parameters['blur'], data = target_img[i])
            # Apply color jitter on Gussian blurred w_target_img
            w_target_img[i] = color_jitter_tar(
                color_jitter=weak_parameters['color_jitter'],
                s=weak_parameters['color_jitter_s'],
                p=weak_parameters['color_jitter_p'],
                mean=weak_parameters['mean'],
                std=weak_parameters['std'],
                data=w_target_img[i])

            # Apply flip/crop
        
        # Generate semantic pseudo-label 
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        """obtaining weakly augmented target domain feature embeddings embedding_tw 
        and ema_logits """
        embedding_tw, ema_logits = self.get_ema_model().encode_decode( # with weakly augmented target images
            w_target_img, w_target_img_metas)
        # ema_logits: torch.Size([2, 19, 512, 512]) is the psudo label after encode_decode

        in_sim_w = self.cal_instance_similarity_w(embedding_tw)
        # pdb.set_trace()
        """in_sim_w calibrate with semantic similarity -> fuse_sim_w"""
        "Unfold semantic seg_logits when buffer is full"
        if self.buffer_idx >= self.buffer_size and self.if_calibrate:
            # pdb.set_trace()
            with torch.no_grad():
                # seg_logit_s: [2, 19, 512, 512] --unfold--> [2, K, 512, 512], K is buffer size
                self.l_buffer = self.l_buffer.to(dev)
                self.l_buffer.detach()
                label_expand = self.l_buffer.expand(2, 512, 512, self.buffer_size) # expand to [2, 512, 512, K]
                # self.l_buffer = self.l_buffer.to('cpu')
                label_expand = rearrange(label_expand, 'b h w c -> b c h w') # reshape to [2, K, 512, 512]
                """unfold seg_logits from 19 -> seg_logit_uf(unfold)"""
                seg_logit_uf = seg_logit_s.gather(1, label_expand) # costs large memory

                if self.a_smooth < 1:
                    """aggregate in_sim_w from K to 19"""
                    sim_w_agg = ema_logits.scatter_add(1, label_expand, in_sim_w) # out.scatter_add(dim, idx, src)
                    # smoothing with sim_w_agg
                    """aggregate instance pseudo labels with semantic ones"""
                    # pdb.set_trace()
                    ema_logits = ema_logits * self.a_smooth + sim_w_agg * (1-self.a_smooth)
                
                self.l_buffer = self.l_buffer.to('cpu')
                # regenerate in_sim_w as fuse_sim_w [2, K, 512, 512]
                fuse_sim_w = in_sim_w * seg_logit_uf
                # normalize on K
                fuse_sim_w /= torch.sum(fuse_sim_w, dim=1, keepdim=True) 

            # free GPU space
            del label_expand, seg_logit_uf
                
        # pdb.set_trace()
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1) # [2, 19, 512, 512], dim 1 is the class channel
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing, following DACS
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images --> loss L_T
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True) # why is gt_semantic_seg missing here: mixed_lbl contains it
        mix_losses.pop('features')
        mix_losses.pop('seg_logits')
        
        embedding_ts = mix_losses.pop('embeddings')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward(retain_graph=True)
        # mix_loss.backward()

        """Calculate instance level similarity and Loss L_in
            with embedding_tw from weakly augmented view,
            embedding_ts from strongly augmented view,
            and labeled embedding_s from source domain data in the memory buffer
        """
        # Generate instance pseudo-label --> in_pseudo = in_sim_w
        # obtain in_sim_w, in_sim_s to calculate loss
        """obtaining strongly augmented target domain feature embeddings embedding_ts """
        in_sim_s = self.cal_instance_similarity_s(embedding_ts)
        # in_sim_s already normalized
        # in_sim_w.shape: [2, K, 512, 512]

        # pdb.set_trace()
        """calculate loss_in with sim_w and sim_s"""
        if self.buffer_idx >= self.buffer_size and self.if_calibrate:
            instance_losses = self.instance_similarity_loss(fuse_sim_w, in_sim_s)
        else:
            # in_sim_w /= torch.sum(in_sim_w, dim=1, keepdim=True) 
            instance_losses = self.instance_similarity_loss(in_sim_w, in_sim_s)

        instance_losses = add_prefix(instance_losses, 'instance')
        in_loss, in_log_vars = self._parse_losses(instance_losses)
        log_vars.update(in_log_vars)
        
        # delete tmp variables
        del in_sim_w, in_sim_s
        in_loss.backward()
        # mix1_loss = mix_loss + in_loss
        # mix1_loss.backward()

        """ delete feature maps to free GPU space"""
        del embedding_s, embedding_ts, embedding_tw

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        if (self.local_iter-1) % self.update_interval == 0  and self.if_update == 1 :
            # write loss_in to csv
            path_cfg = self.train_cfg['work_dir']
            in_loss_out = os.path.join(path_cfg, 'ablation_study/loss_in')
            os.makedirs(in_loss_out, exist_ok=True)
            iter = self.local_iter
            loss_in = log_vars['instance.loss_in']
            df = pd.DataFrame({'a_name':[iter-1],'b_name':[loss_in]})
            save_loss = os.path.join(in_loss_out, 'train_in_loss.csv')
            df.to_csv(save_loss, mode = 'a', index=False, header=False)

            rare_seq1 = [18, 12, 17, 16, 7, 6, 11, 15, 4, 5, 14, 3, 9, 13, 8, 1, 10, 2, 0]
            cnt_cls = []
            cnt = 0
            for cls in rare_seq1:
                for i in self.l_buffer:
                    if cls == i:
                        cnt += 1
                cnt_cls.append(cnt)
                cnt = 0
            # write l_buffer count to csv after each update
            bank_cnt_out = os.path.join(path_cfg, 'ablation_study/label_buffer')
            os.makedirs(bank_cnt_out, exist_ok=True)
            save_bank = os.path.join(bank_cnt_out, 'bank_cnt.csv')
            df = pd.DataFrame({'a_name':[iter-1],'b_name':[cnt_cls]})
            df.to_csv(save_bank, mode = 'a', index=False, header=False)

        # saving hyper parameters:
        if self.local_iter-1 == 0:
            hyper_out = self.train_cfg['work_dir']
            path1 = os.path.join(hyper_out, 'hyper_parameters.csv')
            df1 = pd.DataFrame({'buffer_size':[self.buffer_size],
                                'smooth_agg':[self.a_smooth],
                                'update_interval':[self.update_interval], 
                                'update_stride':[self.update_stride],
                                'stop_update':[self.stop_update],
                                'if_calibrate':[self.if_calibrate]})
            df1.to_csv(path1, mode = 'a', index=False, header=True)

        # save f_buffer and l_buffer when stop updating
        if self.buffer_idx >= self.buffer_size and self.if_update == 0:
            # pdb.set_trace()
            f_bank = self.f_buffer.cpu().detach().numpy()
            l_buffer = self.l_buffer.cpu().detach().numpy()
            np.savez(r'/mnt/data/DAFormer/embedding_cache/f_buffer_s.npz', f_bank)
            np.savez(r'/mnt/data/DAFormer/embedding_cache/l_buffer_s.npz', l_buffer)
        
        return log_vars
