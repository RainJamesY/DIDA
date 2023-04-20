# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

# pdb to dubug
import pdb
import numpy as np
# einops package
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    """
    In order to apply instance simliarities, we add a projection head 
    to map features into low-dimention embeddings
    """
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a feature embedding
        and a semantic segmentation map of the same size as input."""
        x = self.extract_feat(img)

        embedding, out = self._decode_head_forward_test(x, img_metas)
        # resize into a feature map of the same size as input
        x_feat = resize(    # x_feat.shape: [2, 256, 512, 512]
            input=embedding,    
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        return x_feat, out

    def encode_decode_old(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight)
        # pdb.set_trace()
        # pop out embeddings and seg_logits
        embedding = loss_decode.pop('embeddings')
        seg_logit = loss_decode.pop('seg_logits')
        losses.update(add_prefix(loss_decode, 'decode'))
        losses['embeddings'] = embedding
        losses['seg_logits'] = seg_logit

        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_feat, seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_feat, seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,  # the pixel level samantic label; shape: [2, 1, 512, 512]
                      seg_weight=None,
                      return_feat=False):
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
        
        # get cuda device
        # dev = img.device

        # debug via pdb
        # pdb.set_trace()
        # extract features from 2 input images
        x = self.extract_feat(img) # x is the multilevel 4 feature map 

        losses = dict()
        # include features in return
        if return_feat:     
            losses['features'] = x
        
        # include embeddings in return
        # losses['embeddings'] = x_feat

        # calculate normal cross entropy loss on source domain -- L_S
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        # pdb.set_trace()
        """And then obtain a fused feature embedding(map)."""
        embedding = loss_decode.pop('embeddings') # embedding.shape = [2, 256, 128, 128]
        # resize into a feature map of the same size as input
        x_feat = resize(    # x_feat.shape: [2, 256, 512, 512]
            input=embedding,    
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        # pdb.set_trace()
        losses['embeddings'] = x_feat
        losses['seg_logits'] = loss_decode.pop('seg_logits')
        losses.update(loss_decode)


        # calculate semantic similarity loss with semantic pseudo labels -- L_T
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses


    '''Obtaining feature map of input images'''
    def forward_feature_embedding(self, img):
        # get cuda device
        # dev = img.device
        # debug via pdb
        # pdb.set_trace()
        # extract features from 2 input images
        x = self.extract_feat(img) # x is the multilevel 4 feature map 

        """And then imitate the encode_decode method to obtain a  
        low-dimension embedding feature embedding(map).
        decode_features() uses feature fusion same as DAFormer
        """
        embedding = self.decode_head.decode_features(x) # embedding.shape = [2, 256, 128, 128]
        # resize into a embedding feature map of the same size as input
        x_feat = resize(    # x_feat.shape: [2, 256, 512, 512]
            input=embedding,    
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        # project into low-dimension
        # x_feat = rearrange(x_feat, 'b c h w -> b h w c')
        # embedding = self.prj_head(x_feat)   # channels 256 to 64
        # pdb.set_trace()
        # embedding = rearrange(embedding, 'b h w c -> b c h w') # rearrange back

        return x_feat


    def cal_instance_similarity(self, emb_tw, emb_ts):
        """calculate instance similarity
        with labeled souce domain embeddings stored in the buffer"""
        dev = emb_tw.device
        # unfold the embeddings(maybe not)
        # transpose them into 2 x (512*512) x 256
        emb_tw = rearrange(emb_tw, 'b c h w -> b h w c')
        emb_ts = rearrange(emb_ts, 'b c h w -> b h w c')
        # load buffer and send to cuda
        bank = torch.tensor(np.load('/mnt/data/DAFormer/embedding_cache/f_buffer.npz')['arr_0'], 
                                dtype=torch.float32)
        # transpose buffer into 256 x k
        bank = rearrange(bank, 'a b  -> b a')
        bank = bank.to(dev)
        # pdb.set_trace()
        # matrix dot product to calculate similarity(no need to select batch)
        weak_logits = emb_tw @ bank
        strong_logits = emb_ts @ bank
        # softmax on K
        sim_w = F.softmax(weak_logits, dim = 3) # weak_logits: [2, 512, 512, K]
        sim_s = F.softmax(strong_logits, dim = 3)
        # normalize
        sim_w /= torch.sum(sim_w, dim=3, keepdim=True)
        sim_s /= torch.sum(sim_s, dim=3, keepdim=True)
        # transpose back to original 
        sim_w = rearrange(sim_w, 'b h w c  -> b c h w')
        sim_s = rearrange(sim_s, 'b h w c  -> b c h w')
        return sim_w, sim_s


    def instance_similarity_loss(self, sim_w, sim_s):
        # pdb.set_trace()
        losses = dict()
        # calculate instance similarity loss -- L_in
        # sim_s is the predicted logits, sim_w is instance pseudo label
        losses['loss_in'] = torch.sum(-sim_w.detach() * torch.log(sim_s), dim=3) # cross entropy loss
        losses['loss_in'] = losses['loss_in'].mean()
        #self.decode_head.losses_in(sim_s, sim_w, seg_weight=None)
        # losses.update(loss_in)

        return losses


    '''Averaging the x_feat to 19 instance-features'''
    def instance_feature_avg(self, feat, labels, device):
        # pdb.set_trace() # debug
        batch_size1 = feat.shape[0] # batch size
        channel_size = feat.shape[1] # channel size
        # flatten feature HxW to 1 dim
        feat_flt = rearrange(feat, 'b c h w -> b c (h w)')    # feat_flt.shape: [2, 256, 512*512]
        label_flt = rearrange(labels, 'b c h w -> b c (h w)') # label_flt.shape: [2, 1, 512*512]
        # get index of each class
        class_num = 19
        # rare class sequence
        rare_seq = [18, 12, 17, 16, 7, 6, 11, 15, 4, 5, 14, 3, 9, 13, 8, 1, 10, 2, 0]
        # take one image at a time
        for k in range(batch_size1): # normally batch is 2
            # initialize a temporary cache for the 19 instances, channels = 256
            cls_cache = torch.zeros([class_num, channel_size],  dtype=torch.float32, device=device) # mind the dtype!
            # averaging instance(class), with i in rare class sequence
            for i in rare_seq:  
                srch = (label_flt==i).nonzero(as_tuple=False)   # srch: search vector locations with class i
                idx = torch.zeros(srch.shape[0],  dtype=torch.int64, device=device) # remember to add dtype and device
                for j in range(srch.shape[0]):
                    idx[j] = srch[j][2]  # flattened locations at dim=2
                # pdb.set_trace()
                # feature map for batch k, shape[256, 512*512] with channel and index. 
                feat_tmp = feat_flt[k]  
                # select channels with given index: feat_select
                feat_select = torch.index_select(feat_tmp, dim = 1, index = idx) 
                feat_mean = reduce(feat_select, 'a b -> a', reduction='mean')
                # store into cls_cache
                if len(idx)==0: # no matching class for i
                    continue
                else:   
                    cls_cache[i] = feat_mean

                # store into memory buffer
                self.f_buffer[self.buffer_idx] = feat_mean
                self.l_buffer[self.buffer_idx] = i
                print(feat_mean.shape)
            
            # pdb.set_trace()
            # store the 19 feature instances of 1 img into the buffer



    '''Averaging the x_feat to 19 instance-features'''
    # old_version class+batch
    def instance_feature_avg_old(self, feat, labels, device):
        # debug
        # pdb.set_trace()

        batch_size1 = feat.shape[0]
        # flatten feature HxW to 1 dim
        feat_flt = rearrange(feat, 'b c h w -> b c (h w)')    # feat_flt.shape: [2, 256, 512*512]
        label_flt = rearrange(labels, 'b c h w -> b c (h w)') # label_flt.shape: [2, 1, 512*512]
        # get index of each class
        class_num = 19
        # rare class sequence
        rare_seq = [18, 12, 17, 16, 7, 6, 11, 15, 4, 5, 14, 3, 9, 13, 8, 1, 10, 2, 0]
        for i in rare_seq:  # averaging instance(class), with i in rare class sequence
            srch = (label_flt==i).nonzero(as_tuple=False)   # srch: search vector locations with class i
            idx = torch.zeros(srch.shape[0],  dtype=torch.int64, device=device) # remember to add dtype and device
            for j in range(srch.shape[0]):
                idx[j] = srch[j][2]  # flattened locations at dim=2
            
            # pdb.set_trace()
            # select channels and location of each batch of class i
            for k in range(batch_size1): # normally batch is 2
                feat_tmp = feat_flt[k]  # feature map for batch k, shape[256, 512*512] with channel and index. 
                # select channels with given index: feat_select
                feat_select = torch.index_select(feat_tmp, dim = 1, index = idx) 
                feat_mean = reduce(feat_select, 'a b -> a', reduction='mean')
                # feat_mean is the averaged feature vector of class i in batch k
                print(feat_mean.shape)



    





    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                _, crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        _, seg_logit = self.encode_decode(img, img_meta) # _ to store x_feat
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
