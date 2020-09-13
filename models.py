import argparse
import os
import numpy as np
import math

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageOps
from utils import combine_images_maps, rectangle_renderer,BBox,mask_to_bb,GIOU
import torch.nn.utils.spectral_norm as spectral_norm
import logging
import json
import pickle


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, x.shape[-1]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_IOU_penalty_norm(x_fake,given_y,given_w,nd_to_sample,ed_to_sample,tag='fake',serial='1',im_size=256):
    IOU_penalty = [];
    maps_batch = x_fake.detach().cpu().numpy()
    nodes_batch = given_y.detach().cpu().numpy()
    edges_batch = given_w.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    np.seterr(divide='ignore',invalid='ignore') ### ignore nan
    extracted_room_stats = {}
    for b in range(batch_size):
        iou_list = []
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置
        inds_ed = np.where(ed_to_sample==b)
        
        mks = maps_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        eds = edges_batch[inds_ed]
        
        comb_img = np.ones((im_size, im_size, 3)) * 255
        extracted_rooms = []
        for mk, nd in zip(mks, nds):
            r =  im_size/mk.shape[-1]
            x0, y0, x1, y1 = np.array(mask_to_bb(mk)) * r 
            extracted_rooms.append([mk, (x0, y0, x1, y1), nd,eds])
            
        stats_key = tag +'_'+ str(b)
        extracted_room_stats[stats_key] = [extracted_rooms]
        extracted_rooms_len = len(extracted_rooms)  
          
        iou_dict = {}
        iou_list = []
        for i in range(extracted_rooms_len):
            room = extracted_rooms[i]
            mk, axes, nd,ed = room
            j = i+1
            for j in range(j,extracted_rooms_len):
                room_cmp = extracted_rooms[j]
                mk_c,axes_c, nd_c,ed_c = room_cmp
                a_box = list(axes)
                b_box = list(axes_c)
                iou,Giou = GIOU(np.array([a_box]),np.array([b_box]))
                if np.isnan(iou[0][0]):
                    iou[0][0] = 0
                if np.isnan(Giou[0][0]):
                    Giou[0][0] = -1
                key = str(i)+'_'+str(j)
                iou_dict[key] =  [iou[0][0],Giou[0][0]]
                iou_list.append(iou_dict[key])
        extracted_room_stats[stats_key].append(iou_dict)

    np.save('./tracking/area_stats_'+serial+'_'+tag+'_pi.npy',extracted_room_stats)

    return np.array(iou_list)


def compute_penalty(D, x, x_fake, given_y=None, given_w=None, \
                             nd_to_sample=None, ed_to_sample=None, \
                             serial='1',data_parallel=None):
    gradient_penalty = compute_gradient_penalty(D, x, x_fake, given_y, given_w, \
                             nd_to_sample, ed_to_sample, \
                             data_parallel)
    fake_iou_list = compute_IOU_penalty_norm(x_fake,given_y,given_w,nd_to_sample,ed_to_sample,'fake',serial)
    real_iou_list = compute_IOU_penalty_norm(x,given_y,given_w,nd_to_sample,ed_to_sample,'real',serial)
    iou_diff = real_iou_list - fake_iou_list

    iou_norm = np.linalg.norm(iou_diff[:,0], ord=1)  
    giou_norm = np.linalg.norm(iou_diff[:,1], ord=1)  

    return (gradient_penalty,iou_norm,giou_norm)

def compute_gradient_penalty(D, x, x_fake, given_y=None, given_w=None, \
                             nd_to_sample=None, ed_to_sample=None, \
                             data_parallel=None):
    indices = nd_to_sample, ed_to_sample
    batch_size = torch.max(nd_to_sample) + 1
    dtype, device = x.dtype, x.device
    u = torch.FloatTensor(x.shape[0], 1, 1).to(device)
    u.data.resize_(x.shape[0], 1, 1)
    u.uniform_(0, 1)
    logging.debug("x.shape:%s, x_fake.shape:%s,nd_to_sample.shape:%s" % (str(x.shape),str(x_fake.shape),str(nd_to_sample.shape)))
    x_both = x.data*u + x_fake.data*(1-u)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)
    grad_outputs = torch.ones(batch_size, 1).to(device)
    if data_parallel:
        _output = data_parallel(D, (x_both, given_y, given_w, nd_to_sample), indices)
    else:
        _output = D(x_both, given_y, given_w, nd_to_sample)
    grad = torch.autograd.grad(outputs=_output, inputs=x_both, grad_outputs=grad_outputs, \
                               retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False):
    block = []
    
    if upsample:
        if spec_norm:
            block.append(spectral_norm(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True)))
        else:
            block.append(torch.nn.ConvTranspose2d(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True)))
        else:        
            block.append(torch.nn.Conv2d(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(True))
    elif "tanh":
        block.append(torch.nn.Tanh())
    return block

class CMP(nn.Module):
    def __init__(self, in_channels):
        super(CMP, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            *conv_block(3*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky"))
             
    def forward(self, feats, edges=None):
        
        logging.debug("CMP:fea:%s,edgs:%s" % (str(feats.shape),str(edges.shape)))
        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        
        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        logging.debug("CMP:pos_v_src:%s" % (str(pos_v_src.shape)))
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = pooled_v_pos.scatter_add(0, pos_v_dst, pos_vecs_src)
        
        # pool negative edges
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = pooled_v_neg.scatter_add(0, neg_v_dst, neg_vecs_src)
        
        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        logging.debug("CMP:fea:%s,pooled_v_pos:%s,pooled_v_neg%s" % (str(feats.shape),str(pooled_v_pos.shape),str(pooled_v_neg.shape)))
        out = self.encoder(enc_in)
        return out
    
         
class Generator(nn.Module): ## extend nn.Module
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(138, 16 * self.init_size ** 2))
        self.upsample_1 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_2 = nn.Sequential(*conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.cmp_1 = CMP(in_channels=16)
        self.cmp_2 = CMP(in_channels=16)
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 1, 1, act="leaky"),
            *conv_block(256, 128, 3, 1, 1, act="leaky"),    
            *conv_block(128, 1, 3, 1, 1, act="tanh"))                                        
    
    def forward(self, z, given_y=None, given_w=None):
        logging.debug('gen z shape: %s' % (str(z.shape)))
        z = z.view(-1, 128)
        logging.debug('gen z shape: %s' % (str(z.shape)))
        logging.debug('given_y.shape: %s' % (str(given_y.shape)))
        # include nodes
        if True:
            y = given_y.view(-1, 10)
            z = torch.cat([z, y], 1)
        logging.debug("gen y %s ,z shape %s" % (str(y.shape),str(z.shape)))

        x = self.l1(z)    
        logging.debug("gen a_l1:x:%s w:%s" % (str(x.shape),str(given_w.shape)))  
        x = x.view(-1, 16, self.init_size, self.init_size)
        logging.debug("gen x.shape %s " % (str(x.shape)))
        x = self.cmp_1(x, given_w).view(-1, *x.shape[1:])
        logging.debug("gen x.shape %s " % (str(x.shape)))
        x = self.upsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])   
        logging.debug("gen x.shape %s " % (str(x.shape)))
        x = self.upsample_2(x)
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, *x.shape[2:])    
        logging.debug("x shape: %s" % (str(x.shape)))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            *conv_block(9, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))
        self.l1 = nn.Sequential(nn.Linear(10, 8 * 32 ** 2))
        self.cmp_1 = CMP(in_channels=16)
        self.downsample_1 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_2 = CMP(in_channels=16)
        self.downsample_2 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 2, 1, act="leaky"),
            *conv_block(256, 128, 3, 2, 1, act="leaky"),
            *conv_block(128, 128, 3, 2, 1, act="leaky"))
        
        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.fc_layer_global = nn.Sequential(nn.Linear(128, 1))
        self.fc_layer_local = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, given_y=None, given_w=None, nd_to_sample=None):
        logging.debug('dis x shape: %s' % (str(x.shape)))
        x = x.view(-1, 1, 32, 32)
        logging.debug('dis x shape: %s' % (str(x.shape)))
        logging.debug('given_y.shape: %s' % (str(given_y.shape)))
        # include nodes
        if True:
            y = self.l1(given_y)
            y = y.view(-1, 8, 32, 32)
            x = torch.cat([x, y], 1)
        x = self.encoder(x)
        x = self.cmp_1(x, given_w).view(-1, *x.shape[1:])  
        x = self.downsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])
        x = self.downsample_2(x)
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, x.shape[1])
        
        # global loss
        x_g = add_pool(x, nd_to_sample)
        validity_global = self.fc_layer_global(x_g)

        # local loss
        if False:
            x_loc = self.fc_layer_local(x)
            validity_local = add_pool(x_loc, nd_to_sample)
            validity = validity_global+validity_local
            return validity
        else:
            return validity_global
    
