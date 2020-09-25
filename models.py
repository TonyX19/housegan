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
from utils import combine_images_maps, rectangle_renderer,BBox,mask_to_bb,GIOU_v1,transfer_list_to_tensor
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

def compute_iou_list(x_fake,given_y,given_w,nd_to_sample,ed_to_sample,tag='fake',im_size=256):
    maps_batch = x_fake.detach().cpu().numpy()
    nodes_batch = given_y.detach().cpu().numpy()
    edges_batch = given_w.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    np.seterr(divide='ignore',invalid='ignore')
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
                key = str(i)+'_'+str(j)
                iou_dict[key] =  [iou[0][0],Giou[0][0]]
                iou_list.append(iou_dict[key])
        extracted_room_stats[stats_key].append(iou_dict)

    #np.save('./tracking/area_stats_'+serial+'_'+tag+'_pi.npy',extracted_room_stats)

    return iou_list

def compute_iou_list_v1(x_fake,given_w,nd_to_sample,ed_to_sample,tag='fake',im_size=256):
    maps_batch = x_fake.detach().cpu().numpy()
    edges_batch = given_w.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    np.seterr(divide='ignore',invalid='ignore')
    pos = 0
    iou_pos = {}
    iou_neg = {}
    iou_invalid = {}
    iou_dict = {}
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置
        inds_ed = np.where(ed_to_sample==b)
        
        mks = maps_batch[inds_nd]
        eds = edges_batch[inds_ed]

        rooms_axes = []
        for mk in mks:
            r =  im_size/mk.shape[-1]
            x0, y0, x1, y1 = np.array(mask_to_bb(mk)) * r 
            rooms_axes.append([x0, y0, x1, y1])
          
        rooms_cnt = len(rooms_axes)

        for i in range(rooms_cnt):
            axes = rooms_axes[i]
            j = i+1
            for j in range(j,rooms_cnt):
                room_cmp = rooms_axes[j]
                axes_c = room_cmp
                a_box = list(axes)
                b_box = list(axes_c)
                iou ,giou ,center_inter ,margin_inter = GIOU_v1(a_box,b_box)
                key = str(i+pos)+'_'+str(j+pos)
                iou_dict[key] =  [iou,giou,center_inter ,margin_inter]

        for ed in eds:
            s,w,d = ed
            key = str(s) + '_' + str(d)
            if w == 1:
                iou_pos[key] = iou_dict[key]
            else:
                iou_neg[key] = iou_dict[key]
        
        for k,v in iou_dict.items():
            if k in iou_pos.keys() or k in iou_neg.keys():
                continue;
            iou_invalid[k] = v
        pos += rooms_cnt

    return iou_pos,iou_neg,iou_invalid

def GIOU_v2 (center_box , margin_box ):
    "calculate GIOU  "
    '''
    boxes1 shape : shape (n, 4)
    [extracted_rooms[6][1]]
    boxes2 shape : shape (k, 4)
    [extracted_rooms[5][1]]
    gious: shape (n, k)       
    '''
    x1,y1,x2,y2,mask1 = center_box
    xx1,yy1,xx2,yy2,mask2 = margin_box

    #area1 = (x2 -x1) * (y2 -y1)  #求取框的面积\
    area1 = compute_area(mask1,[x1,y1,x2,y2])
    area2 = compute_area(mask2,[xx1,yy1,xx2,yy2])
    
    if (area1 == 0.) and (area2 == 0.):
        #n_v = mask1.clone()[0][0]*0+ torch.tensor(1.) 
        n_v = torch.tensor(1.) #resource problem
        return n_v,n_v,n_v,n_v
        

    inter_max_x = np.minimum(x2, xx2)   #求取重合的坐标及面积
    inter_max_y = np.minimum(y2, yy2)
    inter_min_x = np.maximum(x1, xx1)
    inter_min_y = np.maximum(y1, yy1)

    inter_areas = compute_area(mask1,[inter_min_x,inter_min_y,inter_max_x,inter_max_y])

    out_max_x = np.maximum(x2, xx2)  #求取包裹两个框的集合C的坐标及面积
    out_max_y = np.maximum(y2, yy2)
    out_min_x = np.minimum(x1, xx1)
    out_min_y = np.minimum(y1, yy1)
    out_w = np.maximum(0, out_max_x - out_min_x)
    out_h = np.maximum(0, out_max_y - out_min_y)

    outer_areas = out_w * out_h  ### 待优化
    union = area1 + area2 - inter_areas  #两框的总面积   利用广播机制
    ious = inter_areas / union
    gious = ious - (outer_areas - union)/outer_areas # IOU - ((C\union）/C)
    
    if area1 == 0.:
        area1 = mask1.clone()[0][0]*0+ torch.tensor(0.1) 
    iou_v1_1 = inter_areas/area1
    
    if area2 == 0:
        area2 = mask1.clone()[0][0]*0+ torch.tensor(0.1) 
    
    iou_v1_2 = inter_areas/area2
    # print("ious :",ious)
    # print("gious" ,gious)
    IOU = ious
    GIOU = gious
    center_inter = iou_v1_1
    margin_inter = iou_v1_2

    return IOU ,GIOU ,center_inter ,margin_inter

def GIOU_v3 (center_box , margin_box ):
    "calculate GIOU  "
    '''
    boxes1 shape : shape (n, 4)
    [extracted_rooms[6][1]]
    boxes2 shape : shape (k, 4)
    [extracted_rooms[5][1]]
    gious: shape (n, k)       
    '''
    x1,y1,x2,y2,mask1 = center_box
    xx1,yy1,xx2,yy2,mask2 = margin_box

    #area1 = (x2 -x1) * (y2 -y1)  #求取框的面积\
    area1 = compute_area(mask1,[x1,y1,x2,y2])
    area2 = compute_area(mask2,[xx1,yy1,xx2,yy2])
    
    if (area1 == 0.) and (area2 == 0.):
        #n_v = mask1.clone()[0][0]*0+ torch.tensor(1.) 
        n_v = torch.tensor(1.) #resource problem
        return n_v,n_v,n_v,n_v
        

    inter_max_x = np.minimum(x2, xx2)   #求取重合的坐标及面积
    inter_max_y = np.minimum(y2, yy2)
    inter_min_x = np.maximum(x1, xx1)
    inter_min_y = np.maximum(y1, yy1)

    inter_areas = compute_area(mask1,[inter_min_x,inter_min_y,inter_max_x,inter_max_y])

    out_max_x = np.maximum(x2, xx2)  #求取包裹两个框的集合C的坐标及面积
    out_max_y = np.maximum(y2, yy2)
    out_min_x = np.minimum(x1, xx1)
    out_min_y = np.minimum(y1, yy1)
    out_w = np.maximum(0, out_max_x - out_min_x)
    out_h = np.maximum(0, out_max_y - out_min_y)

    outer_areas = out_w * out_h  ### 待优化
    union = area1 + area2 - inter_areas  #两框的总面积   利用广播机制
    ious = inter_areas / union
    gious = ious - (outer_areas - union)/outer_areas # IOU - ((C\union）/C)
    
    if area1 == 0.:
        iou_v1_1 = torch.tensor(0.)
    else:
        iou_v1_1 = inter_areas/area1
    
    if area2 == 0:
        iou_v1_2 = torch.tensor(0.)
    else:
        iou_v1_2 = inter_areas/area2
    
    # print("ious :",ious)
    # print("gious" ,gious)
    IOU = ious
    GIOU = gious
    center_inter = iou_v1_1
    margin_inter = iou_v1_2

    return IOU ,GIOU ,center_inter ,margin_inter


def compute_iou_list_v2(masks,given_w,nd_to_sample,ed_to_sample,tag='fake',im_size=256):
    maps_batch = masks.detach().cpu().numpy()
    edges_batch = given_w.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    pos = 0
    iou_pos = {}
    iou_neg = {}
    iou_invalid = {}
    iou_dict = {}
    mks_idx = 0
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置
        inds_ed = np.where(ed_to_sample==b)
        mks = maps_batch[inds_nd]
        eds = edges_batch[inds_ed]

        rooms_axes = []
        for mk in mks:
            x0, y0, x1, y1 = mask_to_bb(mk) 
            rooms_axes.append([x0, y0, x1, y1,masks[mks_idx]])
            mks_idx +=1
          
        rooms_cnt = len(rooms_axes)

        for i in range(rooms_cnt):
            axes = rooms_axes[i]
            j = i+1
            for j in range(j,rooms_cnt):
                room_cmp = rooms_axes[j]
                axes_c = room_cmp
                iou ,giou ,center_inter ,margin_inter = GIOU_v2(axes,axes_c)
                #iou_c ,giou_c ,center_inter_c ,margin_inter_c = GIOU_v1(axes,axes_c) #debug 
                
                key = str(i+pos)+'_'+str(j+pos)
                iou_dict[key] =  [iou,giou,center_inter ,margin_inter]
                #,[iou_c ,giou_c ,center_inter_c ,margin_inter_c]]#debug 

        for ed in eds:
            s,w,d = ed
            key = str(s) + '_' + str(d)
            if w == 1:
                iou_pos[key] = iou_dict[key]
            else:
                iou_neg[key] = iou_dict[key]
        
        for k,v in iou_dict.items():
            if k in iou_pos.keys() or k in iou_neg.keys():
                continue;
            iou_invalid[k] = v
        pos += rooms_cnt


    return iou_pos,iou_neg,iou_invalid

def compute_iou_list_v3(masks,given_w,nd_to_sample,ed_to_sample,im_size=256):
    maps_batch = masks.detach().cpu().numpy()
    edges_batch = given_w.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    pos = 0
    iou_pos = {}
    iou_neg = {}
    iou_dict = {}
    mks_idx = 0
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置
        inds_ed = np.where(ed_to_sample==b)
        mks = maps_batch[inds_nd]
        eds = edges_batch[inds_ed]

        rooms_axes = []
        for mk in mks:
            x0, y0, x1, y1 = mask_to_bb(mk) 
            rooms_axes.append([x0, y0, x1, y1,masks[mks_idx]])
            mks_idx +=1
          
        rooms_cnt = len(rooms_axes)

        for i in range(rooms_cnt):
            axes = rooms_axes[i]
            j = i+1
            for j in range(j,rooms_cnt):
                room_cmp = rooms_axes[j]
                axes_c = room_cmp
                iou ,giou ,center_inter ,margin_inter = GIOU_v3(axes,axes_c)
                #iou_c ,giou_c ,center_inter_c ,margin_inter_c = GIOU_v1(axes,axes_c) #debug 
                
                key = str(i+pos)+'_'+str(j+pos)
                iou_dict[key] =  [iou,giou,center_inter ,margin_inter]
                #,[iou_c ,giou_c ,center_inter_c ,margin_inter_c]]#debug 

        for ed in eds:
            s,w,d = ed
            key = str(s) + '_' + str(d)
            if w == 1:
                iou_pos[key] = iou_dict[key]
            else:
                iou_neg[key] = iou_dict[key]
        
        pos += rooms_cnt


    return iou_pos,iou_neg



def compute_common_loss(real_mks,gen_mks, given_eds, nd_to_sample,  ed_to_sample,criterion):
    real_rate = compute_common_area(real_mks, given_eds, nd_to_sample,  ed_to_sample)
    fake_rate = compute_common_area(gen_mks, given_eds, nd_to_sample,  ed_to_sample)

    return criterion(fake_rate,real_rate)

def compute_common_area(masks,given_w,nd_to_sample,ed_to_sample,im_size=256): ####only positive
    edges_batch = given_w.detach().cpu().numpy()
    pos_edges_batch = edges_batch[edges_batch[:,1]>0]
    ret = torch.zeros((pos_edges_batch.shape[0],1))
    
    pos_idx = 0
    for ed in edges_batch:
        s,w,d = ed
        if w < 0:
            continue;
        
        master_mk = masks[s]
        margin_mk = masks[d]
        intersection = torch.sum(master_mk[(master_mk >0) & (margin_mk >0)])
        master_area = torch.sum(master_mk[master_mk>0])
        if master_area == 0.:
            master_area = master_area + torch.tensor(1.)
        ret[pos_idx] = intersection/master_area
        pos_idx+=1
            
    return ret

def compute_area_list(x_fake,given_y,nd_to_sample,im_size=256):
    maps_batch = x_fake.detach().cpu().numpy()
    nodes_batch = given_y.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    rooms_areas = {}
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置 
        mks = maps_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        i = 0
        for mk, nd in zip(mks, nds):
            room_type = str(np.where(nd>0)[0][0])
            x0, y0, x1, y1 = np.array(mask_to_bb(mk)) 
            area = compute_area(x_fake[inds_nd][i],[x0, y0, x1-1, y1-1])
            # if area > 0 :
            #     print([x0, y0, x1-1, y1-1])
            #     np.save("./tracking/debug_fake.npy",x_fake[inds_nd][i].detach().numpy())
            #     exit()
            if area < 0:
                area = abs(area * 0)
            area_rate = area / (mk.shape[-1]*mk.shape[-1])
            
            i+=1
            if room_type not in rooms_areas.keys():
                rooms_areas[room_type] = [area_rate]
                continue;
            rooms_areas[room_type].append(area_rate)
    return rooms_areas

def compute_area_norm_penalty(real_mask,fake_mask,given_y,nd_to_sample,criterion):
    real_area,real_shape = compute_area_list_v1(real_mask,given_y,nd_to_sample)
    fake_area,fake_shape = compute_area_list_v1(fake_mask,given_y,nd_to_sample)
    area_ret = {}
    for fr_type,f_area_list in fake_area.items():
        f_area_list = torch.stack(f_area_list)
        r_area_list = torch.stack(real_area[fr_type])
        # if sum(f_area_list) < 1000.: #####前期惩罚过大
        #     f_area_list = r_area_list
#         f_mean = f_area_list.mean()
#         avg_bias = torch.tensor(real_shape[fr_type]) * (torch.tensor(1.0) - torch.tensor(fake_avg[fr_type]))
#         l1_norm = (r_area_list - avg_bias - f_area_list).norm(p=1)#均值存在问题，就是0情况下 norm很小
        #l1_shape_norm =  (torch.FloatTensor() - torch.FloatTensor(fake_shape[fr_type])).norm(p=1) #分散成度
        fr_type_t = torch.FloatTensor(fake_shape[fr_type]).to(f_area_list.device)
        #l1_area_norm = ((r_area_list - f_area_list)/torch.FloatTensor(fake_shape[fr_type])).norm(p=1).mean() #real 和 fake 面积同差除以real shape 就是不同面积在同等尺度上的比较避免了 fake散开的情况
        #如果是real shape 没有意义 r_area_list 永远是1
        area_ret[fr_type] = criterion(f_area_list/fr_type_t,r_area_list/fr_type_t)

    return area_ret


def compute_area_list_v1(mask,given_y,nd_to_sample,im_size=256):
    maps_batch = mask.detach().cpu().numpy()
    nodes_batch = given_y.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    rooms_areas = {}
    rooms_shape = {}
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置 
        mks = maps_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        i = 0
        for nd in nds:
            room_type = str(np.where(nd>0)[0][0])
            pos = mask[inds_nd][i][mask[inds_nd][i] > 0]
            area = torch.sum(pos)
            _shape = mask[inds_nd][i][mask[inds_nd][i] > 0].size()[0]
            if _shape == 0:
                _shape = 1 #避免 real是 infinite
            i+=1
            if room_type not in rooms_areas.keys():
                rooms_areas[room_type] = [area]
                rooms_shape[room_type] = [_shape]
                continue;
            rooms_areas[room_type].append(area)
            rooms_shape[room_type].append(_shape)
    return rooms_areas,rooms_shape

def compute_area(mask,axes):
    #mask [32,32]
    x0, y0, x1, y1 = axes
    x0, y0, x1, y1 = int(x0),int(y0),int(x1),int(y1)

    device = mask.device
    area_v = torch.tensor(0.).to(device) #init
    for y_idx in range(y0,y1):
        for x_idx in range(x0,x1):
            area_v = area_v + mask[y_idx][x_idx]
    
    if area_v < 0: #gen_mks 存在负矩阵 这时坐标返回[0,0,0,0] gen_mks[0,0,0,0]仍为负所以导致面积为负数
        return torch.tensor(0.)
    #避免提前介入,面积为0 不backward
    # if area_v == 0.:
    #     print("area_v is 0")
        #area_v = area_v + mask[0][0] * 0.

    return area_v

def compute_empty_area(mask,axes):
    #mask [32,32]
    x0, y0, x1, y1 = axes
    x0, y0, x1, y1 = int(x0),int(y0),int(x1),int(y1)
    
    device = mask.device
    area_v = torch.tensor(0.).to(device) #init
    for y_idx in range(y0,y1):
        for x_idx in range(x0,x1):
            if mask[y_idx][x_idx] < 0:
                area_v = area_v + mask[y_idx][x_idx]
    
    return area_v

def compute_sparsity_penalty(masks,given_w,nd_to_sample,criterion):
    maps_batch = masks.detach().cpu().numpy()
    edges_batch = given_w.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    ret = []
    mks_idx = 0
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置
        mks = maps_batch[inds_nd]

        rooms_axes = []
        for mk in mks:
            x0, y0, x1, y1 = mask_to_bb(mk) 
            empty_area = compute_empty_area(masks[mks_idx],[x0, y0, x1, y1])
            mks_idx +=1
            ret.append(empty_area)
    ret_tensor = transfer_list_to_tensor(ret)
    object_ = torch.zeros(ret_tensor.shape[-1])
    return criterion(ret_tensor,object_)

def compute_sparsity_penalty_v1(masks,nd_to_sample,criterion):
    ret_tensor = torch.zeros(masks.shape[0])
    maps_batch = masks.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    ret = []
    mks_idx = 0
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b) #b ~ b_index #根据坐标获取位置
        mks = maps_batch[inds_nd]

        for mk in mks:
            new_mask = torch.zeros(mk.shape)
            mk_np = np.around(mk, decimals=1) #精度太低导致弱值发散
            m_x0, m_y0, m_x1, m_y1 = mask_to_bb(mk)
            if [m_x0, m_y0, m_x1, m_y1] == [0, 0, 0, 0]:
                ret_tensor[mks_idx] = torch.tensor(0.)
                mks_idx +=1
                continue;
            
            if mk_np[m_y0:m_y1,m_x0:m_x1][mk_np[m_y0:m_y1,m_x0:m_x1]>0].size == 0:
                ret_tensor[mks_idx] = torch.tensor(0.)
                mks_idx +=1
                continue;
            avg_np = mk_np[m_y0:m_y1,m_x0:m_x1][mk_np[m_y0:m_y1,m_x0:m_x1]>0].mean()
            #avg = mk[fy1:fy2,fx1:fx2][mk[fy1:fy2,fx1:fx2]>0].mean()
            sp_d = avg_np ### 对于gen_mk[0] 还是存在问题
            ####max list center axes######
            x_max_l ,y_max_l = np.where(mk_np==np.max(mk_np))
            x_max = np.median(x_max_l)
            y_max = np.median(y_max_l)
############get noising area########################
####step 1 get pos area masking ################
            mk_peak_area = (mk - sp_d)
            mk[mk_peak_area>0] = 0.
            noise_x_axis,noise_y_axis = np.array(np.where((mk>0)))
            for idx,v in enumerate(noise_x_axis):
                dist = math.sqrt(abs(v - x_max)**2 + abs(noise_y_axis[idx]-y_max)**2)
                new_mask[v][noise_y_axis[idx]] = masks[mks_idx][v][noise_y_axis[idx]] * dist
            penalty_sum = torch.sum(new_mask[new_mask>0])
            penalty_rate = ((masks[mks_idx][m_y0:m_y1,m_x0:m_x1]<0).nonzero(as_tuple=False).shape[0]/abs((m_y1 - m_y0)*(m_x1 - m_x0)))
            penalty = penalty_rate * penalty_sum
            ret_tensor[mks_idx] = penalty
            mks_idx +=1

    object_ = torch.zeros(ret_tensor.shape[-1])
    return criterion(ret_tensor,object_)

def compute_iou_penalty_norm(x_real,x_fake,given_y,given_w,nd_to_sample,ed_to_sample,serial='1'):
    fake_iou_list = compute_iou_list(x_fake,given_y,given_w,nd_to_sample,ed_to_sample,'fake')
    real_iou_list = compute_iou_list(x_real,given_y,given_w,nd_to_sample,ed_to_sample,'real')

    iou_diff = np.array(real_iou_list)-np.array(fake_iou_list)
    
    iou_norm = np.linalg.norm(iou_diff[:,0], ord=1)  
    giou_norm = np.linalg.norm(iou_diff[:,1], ord=1) 

    return iou_norm,giou_norm

def compute_iou_norm_v1(x_real,x_fake,given_w,nd_to_sample,ed_to_sample,serial='1'):
    fake_iou_pos,fake_iou_neg,fake_iou_invalid = compute_iou_list_v2(x_fake,given_w,nd_to_sample,ed_to_sample,'fake')
    real_iou_pos,real_iou_neg,real_iou_invalid = compute_iou_list_v2(x_real,given_w.data,nd_to_sample.data,ed_to_sample.data,'real')
    
    return fake_iou_pos,fake_iou_neg,fake_iou_invalid,real_iou_pos,real_iou_neg,real_iou_invalid

def compute_iou_norm_v2(x_real,x_fake,given_w,nd_to_sample,ed_to_sample,serial='1'):
    fake_iou_pos,fake_iou_neg = compute_iou_list_v3(x_fake,given_w,nd_to_sample,ed_to_sample)
    real_iou_pos,real_iou_neg = compute_iou_list_v3(x_real,given_w.data,nd_to_sample.data,ed_to_sample.data)
    
    return fake_iou_pos,fake_iou_neg,real_iou_pos,real_iou_neg

def compute_iou_norm(x_real,x_fake,given_w,nd_to_sample,ed_to_sample,serial='1'):
    fake_iou_pos,fake_iou_neg,fake_iou_invalid = compute_iou_list_v2(x_fake,given_w,nd_to_sample,ed_to_sample,'fake')
    real_iou_pos,real_iou_neg,real_iou_invalid = compute_iou_list_v2(x_real,given_w.data,nd_to_sample.data,ed_to_sample.data,'real')
    # iou_diff = np.array(real_iou_list)-np.array(fake_iou_list)
    
    # real_iou_norm = np.linalg.norm(np.array(real_iou_list)[:,0], ord=1)  
    # fake_iou_norm = np.linalg.norm(np.array(fake_iou_list)[:,0], ord=1)  
    # real_giou_norm = np.linalg.norm(np.array(real_iou_list)[:,1], ord=1) 
    # fake_giou_norm = np.linalg.norm(np.array(fake_iou_list)[:,1], ord=1) 
    
    return fake_iou_pos,fake_iou_neg,fake_iou_invalid,real_iou_pos,real_iou_neg,real_iou_invalid

def compute_penalty(D, x, x_fake, given_y=None, given_w=None, \
                             nd_to_sample=None, ed_to_sample=None, \
                             serial='1',data_parallel=None):
    gradient_penalty = compute_gradient_penalty(D, x, x_fake, given_y, given_w, \
                             nd_to_sample, ed_to_sample, \
                             data_parallel)
    iou_norm,giou_norm = compute_iou_penalty_norm(x,x_fake, given_y, given_w, \
                             nd_to_sample, ed_to_sample)
                             
    return gradient_penalty,iou_norm,giou_norm

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

def compute_div_loss_v1(D, real_x, fake_x,given_y=None, given_w=None, \
                             nd_to_sample=None, ed_to_sample=None, \
                             serial='1',data_parallel=None, p=6):
    indices = nd_to_sample, ed_to_sample
    #batch_size = torch.max(nd_to_sample) + 1
    dtype, device = real_x.dtype, real_x.device
    alpha = torch.rand((real_x.shape[0], 1, 1)).to(device)
    x_both = (alpha * real_x + (1-alpha) * fake_x).requires_grad_(True)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)

    if data_parallel:
        _output = data_parallel(D, (x_both, given_y, given_w, nd_to_sample), indices)
    else:
        _output = D(x_both, given_y, given_w, nd_to_sample)
    
    grad_outputs = torch.ones_like(_output).to(device)

    # cal f'(x)
    grad = autograd.grad(
        outputs=_output,
        inputs=x_both,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(x_both.shape[0], -1)
    div = (grad.norm(2, dim=1) ** p).mean()
    return div

def compute_div_loss(D, x, x_fake,given_y=None, given_w=None, \
                             nd_to_sample=None, ed_to_sample=None, \
                             serial='1',data_parallel=None, p=6):
    indices = nd_to_sample, ed_to_sample
    batch_size = torch.max(nd_to_sample) + 1
    dtype, device = x.dtype, x.device
    # u = torch.FloatTensor(x.shape[0], 1, 1).to(device)
    # u.data.resize_(x.shape[0], 1, 1)
    # u.uniform_(0, 1)
    u = torch.rand((x.shape[0], 1, 1)).to(device)
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
    div = (grad.norm(2, 1).norm(2, 1)  ** p).mean()
    return div



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
        #print(x.shape,x_g.shape,nd_to_sample.shape)
        # torch.Size([244, 128]) torch.Size([32, 128]) torch.Size([244])
        # nd_to_sample 1-d 节点对应序号
        validity_global = self.fc_layer_global(x_g)

        # local loss
        if False:
            x_loc = self.fc_layer_local(x)
            validity_local = add_pool(x_loc, nd_to_sample)
            validity = validity_global+validity_local
            return validity
        else:
            return validity_global
    
