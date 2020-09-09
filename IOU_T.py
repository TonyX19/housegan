import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn, is_adjacent
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from utils import bb_to_img, bb_to_vec, bb_to_seg, mask_to_bb
from PIL import Image, ImageDraw
from reconstruct import reconstructFloorplan
import svgwrite

from models import Generator
import networkx as nx
import matplotlib.pyplot as plt
from utils import ID_COLOR,combine_images_maps
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches - does not support larger batchs")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--with_boundary", action='store_true', default=True, help="include floorplan footprint")
parser.add_argument("--num_variations", type=int, default=10, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")
parser.add_argument("--target_set", type=str, default='D', help="which split to remove")

opt = parser.parse_args()
print(opt)
os.makedirs("./stats/", exist_ok=True)

def return_eq(node1, node2):
    return node1['label']==node2['label']

def compute_dist(bb1, bb2):

    x0, y0, x1, y1 = bb1
    x2, y2, x3, y3 = bb2 

    h1, h2 = x1-x0, x3-x2
    w1, w2 = y1-y0, y3-y2

    xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
    yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0

    delta_x = abs(xc2-xc1) - (h1 + h2)/2.0
    delta_y = abs(yc2-yc1) - (w1 + w2)/2.0

    return delta_x, delta_y


def retrieve_connections(nodes, room_bb):
    edges = []
    nodes = [x for x in nodes if x >= 0]
    room_bb = room_bb.reshape((-1, 4))
    for k, bb1 in enumerate(room_bb):
        for l, bb2 in enumerate(room_bb):
            if k > l:
                if is_adjacent(bb1, bb2):
                    edges.append((k, l))
    return nodes, edges

def draw_floorplan(dwg, junctions, juncs_on, lines_on):

    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])/2.0
        x2, y2 = np.array(junctions[l])/2.0
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=0.5))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])/2.0
        dwg.add(dwg.circle(center=(x, y), r=2, stroke='red', fill='white', stroke_width=1, opacity=0.75))
    return 




# Configure data loader
rooms_path = '/Users/home/Dissertation/Code/house-gan/dataset_paper/'
fp_dataset = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), split='eval',target_set=opt.target_set)
#fp_dataset.subgraphs[0] = fp_dataset.subgraphs[1]
# print(fp_dataset.subgraphs[1])
fp_loader = torch.utils.data.DataLoader(fp_dataset, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=floorplan_collate_fn)
fp_iter = tqdm(fp_loader, total=len(fp_dataset) // opt.batch_size + 1)

# Generate samples
cuda = False #True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
graphs = []
for i, batch in enumerate(fp_loader):#len(fp_iter) = all_samples/batch_size

    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch #mks is all single batch sampls combined, 39 = sampl_1.len[=2] + sampl_2.len[=2]+...+sampl_25.len
    
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    
    real_imgs_tensor = combine_images_maps(real_mks, given_nds, given_eds, \
                                               nd_to_sample, ed_to_sample)
    save_image(real_imgs_tensor, "/Users/home/Dissertation/Code/housegan/{}/{}_real.png".format("stats/", i), \
                   nrow=12, normalize=False)
    exit();
#     real_nodes = np.where(nds.detach().cpu()==1)[-1]
#     graphs.append(len(real_nodes))

# samples_per_len = defaultdict(int)
# for g_len in graphs:
#     samples_per_len[g_len] += 1

# print(samples_per_len)
