import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image, ImageDraw
from reconstruct import reconstructFloorplan
import svgwrite
from utils import bb_to_img, bb_to_vec, bb_to_seg, mask_to_bb, remove_junctions, ID_COLOR, bb_to_im_fid
from models import Generator
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num_variations", type=int, default=10, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")

opt = parser.parse_args()
print(opt)

numb_iters = 200000
exp_name = 'exp_with_graph_global_new'
target_set = 'D'
phase='eval'
checkpoint = './checkpoints/{}_{}_{}.pth'.format(exp_name, target_set, numb_iters)
#checkpoint = '/Users/home/Dissertation/Code/dataSet/house_gan/eg_exp_31_D_505000.pth'
checkpoint = '/Users/home/Dissertation/Code/dataSet/house_gan/exp_demo_D_500000.pth'
rooms_path = '/Users/home/Dissertation/Code/dataSet/dataset_paper/'
checkpoint = '/home/tony_chen_x19/dataset/exp_demo_D_500000.pth'
#rooms_path = '/home/tony_chen_x19/dataset/'
#Initialize variables
generator = Generator()
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    generator.load_state_dict(torch.load(checkpoint))
else:

    generator.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')))#['generator_model'])


# Initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set, split=phase)
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=True, collate_fn=floorplan_collate_fn)
from tqdm import tqdm
fp_iter = tqdm(fp_loader, total=len(fp_dataset_test) // opt.batch_size + 1)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Vectorize
# ------------
globalIndexReal = 0
globalIndexFake = 0
final_images = []
all_data = []

for i, batch in enumerate(fp_iter):
        
    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch
    
    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    
    # for k in range(opt.num_variations):
        # plot images
    z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
    with torch.no_grad():
        gen_mks = generator(z, given_nds, given_eds)
        
        
        all_data.append([gen_mks.detach().cpu().numpy(),nds.detach().cpu().numpy(),nd_to_sample.detach().cpu().numpy(),eds.detach().cpu().numpy(),ed_to_sample.detach().cpu().numpy()])

#np.save('./last_v_stats_n',all_data)
np.save('./housegan_stats',all_data)