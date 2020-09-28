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
from utils import combine_images_maps, rectangle_renderer,transfer_list_to_tensor
from models import Discriminator, Generator, compute_div_loss_v1, weights_init_normal,compute_gradient_penalty,compute_area_norm_penalty,compute_area_norm_penalty_v1,compute_sparsity_penalty,compute_common_loss,compute_sparsity_penalty_v1,compute_sparsity_penalty_v2,compute_sparsity_penalty_v3,compute_avg_loss\
    ,compute_area_norm_penalty_v2
import os
from datetime import datetime
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--optim", type=str, default='adam', help="adam: learning rate")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--d_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--target_set", type=str, default='D', help="which split to remove")
parser.add_argument("--eloss_lim", type=int, default=1, help="extra_loss_limitation")
parser.add_argument("--is_mean", type=bool, default=False, help="extra_loss_mean")
parser.add_argument("--debug", type=bool, default=False, help="debug")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
opt = parser.parse_args()
debug = opt.debug


import logging
if debug : ## debug variable impact the rest of packages
    logging.basicConfig(level=logging.DEBUG)



extra_loss_lim = opt.eloss_lim
cuda = True if torch.cuda.is_available() else False
lambda_gp = 10
multi_gpu = False
is_mean = opt.is_mean
# exp_folder = "{}_{}_g_lr_{}_d_lr_{}_bs_{}_ims_{}_ld_{}_b1_{}_b2_{}".format(opt.exp_folder, opt.target_set, opt.g_lr, opt.d_lr, \
#                                                                         opt.batch_size, opt.img_size, \
#                                                                         opt.latent_dim, opt.b1, opt.b2)
exp_folder = "{}_{}".format(opt.exp_folder, opt.target_set)
os.makedirs("./exps/"+exp_folder, exist_ok=True)
os.makedirs("./exps/"+exp_folder, exist_ok=True)
os.makedirs("./temp/", exist_ok=True)
os.makedirs("./tracking/", exist_ok=True)
os.makedirs("./checkpoints/", exist_ok=True)

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Support to multiple GPUs
def graph_scatter(inputs, device_ids, indices):
    nd_to_sample, ed_to_sample = indices
    batch_size = (torch.max(nd_to_sample) + 1).detach().cpu().numpy()
    N = len(device_ids)
    shift = np.round(np.linspace(0, batch_size, N, endpoint=False)).astype(int)
    shift = list(shift) + [int(batch_size)] 
    outputs = []
    for i in range(len(device_ids)):
        if len(inputs) <= 3:
            x, y, z = inputs
        else:
            x, y, z, w = inputs
        inds = torch.where((nd_to_sample>=shift[i])&(nd_to_sample<shift[i+1]))[0]
        x_split = x[inds]
        y_split = y[inds]
        inds = torch.where(nd_to_sample<shift[i])[0]
        min_val = inds.size(0)      
        inds = torch.where((ed_to_sample>=shift[i])&(ed_to_sample<shift[i+1]))[0]
        z_split = z[inds].clone()
        z_split[:, 0] -= min_val
        z_split[:, 2] -= min_val
        if len(inputs) > 3:
            inds = torch.where((nd_to_sample>=shift[i])&(nd_to_sample<shift[i+1]))[0]
            w_split = (w[inds]-shift[i]).long()            
            _out = (x_split.to(device_ids[i]), \
                    y_split.to(device_ids[i]), \
                    z_split.to(device_ids[i]), \
                    w_split.to(device_ids[i]))
        else:   
            _out = (x_split.to(device_ids[i]), \
                    y_split.to(device_ids[i]), \
                    z_split.to(device_ids[i]))
        outputs.append(_out)
    return outputs

def data_parallel(module, _input, indices):
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = graph_scatter(_input, device_ids, indices)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)

# # Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# Visualize a single batch
def visualizeSingleBatch(fp_loader_test, opt,):
    with torch.no_grad():
        # Unpack batch
        mks, nds, eds, nd_to_sample, ed_to_sample = next(iter(fp_loader_test))
        real_mks = Variable(mks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds
                                    
        # Generate a batch of images
        z_shape = [real_mks.shape[0], opt.latent_dim]
        z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
        gen_mks = generator(z, given_nds, given_eds)
            
        # Generate image tensors
        real_imgs_tensor = combine_images_maps(real_mks, given_nds, given_eds, \
                                               nd_to_sample, ed_to_sample)
        fake_imgs_tensor = combine_images_maps(gen_mks, given_nds, given_eds, \
                                               nd_to_sample, ed_to_sample)
        #iou_norm,giou_norm = compute_iou_list(real_mks,given_y,given_w,nd_to_sample,ed_to_sample,'valid')
        # Save images
        #np.save()
        save_image(real_imgs_tensor, "./exps/{}/{}_real.png".format(exp_folder, batches_done), \
                   nrow=12, normalize=False)
        save_image(fake_imgs_tensor, "./exps/{}/{}_fake.png".format(exp_folder, batches_done), \
                   nrow=12, normalize=False)
        
        extracted_room_stats = [gen_mks.detach().cpu().numpy(),real_mks.detach().cpu().numpy(), given_nds.detach().cpu().numpy(), given_eds.detach().cpu().numpy(), \
                                    nd_to_sample.detach().cpu().numpy(), ed_to_sample.detach().cpu().numpy()]
        np.save('./tracking/vaild_stats_'+exp_folder+str(batches_done)+'.npy',extracted_room_stats)
        #return iou_norm,giou_norm

def visualizeBatch(real_mks,gen_mks,given_nds,given_eds,nd_to_sample,ed_to_sample):
    with torch.no_grad():
        imgs_tensor = combine_images_maps(gen_mks, given_nds, given_eds, \
                                                nd_to_sample, ed_to_sample)
        save_image(imgs_tensor,"./exps/{}/{}_train_gen.png".format(exp_folder, batches_done), \
                    nrow=12, normalize=False)
        imgs_tensor = combine_images_maps(real_mks, given_nds, given_eds, \
                                                nd_to_sample, ed_to_sample)
        save_image(imgs_tensor,"./exps/{}/{}_train_real.png".format(exp_folder, batches_done), \
                    nrow=12, normalize=False)
        extracted_room_stats = [gen_mks.detach().cpu().numpy(),real_mks.detach().cpu().numpy(), given_nds.detach().cpu().numpy(), given_eds.detach().cpu().numpy(), \
                                    nd_to_sample.detach().cpu().numpy(), ed_to_sample.detach().cpu().numpy()]
        np.save('./tracking/train_stats_'+exp_folder+str(batches_done)+'.npy',extracted_room_stats)
    return
    
if __name__ == '__main__':
    # Configure data loader
    rooms_path = '/Users/home/Dissertation/Code/dataSet/dataset_paper/' # replace with your dataset path need abs path
    #rooms_path = '/home/tony_chen_x19/dataset/'
    fp_dataset_train = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=opt.target_set)
    fp_loader = torch.utils.data.DataLoader(fp_dataset_train, 
                                            batch_size=opt.batch_size, 
                                            shuffle=True,
                                            num_workers=opt.n_cpu,
                                            collate_fn=floorplan_collate_fn)

    fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=opt.target_set, split='eval')
    fp_loader_test = torch.utils.data.DataLoader(fp_dataset_test, 
                                            batch_size=64, 
                                            shuffle=False,
                                            num_workers=opt.n_cpu,
                                            collate_fn=floorplan_collate_fn)

    # Optimizers
    if opt.optim == 'adam' :
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2)) 
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
    else:
        #RMSprop
        print("traning by useing RMSprop")
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.g_lr,alpha=0.9) 
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.d_lr, alpha=0.9)


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    print('training')
    # ----------
    #  Training
    # ----------
    batches_done = 0
    BCE_logitLoss = nn.BCEWithLogitsLoss()
    BCE_loss = nn.BCELoss()
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    sig = nn.Sigmoid()
    for epoch in range(opt.n_epochs):
        for b_idx, batch in enumerate(fp_loader):
            # Unpack batch
            mks, nds, eds, nd_to_sample, ed_to_sample = batch
            logging.debug("mks: %s nds:%s nd_to_sample:%s" % (str(mks.shape),str(nds.shape),str(nd_to_sample.shape)))
            indices = nd_to_sample, ed_to_sample
            # Adversarial ground truths
            batch_size = torch.max(nd_to_sample) + 1
            valid = Variable(Tensor(batch_size, 1)\
                            .fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batch_size, 1)\
                            .fill_(0.0), requires_grad=False)
        
            # Configure input
            real_mks = Variable(mks.type(Tensor))
            logging.debug('real_mks: %s' % (str(real_mks.shape)))
            given_nds = Variable(nds.type(Tensor))
            given_eds = eds
            
            # Set grads on
            for p in discriminator.parameters():
                p.requires_grad = True
                # # WGAN需要将判别器的参数绝对值截断到不超过一个固定常数c
                # p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
                
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate a batch of images
            z_shape = [real_mks.shape[0], opt.latent_dim] # latent_dim ~ to map high dim space
            logging.debug("z.shape %s" % (str(z_shape)))
            z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
            logging.debug("z.shape after N %s" % (str(z.shape)))
            if multi_gpu:
                gen_mks = data_parallel(generator, (z, given_nds, given_eds), indices)
            else:
                gen_mks = generator(z, given_nds, given_eds)
            
            # Real images
            if multi_gpu:
                real_validity = data_parallel(discriminator, \
                                            (real_mks, given_nds, \
                                            given_eds, nd_to_sample), \
                                            indices)
            else:
                real_validity = discriminator(real_mks, given_nds, given_eds, nd_to_sample)
                #[32, 1]
            # y=A(x), z=B(y) 求B中参数的梯度，不求A中参数的梯度
            # # 第一种方法
            # y = A(v1)
            # z = B(y.detach()) # 直接用y的value 训练B(y)
            # z.backward()

            # Fake images
            if multi_gpu:
                fake_validity = data_parallel(discriminator, \
                                            (gen_mks.detach(), given_nds.detach(), \
                                            given_eds.detach(), nd_to_sample.detach()),\
                                            indices)
            else:
                fake_validity = discriminator(gen_mks.detach(), given_nds.detach(), \
                                            given_eds.detach(), nd_to_sample.detach())

            # Measure discriminator's ability to classify real from generated samples
            k = 2
            p = 6
            if multi_gpu:
                div_loss = compute_div_loss_v1(discriminator, real_mks.data, \
                                                            gen_mks.data, given_nds.data, \
                                                            given_eds.data, nd_to_sample.data,\
                                                             ed_to_sample.data,str(batches_done),data_parallel,p=p)
                # div_loss = compute_div_loss(discriminator, real_mks.data, \
                #                                             gen_mks.data, given_nds.data, \
                #                                             given_eds.data, nd_to_sample.data,\
                #                                              ed_to_sample.data,str(batches_done),data_parallel,p=p)
            else:
                div_loss = compute_div_loss_v1(discriminator, real_mks.data, \
                                                            gen_mks.data, given_nds.data, \
                                                            given_eds.data, nd_to_sample.data, \
                                                            ed_to_sample.data,str(batches_done),None,p=p)
                # gradient_penalty = compute_gradient_penalty(discriminator, real_mks.data, \
                #                                             gen_mks.data, given_nds.data, \
                #                                             given_eds.data, nd_to_sample.data, \
                #                                             None, None)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + k*div_loss
            #+ lambda_gp * gradient_penalty
            

            # Update discriminator
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Set grads off
            for p in discriminator.parameters():
                p.requires_grad = False
                
            # Train the generator every n_critic steps
            if b_idx % opt.n_critic == 0:
                
                # Generate a batch of images
                z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
                gen_mks = generator(z, given_nds, given_eds)

                # Score fake images
                if multi_gpu:
                    fake_validity = data_parallel(discriminator, \
                                                (gen_mks, given_nds, \
                                                given_eds, nd_to_sample), \
                                                indices)
                else:
                    fake_validity = discriminator(gen_mks, given_nds, given_eds, nd_to_sample)
                    #[32, 1]
                if is_mean :
                    l1_loss = torch.nn.L1Loss(reduction='mean')
                else:
                    smooth_l1 = torch.nn.SmoothL1Loss(reduction='sum')
                    l1_loss = torch.nn.L1Loss(reduction='sum')
                smooth_l1_mean = torch.nn.SmoothL1Loss(reduction='mean')
########################avg loss#########
                avg_loss = compute_avg_loss(gen_mks.clone(),smooth_l1)
                area_dict = compute_area_norm_penalty_v2(real_mks.data,gen_mks.clone(),given_nds,nd_to_sample,smooth_l1_mean)
                all_areas_loss = sum(area_dict.values())  
                g_loss = -torch.mean(fake_validity) + avg_loss + all_areas_loss
                #np.save('./data_debug.npy',[gen_mks,mks, nds, eds, nd_to_sample, ed_to_sample])

                if epoch > extra_loss_lim:
###########################iou loss################
                    #pos:
                    common_pen = compute_common_loss(real_mks.data,gen_mks.clone(),given_eds,nd_to_sample,ed_to_sample,criterion=smooth_l1)
                    #neg:
#################################
#########area#####################
                    ##sp = compute_sparsity_penalty(gen_mks,given_eds,nd_to_sample,smooth_l1)
                    sp = compute_sparsity_penalty_v3(gen_mks.clone(),nd_to_sample,smooth_l1)##会修改gen_masks       
##############################
                    # Update generator
                    sp_k = 4
                    cp_k = 1;
                    if not is_mean:
                        sp_k = 1;
                    g_loss = g_loss   + sp_k * sp  + cp_k * common_pen
                    ##+ common_pen + 7*all_areas_loss
                    
                    ## area_loss_dict = {}
                    ## for k,v in area_dict.items():
                    ##     area_loss_dict[k] = float(v.data)
                #+ pos_ci_norm + neg_giou_norm
                ###debug
                if torch.isinf(g_loss) :
                    print("bug data saving")
                    visualizeBatch(real_mks,gen_mks, given_nds, given_eds, nd_to_sample,ed_to_sample)
                    print("bug data done")
                g_loss.backward()
                # for name, parms in generator.named_parameters():	
                #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                #     ' -->grad_value:',parms.grad.shape)

                optimizer_G.step()

                if epoch > extra_loss_lim:
                    print("[time:%s]\t[Epoch:%d/%d]\t[Batch:%d/%d]\t[Batch_done:%d]\t[D_loss: %f]\t[G_loss: %f]\t[avg:%s]\t[div:%f]\t[area_loss:%f]\t[area_is_grad:%s]\t[area_detail:%s]\t[cp:%s]\t[sp:%s]"#\t[pos_ci_loss:%f]\t[ci_grad:%s]\t[neg_giou_loss:%f]\t[neg_giou_grad:%s]\t[pos_giou_loss:%f]\t[all_giou_loss:%f] "
                            % (str(datetime.now()),epoch, opt.n_epochs, b_idx, len(fp_loader),batches_done, \
                                d_loss.item(), g_loss.item(),avg_loss.item(),div_loss\
                                #lambda_gp * gradient_penalty\
                                 ,float(all_areas_loss.data),str(all_areas_loss.grad_fn),str(area_dict)
                                    ,str(sp_k * sp )\
                                    ,str(common_pen)\
                                    ))
                else:
                    print("[time:%s]\t[Epoch:%d/%d]\t[Batch:%d/%d]\t[Batch_done:%d]\t[D_loss: %f]\t[G_loss: %f]\t[avg:%s]\t[div:%f]\t[area_loss:%f]\t[area_is_grad:%s]\t[area_detail:%s]"#\t[sp:%s]\t[pos_ci_loss:%f]\t[ci_grad:%s]\t[neg_giou_loss:%f]\t[neg_giou_grad:%s]\t[pos_giou_loss:%f]\t[all_giou_loss:%f] "
                            % (str(datetime.now()),epoch, opt.n_epochs, b_idx, len(fp_loader),batches_done, \
                                d_loss.item(), g_loss.item(),avg_loss.item(),div_loss\
                                ,all_areas_loss.item(),str(all_areas_loss.grad_fn),str(area_dict)
                                #lambda_gp * gradient_penalty\
                                    #,str(sp)
                                    ))               
                #print("batches_done: %s samepe_interval: %s eq_val: %s" % (batches_done,opt.sample_interval,(batches_done % opt.sample_interval == 0) and batches_done))
                if (batches_done % opt.sample_interval == 0) and batches_done:
                    torch.save(generator.state_dict(), './checkpoints/{}_{}.pth'.format(exp_folder, batches_done))
                    print("checkpoints save done")
                    visualizeBatch(real_mks,gen_mks, given_nds, given_eds, nd_to_sample,ed_to_sample)
                    print("training data save done")
                    visualizeSingleBatch(fp_loader_test, opt)
                    #print("images save done [valid iou:%f giou:giou_norm]" % (iou_norm,giou_norm))

                batches_done += opt.n_critic
                
