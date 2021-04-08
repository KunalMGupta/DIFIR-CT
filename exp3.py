import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import warnings
from torch import optim         
from torch.optim.lr_scheduler import StepLR
from sklearn.mixture import GaussianMixture
from copy import deepcopy
import imageio

from config import *
from anatomy import *
from renderer import *
from siren import *
from model import *
from tqdm import tqdm
import argparse
from scipy.ndimage import rotate

num = 1
torch.manual_seed(num)
import random
random.seed(num)
import numpy as np
np.random.seed(num)

from scipy.ndimage import binary_erosion

parser = argparse.ArgumentParser()
parser.add_argument("--offset", default=0.0, type=float, help = "Gantry offset")
parser.add_argument("--rate", default=1.0, type=float, help = "Heartrate")
parser.add_argument("--tvs", default=0.5, type=float, help = "Coeff. of spatial TV")
parser.add_argument("--tvt", default=0.5, type=float, help = "Coeff. of temporal TV")
parser.add_argument("--fmax", default=1.5, type=float, help = "Fmax")


opt = parser.parse_args()
print('Doing for Fmax {}'.format(opt.fmax))

folder = 'exp3_results/'

if not os.path.exists(folder):
    os.system('mkdir {}'.format(folder))
        
filename = '{}/movie_{}'.format(folder, int(opt.fmax*10**3))

config = Config(np.array([[0.5]]), TYPE=0, NUM_HEART_BEATS=opt.rate)
body = Body(config, [Organ(config,[0.5,0.5],RADIUS,RADIUS,'const','circle')])
sinogram, reconstruction_fbp = fetch_fbp_movie_exp1(config, body, gantry_offset=opt.offset)
all_thetas = np.linspace(-config.THETA_MAX/2, config.THETA_MAX/2, config.TOTAL_CLICKS)
np.save(filename+'_fbp',reconstruction_fbp)

pretraining_sdfs, init = get_pretraining_sdfs(config, sdf=reconstruction_fbp)

sdf, init = pretrain_sdf(config, pretraining_sdfs, init, lr = 1e-4, scale = opt.fmax)
gt_sinogram = torch.from_numpy(get_sinogram(config, SDFGt(config, body),Intensities(config, learnable = False), all_thetas,offset = opt.offset)).cuda()
sdf,intensities = train(config, sdf, gt_sinogram, init=init[:,0], gantry_offset = opt.offset, coefftvs = opt.tvs, coefftvt=opt.tvt)

pretraining_sdfs, _ = get_pretraining_sdfs(config, sdf=sdf)
pretraining_sdfs = rotate(pretraining_sdfs, -90, reshape=False)
sdf, _ = pretrain_sdf(config, pretraining_sdfs, intensities, lr = 5e-5, scale = opt.fmax)
sdf,intensities = train(config, sdf, gt_sinogram, init=init[:,0], gantry_offset = opt.offset, lr=1e-5, coefftvs = opt.tvs, coefftvt=opt.tvt)

# movie = fetch_movie(config, sdf)
movie = rotate(fetch_movie(config, sdf, None), -90, reshape=False)
np.save(filename+'_nct',movie)

sdfgt = SDFGt(config, body)
movie = fetch_movie(config, sdfgt,all_thetas)
np.save(filename+'_gt',movie)