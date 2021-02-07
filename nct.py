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

from config import *
from anatomy import *
from renderer import *
from siren import *
from tqdm import tqdm

torch.cuda.set_device(0)

config = Config(np.array([[0.3, 0.8]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
body = Body(config, [Organ(config,[0.6,0.6],0.2,0.2,'pseudo_heart','const2'),
                     Organ(config,[0.3,0.3],0.1,0.1,'simple_sin','const2')])


def fetch_fbp_movie(config, body):
    
    
    config = deepcopy(config)
    body = deepcopy(body)
    incr = 180
    
    config.THETA_MAX = config.THETA_MAX+2*incr
    body.config.THETA_MAX = config.THETA_MAX 
    sdf = SDFGt(config, body)
    THETA_MAX = config.THETA_MAX+incr
    
    with torch.no_grad():
        from skimage.transform import iradon
        all_thetas = np.linspace(-incr, config.THETA_MAX, config.TOTAL_CLICKS + 2*incr*2)
        gtrenderer = Renderer(config,sdf)
        sinogt = gtrenderer(all_thetas).detach().cpu().numpy()
        
    sinogram = sinogt.reshape(config.IMAGE_RESOLUTION, config.TOTAL_CLICKS+ 2*incr*int(config.GANTRY_VIEWS_PER_ROTATION/360))
    reconstruction_fbp = np.zeros((config.IMAGE_RESOLUTION,config.IMAGE_RESOLUTION,config.TOTAL_CLICKS))
    count=0
    for i in tqdm(range(0, config.TOTAL_CLICKS)):
        reconstruction_fbp[...,count] = iradon(sinogram[...,i:i+2*incr], theta = all_thetas[i:i+2*incr],circle=True).T
        count+=1
    
    return sinogram, 132*reconstruction_fbp

sinogram, reconstruction_fbp = fetch_fbp_movie(config, body)

def find_background_channel(image):
    
    assert isinstance(image, np.ndarray) and len(image.shape) == 3
    
    total = []
    for i in range(image.shape[2]):
        total.append(np.sum(image[...,i]))
        
    return total.index(max(total))

def get_connect_componentes_per_frame(labels, num_components=2):
    
    num_components +=1
    component_image = np.zeros((labels.shape[0],labels.shape[1],num_components))
    
    for idx in range(num_components):
        
        image = (labels==idx).astype('uint8')
        
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
        
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        component_image[...,idx] = (output == max_label)
        
    background_index = find_background_channel(component_image)
    
    component_image = np.delete(component_image,background_index, axis=2).copy()
    
    return component_image

def get_n_objects(img, num_components=2):
    
    assert isinstance(img, np.ndarray) and len(img.shape) == 3, 'img must be a 3D numpy array'
    assert isinstance(num_components, int) and num_components >= 2, 'num_components must be a integer more than or equal to 2'

    num_components+=3
    proceed = True
    count = 0
    while proceed and count<3:
        mask = np.random.randint(0,img.shape[2],int(0.02*img.shape[2]))
        X = img[...,mask].reshape(-1,1)
        gm = GaussianMixture(n_components=5, random_state=0).fit(X)
        labels_to_use = np.where(gm.means_[:,0]>0.1)[0]
        
        if labels_to_use.shape[0] == num_components-3:
            proceed = False
        else:
            print('Segmentation failed, found labels : {}. Tried {} time'.format(labels_to_use,count+1))
            count+=1
            
    labels = np.zeros((img.shape))
    
    for i in range(img.shape[2]):
        label_image = np.zeros((img.shape[0],img.shape[1]))
        count=1
        for k in labels_to_use:
            label_image += (gm.predict(reconstruction_fbp[...,i].reshape(-1,1)).reshape(img.shape[0],img.shape[1]) == k)*count
            count+=1
        labels[...,i] = label_image
        
    return labels.reshape(img.shape)

def get_n_objects_for_movie(fbp_movie, num_components=2):
    
    print("Computing Segmentations...")
    movie = get_n_objects(fbp_movie.copy(),num_components=num_components)
    movie_objects = np.zeros((movie.shape[0],movie.shape[1],movie.shape[2],num_components))
    print("Computing connected components...")
    for i in tqdm(range(movie.shape[2])):
        movie_objects[...,i,:] = get_connect_componentes_per_frame(movie[...,i],num_components=num_components)

    print("Computing SDFs...")
    init = np.zeros((1,num_components))
    for j in tqdm(range(num_components)):
        for i in range(movie.shape[2]):
#             movie_objects[...,i,j] = denoise_tv_chambolle(occ_to_sdf(movie_objects[...,i,j][...,np.newaxis]))[...,0]
            occupancy = np.round(denoise_tv_chambolle(movie_objects[...,i,j][...,np.newaxis]))
            movie_objects[...,i,j] = occ_to_sdf(occupancy)[...,0]
            
        img = fbp_movie[...,0]
        test = np.where(occupancy>0.5)#[...,0]
#         print(img.shape, test[0].shape)
        init[0,j] = np.median(img[test[0], test[1]])
        
    return movie_objects, init

def get_pretraining_sdfs(config, sdf=None):
    if sdf is None:
        pretraining_sdfs = np.zeros((config.IMAGE_RESOLUTION,config.IMAGE_RESOLUTION,config.TOTAL_CLICKS,config.NUM_SDFS))
        for i in range(config.NUM_SDFS):
            cfg = Config(np.array([[np.random.rand()]]), config.TYPE, config.NUM_HEART_BEATS, 1)
            if i ==0:
                organ = Organ(cfg, [0.6,0.6], 0.1, 0.1, 'simple_sin', 'simple_sin2')
            else:
                organ = Organ(cfg, [0.3,0.3], 0.1, 0.1, 'simple_sin', 'simple_sin2')
#             organ = Organ(cfg, [np.random.rand()*0.7+0.1,
#                                 np.random.rand()*0.7+0.1], 0.1, 0.1, 'simple_sin', 'simple_sin2')
            body = Body(cfg,[organ])
            sdf = SDFGt(cfg, body)
            all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)
            for j in range(config.TOTAL_CLICKS):
                pretraining_sdfs[...,j,i] = denoise_tv_chambolle(sdf(all_thetas[j])[...,0].detach().cpu().numpy())
    
        init = config.INTENSITIES
    elif isinstance(sdf, np.ndarray):
        pretraining_sdfs, init = get_n_objects_for_movie(sdf,num_components=config.NUM_SDFS)
        
    else:
        pretraining_sdfs = np.zeros((config.IMAGE_RESOLUTION,config.IMAGE_RESOLUTION,config.TOTAL_CLICKS,config.NUM_SDFS))
        for i in range(config.NUM_SDFS):
            all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)
            for j in range(config.TOTAL_CLICKS):
                pretraining_sdfs[...,j,i] = occ_to_sdf(np.round(denoise_tv_chambolle(sdf_to_occ(sdf(all_thetas[j]))[...,i].detach().cpu().numpy(),weight=5))[...,np.newaxis])[...,0]
        
        init = None
    return pretraining_sdfs, init

pretraining_sdfs, _ = get_pretraining_sdfs(config)
print(np.mean(np.sqrt(np.gradient(pretraining_sdfs,axis=0)**2 + np.gradient(pretraining_sdfs,axis=1)**2)))
print(np.mean(np.abs(np.gradient(pretraining_sdfs,axis=2))))

class FourierFeatures(nn.Module):
    '''
    Learning a function as a fourier series
    Refer: https://colab.research.google.com/github/ndahlquist/pytorch-fourier-feature-networks/blob/master/demo.ipynb#scrollTo=QDs4Im9WTQoy
    '''
    
    def __init__(self, input_channels, output_channels, mapping_size = 128, scale=2.0, testing=False):
        super(FourierFeatures, self).__init__()
        
        assert isinstance(input_channels, int), 'input_channels must be an integer'
        assert isinstance(output_channels, int), 'output_channels must be an integer'
        assert isinstance(mapping_size, int), 'maping_size must be an integer'
        assert isinstance(scale, float), 'scale must be an float'
        assert isinstance(testing, bool), 'testing should be a bool'
        
        self.mapping_size = mapping_size
        self.output_channels = output_channels
        self.testing = testing
        
        if self.testing:
            self.B = torch.ones((1, self.mapping_size, self.output_channels))
        else:
            self.B = torch.randn((1, self.mapping_size, self.output_channels))*scale
            
        self.B = self.B.cuda()
        self.net = Siren(input_channels,128,3,(2*self.mapping_size+1)*self.output_channels)
        
    def forward(self, x, t):
        
        assert isinstance(x, torch.Tensor) and len(x.shape) == 2, 'x must be a 2D tensor'
        assert isinstance(t, torch.Tensor) or isinstance(t, float) and t>=-1 and t <=1, 't must be a float between -1 and 1'

        if self.testing:
            fourier_coeffs = torch.ones((x.shape[0],self.mapping_size*2+1, self.output_channels)).type_as(x)
        else:
            fourier_coeffs = self.net(x).view(-1, self.mapping_size*2+1, self.output_channels)
            
        fourier_coeffs_dc = fourier_coeffs[:,-1:,:]
        fourier_coeffs_ac = fourier_coeffs[:,:-1,:]
        
        assert fourier_coeffs_dc.shape == (x.shape[0], 1, self.output_channels), 'Inavild size for fourier_coeffs_dc : {}'.format(fourier_coeffs_dc.shape)
        assert fourier_coeffs_ac.shape == (x.shape[0], self.mapping_size*2, self.output_channels),  'Inavild size for fourier_coeffs_ac : {}'.format(fourier_coeffs_ac.shape)

        t = (2*np.pi*t*self.B).repeat(x.shape[0],1,1)
        
        tsins = torch.cat([torch.sin(t), torch.cos(t)], dim=1).type_as(x)

        assert tsins.shape == (x.shape[0],2*self.mapping_size,self.output_channels)
        series = torch.mul(fourier_coeffs_ac, tsins)
        assert series.shape ==  (x.shape[0],2*self.mapping_size,self.output_channels)
        val_t = torch.mean(series, dim=1, keepdim=True)
        assert val_t.shape == (x.shape[0],1,self.output_channels)
        val_t = val_t + fourier_coeffs_dc
        assert val_t.shape == (x.shape[0],1,self.output_channels)
        
        return val_t.squeeze(1)
    
ff =  FourierFeatures(2, 2, testing=True).cuda()
x = torch.Tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]).cuda()
t = 1.0
val_t = ff(x,t)
print(torch.norm(val_t - 1.5*torch.ones(x.shape).type_as(x)))

class SDFNCT(SDF):
    def __init__(self, config):
        super(SDFNCT, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        
        self.config = config
        x,y = np.meshgrid(np.linspace(0,1,self.config.IMAGE_RESOLUTION),np.linspace(0,1,self.config.IMAGE_RESOLUTION))
        self.pts = torch.autograd.Variable(2*(torch.from_numpy(np.hstack((x.reshape(-1,1),y.reshape(-1,1)))).cuda().float()-0.5),requires_grad=True)
        
        self.encoder = Siren(2,256,3,config.NUM_SDFS).cuda()
        self.velocity = FourierFeatures(2,config.NUM_SDFS).cuda()
        
    def compute_sdf_t(self, t):
        assert isinstance(t, torch.Tensor) or isinstance(t, float), 't = {} must be a float or a tensor here'.format(t)
        assert t >= -1 and t <= 1, 't = {} is out of range'.format(t)
        
        displacement = self.velocity(self.pts, t)
        init_sdf = self.encoder(self.pts)
        assert init_sdf.shape == displacement.shape
        
        canvas = (init_sdf + displacement)*self.config.SDF_SCALING
        if not (torch.min(canvas) < -1 and torch.max(canvas) > 1):
            warnings.warn('SDF values are in a narrow range between (-1,1)')
            
        canvas = canvas.view(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION,self.config.NUM_SDFS)
        
        return canvas
            
    def forward(self, t):
        
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        t = 2*get_phase(self.config,t) - 1
        
        canvas = self.compute_sdf_t(t)        
        assert len(canvas.shape) == 3, 'Canvas must be a 3D tensor, instead is of shape: {}'.format(canvas.shape)
        
        return canvas
    
    def grad(self, t):
        
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        t = torch.autograd.Variable(torch.Tensor([2*get_phase(self.config,t) - 1]).cuda().float(),requires_grad=True)
        
        canvas = self.compute_sdf_t(t)/self.config.SDF_SCALING
        
        dc_dxy = gradient(canvas, self.pts)
        assert len(dc_dxy.shape) == 2, 'Must be a 2D tensor, instead is {}'.format(dc_dxy.shape)

        occupancy = sdf_to_occ(canvas)
        assert len(occupancy.shape) == 3
        
        do_dxy = gradient(occupancy, self.pts)
        assert len(do_dxy.shape) == 2, 'Must be a 2D tensor, instead is {}'.format(do_dxy.shape)
        
        dc_dt = gradient(canvas, t)/(np.prod(canvas.shape))
        assert len(dc_dt.shape) == 1, 'Must be a 1D tensor, instead is {}'.format(dc_dt.shape)
        
        eikonal = torch.abs(torch.norm(dc_dxy, dim=1) - 1).mean()
        total_variation_space = torch.norm(do_dxy, dim=1).mean()
        total_variation_time = torch.abs(dc_dt)
        
        return eikonal, total_variation_space, total_variation_time
    
def pretrain_sdf(config, sdf = None, lr = 1e-4):
    

    pretraining_sdfs, init = get_pretraining_sdfs(config, sdf)
        
    assert len(pretraining_sdfs.shape) == 4, 'Invalid shape : {}'.format(pretraining_sdfs.shape)
    
    sdf = SDFNCT(config)
    gt = torch.from_numpy(pretraining_sdfs).cuda()
    
    optimizer = optim.Adam(list(sdf.parameters()), lr = lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    
    for itr in range(1000):
        optimizer.zero_grad()
        t = np.random.randint(0,config.TOTAL_CLICKS,1)[0]
        theta = t*(config.THETA_MAX/config.TOTAL_CLICKS)
        
        pred = sdf(theta)
        target = gt[...,t,:]
        assert target.shape == pred.shape, 'target has shape : {} while prediction has shape :{}'.format(target.shape, pred.shape)
        eikonal, _, _ = sdf.grad(theta)

        loss1 = torch.abs(pred - target).mean()
        loss = loss1 + 0.1*eikonal
        loss.backward()
        optimizer.step()
        
        if itr %200 == 0:
            print('itr: {}, loss: {:.4f}, lossP: {:.4f}, lossE: {:.4f}, lr: {:.4f}'.format(itr, loss.item(), loss1.item(), 
                                                                                           eikonal.item(),scheduler.get_last_lr()[0]*10**4))
            scheduler.step()
            
    return sdf, init

# sdf, init = pretrain_sdf(config,reconstruction_fbp)
sdf, init = pretrain_sdf(config)
print(init)

def get_sinogram(config, sdf, intensities):
    renderer = Renderer(config, sdf,intensities)
    all_thetas = np.linspace(0,config.THETA_MAX, config.TOTAL_CLICKS)
    sinogram = renderer.forward(all_thetas).detach().cpu().numpy()
    
    return sinogram

gt_sinogram = torch.from_numpy(get_sinogram(config, SDFGt(config, body),Intensities(config, learnable = False))).cuda()

class Intensities(nn.Module):
    def __init__(self, config, learnable = False, init = np.array([0.25,0.82])):
        super(Intensities, self).__init__()
        assert isinstance(config, Config), 'config must be an instance of class Config'
        
        if learnable:
            self.inty = torch.nn.Parameter(torch.from_numpy(init).view(1,1,-1)) 
        else:
            self.inty = 0*torch.from_numpy(config.INTENSITIES).view(1,1,-1)
        
        self.config = config
        self.default = torch.from_numpy(config.INTENSITIES).view(1,1,-1)
        
    def forward(self):    
        residual = torch.clamp(self.inty, -1, 1)*0.05
        
        return self.default + residual
    
class Renderer(nn.Module):
    def __init__(self, config, sdf, intensities):
        super(Renderer, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(sdf, SDF), 'sdf must be an instance of class SDF'
        
        self.config = config
        self.sdf = sdf
        self.intensities = intensities
            
    def snapshot(self,t):
        '''
        Rotates the canvas at a particular angle and calculates the intensity
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        rotM = kornia.get_rotation_matrix2d(torch.Tensor([[self.config.IMAGE_RESOLUTION/2,self.config.IMAGE_RESOLUTION/2]]), torch.Tensor([t]) , torch.ones(1)).cuda()
        
        canvas = sdf_to_occ(self.sdf(t))
#         canvas = torch.sum(self.intensities(canvas).type_as(canvas),dim=2)
        canvas = torch.sum(canvas*self.intensities().type_as(canvas),dim=2)

        assert canvas.shape == (self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)
        
        canvas = kornia.warp_affine(canvas.unsqueeze(0).unsqueeze(1).cuda(), rotM, dsize=(self.config.IMAGE_RESOLUTION, self.config.IMAGE_RESOLUTION)).view(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)

        result = (torch.sum(canvas, axis=1)/self.config.IMAGE_RESOLUTION)
        
        assert len(result.shape) ==1, 'result has shape :{} instead should be a 1D array'.format(result.shape)
        return result
        
    def forward(self, all_thetas):
        
        assert isinstance(all_thetas, np.ndarray) and len(all_thetas.shape) ==1, 'all_thetas must be a 1D numpy array of integers'
        assert all_thetas.dtype == float, 'all_thetas must be a float, instead is : {}'.format(all_thetas.dtype)
        assert all(abs(t) <= self.config.THETA_MAX for t in all_thetas), 'all_theta is out of range'.format(all_thetas)
        self.intensity = torch.zeros((self.config.IMAGE_RESOLUTION, all_thetas.shape[0])).cuda()
        for i, theta in enumerate(all_thetas):
            self.intensity[:,i] = self.snapshot(theta)
            
        return self.intensity
    
    def compute_rigid_fbp(self, x, all_thetas):
        '''
        Computes the filtered back projection assuming rigid bodies
        '''
        assert isinstance(x, np.ndarray) and len(x.shape) == 2, 'x must be a 2D numpy array'
        assert isinstance(x, np.ndarray) and len(all_thetas.shape) == 1, 'all_thetas must be a 1D numpy array'
        assert all_thetas.shape[0] == x.shape[1], 'number of angles are not equal to the number of sinogram projections!'

        return iradon(x, theta=all_thetas,circle=True)
    
    
    
def train(config, sdf, gt_sinogram, lr=1e-4, init = np.array([0.25,0.82])):
        
    intensities = Intensities(config, learnable = False, init = init)
    renderer = Renderer(config, sdf,intensities)
    optimizer = optim.Adam(list(sdf.parameters()), lr = lr)
#     optimizer2 = optim.Adam(list(intensities.parameters()), lr = 1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    
    for itr in range(2000):
        optimizer.zero_grad()
#         optimizer2.zero_grad()
        t = np.random.randint(0,config.TOTAL_CLICKS,config.BATCH_SIZE)
        theta = t*(config.THETA_MAX/config.TOTAL_CLICKS)
        pred = renderer(theta)
        target = gt_sinogram[:,t]
        loss1 = torch.abs(pred - target).mean()*100
        
        eikonal, total_variation_space, total_variation_time = sdf.grad(theta[0])
        assert target.shape == pred.shape, 'target has shape : {} while prediction has shape :{}'.format(target.shape, pred.shape)
        
        loss = loss1 + 0.5*eikonal + 0.5*total_variation_space + 0.01*total_variation_time
        
        loss.backward()
        optimizer.step()
#         optimizer2.step()
        
        if itr %200 == 0:
            print('itr: {}, loss: {:.4f}, lossP: {:.4f}, lossE: {:.4f}, lossTVs: {:.4f}, lossTVt: {:.4f}, lr: {:.4f}'.format(itr, loss.item(), loss1.item(), eikonal.item(), 
                                     total_variation_space.item(), total_variation_time.item(),scheduler.get_last_lr()[0]*10**4))
            scheduler.step()
            
        if loss1.item() < 0.08:
            break
            
    return sdf, intensities
#         break
        
    
sdf,intensities = train(config, sdf, gt_sinogram, init=init)
# sdf, _ = pretrain_sdf(config, sdf, lr = 1e-5)
# sdf,intensities  = train(config, sdf, gt_sinogram, lr=1e-5, init=init)

print(intensities.inty)

def fetch_movie(config, sdf):
    frames = np.zeros((config.IMAGE_RESOLUTION,config.IMAGE_RESOLUTION,config.TOTAL_CLICKS,config.NUM_SDFS))
    for i in range(config.NUM_SDFS):
        all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)
        for j in range(config.TOTAL_CLICKS):
            frames[...,j,i] = sdf_to_occ(sdf(all_thetas[j]))[...,i].detach().cpu().numpy()
            
    intensities = config.INTENSITIES.reshape(1,1,1,-1)
    movie = np.sum(frames*intensities, axis=3)
    
    return movie       

movie = fetch_movie(config, sdf)

import os
os.system('rm -r movie/ && mkdir movie')
for i in range(movie.shape[2]):
    plt.imsave('movie/{}.png'.format(i), movie[...,i], cmap='gray')
os.system('rm movie.zip && zip -r movie.zip movie/')