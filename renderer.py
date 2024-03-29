import numpy as np
import torch
import torch.nn as nn
import kornia
from skimage.transform import iradon
from scipy import ndimage
from scipy.ndimage import rotate



from config import Config
from anatomy import Body

def sdf_to_occ(x):
    '''
    Converts sign distance to occupancy for rendering
    '''
    assert isinstance(x, torch.Tensor) and len(x.shape) == 3, 'Input must be a 3D torch tensor'
    occ = torch.zeros(x.shape)
    for i in range(x.shape[2]):
        occ[...,i] = torch.clamp(50*(torch.sigmoid(x[...,i]) - 0.5),0,1)  # (0,1) -> (-0.5,0.5) -> (-10,10) -> (0,1)
        
    return occ

def occ_to_sdf(x):
    '''
    This function convets a binary occupancy image into a signed distance image.
    '''
    assert isinstance(x, np.ndarray) and len(x.shape) == 3, 'x must be a 3D array containing separate images for each organ'
    assert np.sum(x==1) + np.sum(x==0) == x.shape[0]*x.shape[1]*x.shape[2], 'x must only have values 0 and 1' 
    
    dist_img = np.zeros_like(x)

    for i in range(x.shape[2]):
        dist_img[...,i] = ndimage.distance_transform_bf(x[...,i]==1) - ndimage.distance_transform_bf(x[...,i]==0)

    return dist_img


class SDF(nn.Module):
    def __init__(self):
        super(SDF, self).__init__()
    
    def forward(self):
        raise NotImplementedError
        pass

class SDFGt(SDF):
    def __init__(self, config, body):
        super(SDFGt, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(body, Body), 'body must be an instance of class Body'
        assert config.INTENSITIES.shape[1] == len(body.organs), 'Number of organs must be equal to the number of intensities'
        
        self.config = config
        self.body = body
        x,y = np.meshgrid(np.linspace(0,1,config.IMAGE_RESOLUTION),np.linspace(0,1,config.IMAGE_RESOLUTION))
        self.pts = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
      
    def forward(self, t):
        '''
        Calculates the ground truth SDF or Image for given time of gantry
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        inside = self.body.is_inside(self.pts, t)
        inside_sdf = occ_to_sdf(inside.reshape(self.config.IMAGE_RESOLUTION,
                                               self.config.IMAGE_RESOLUTION,self.config.INTENSITIES.shape[1]))
        
        canvas = torch.from_numpy(inside_sdf)
        assert len(canvas.shape) == 3, 'Canvas must be a 3D tensor, instead is of shape: {}'.format(canvas.shape)
        return canvas

class Intensities(nn.Module):
    def __init__(self, config, learnable = False, init = np.array([0.3,0.5]), bandwidth = 0.05):
        super(Intensities, self).__init__()
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(learnable, bool), 'learnable must be a boolean'
        assert isinstance(init, np.ndarray) and len(init.shape) == 1, 'init must be a 1D array of intensities'
        
        assert isinstance(bandwidth, float) and bandwidth > 0 and bandwidth < 1, 'bandwidth must be a float between 0 and 1'
        
        if learnable:
            assert config.NUM_SDFS == init.shape[0], 'init must exactly the same intensities as number of sdfs'
            self.inty = torch.nn.Parameter(torch.from_numpy(init).view(1,1,-1)) 
            self.default = 0*torch.from_numpy(config.INTENSITIES).view(1,1,-1)
        else:
            self.inty = 0*torch.from_numpy(config.INTENSITIES).view(1,1,-1)
            self.default = torch.from_numpy(config.INTENSITIES).view(1,1,-1)
            
        self.config = config
        self.bandwidth = bandwidth
        
    def forward(self):    
        residual = torch.clamp(self.inty, -1, 1)*self.bandwidth
        return self.default + residual
    
class Renderer(nn.Module):
    def __init__(self, config, sdf, intensities, offset=0.0):
        super(Renderer, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(sdf, SDF), 'sdf must be an instance of class SDF'
        assert isinstance(intensities, Intensities), 'intensities must be an instance of class Intensities'
        
        self.config = config
        self.sdf = sdf
        self.intensities = intensities
        self.offset = offset    
    def snapshot(self,t):
        '''
        Rotates the canvas at a particular angle and calculates the intensity
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        rotM = kornia.get_rotation_matrix2d(torch.Tensor([[self.config.IMAGE_RESOLUTION/2,self.config.IMAGE_RESOLUTION/2]]), torch.Tensor([t+self.offset]) , torch.ones(1)).cuda()
        
        canvas = sdf_to_occ(self.sdf(t))
        
        intensities = self.intensities().type_as(canvas)
        assert len(intensities.shape) == 3, 'intensities must be a 3D tensor'
        
        canvas = canvas*intensities
        assert canvas.shape == (self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION, self.config.NUM_SDFS)
        
        canvas = torch.sum(canvas, dim=2)
#         canvas = torch.sum(canvas*self.intensities().type_as(canvas),dim=2)
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
    
    