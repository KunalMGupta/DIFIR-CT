import numpy as np
import torch
import torch.nn as nn
import kornia
from config import Config
from anatomy import Body

def sdf_to_occ(x):
    '''
    Converts sign distance to occupancy for rendering
    '''
    return torch.clamp(20*(torch.sigmoid(x) - 0.5),0,1)  # (0,1) -> (-0.5,0.5) -> (-10,10) -> (0,1)

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
        
    def forward(self, t, combine=False):
        '''
        Calculates the ground truth SDF or Image for given time of gantry
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        assert isinstance(combine, bool), 'combine must be a boolean'
        
        if combine:
            canvas = torch.from_numpy(self.body.is_inside(self.pts, t)@self.config.INTENSITIES.T).\
            view(1,1,self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION).cuda().float()
        else:
            canvas = torch.from_numpy(self.body.is_inside(self.pts, t)).view(1,1,self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION,self.config.INTENSITIES.shape[1]).cuda().float()
            
        return canvas
    
    
class Renderer(nn.Module):
    def __init__(self, config, sdf):
        super(Renderer, self).__init__()
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(sdf, SDF), 'sdf must be an instance of class SDF'
        
        self.config = config
        self.sdf = sdf
        
    def snapshot(self,t):
        '''
        Rotates the canvas at a particular angle and calculates the intensity
        '''
        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        
        rotM = kornia.get_rotation_matrix2d(torch.Tensor([[self.config.IMAGE_RESOLUTION/2,self.config.IMAGE_RESOLUTION/2]]), torch.Tensor([t*360/self.config.GANTRY_VIEWS_PER_ROTATION]) , torch.ones(1)).cuda()
        
        canvas = kornia.warp_affine(self.sdf(t, combine=True), rotM, dsize=(self.config.IMAGE_RESOLUTION, self.config.IMAGE_RESOLUTION))
        
        canvas = sdf_to_occ(canvas)
        
        result = torch.sum(canvas.view(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION), axis=1)/self.config.IMAGE_RESOLUTION
        
        assert len(result.shape) ==1, 'Canvas should be a 1D array'
        return result
        
    def forward(self, all_thetas):
        
        assert isinstance(all_thetas, np.ndarray) and len(all_thetas.shape) ==1 and all_thetas.dtype == float, 'all_thetas must be a 1D numpy array of integers'
        
        self.intensity = torch.zeros((self.config.IMAGE_RESOLUTION, all_thetas.shape[0])).cuda()
        for i, theta in enumerate(all_thetas):
            self.intensity[:,i] = self.snapshot(theta)
            
        return self.intensity
        
        
        
        
        
        