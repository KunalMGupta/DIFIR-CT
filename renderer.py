import numpy as np
import torch
import torch.nn as nn
import kornia
from config import Config
from anatomy import Body

class SDFGt(nn.Module):
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
        assert isinstance(t, int), 't = {} must be an integer here'.format(t)
        assert t >= -self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(t)
        assert isinstance(combine, bool), 'combine must be a boolean'
        
        if combine:
            canvas = torch.from_numpy(self.body.is_inside(self.pts, t)@self.config.INTENSITIES.T).\
            view(1,1,self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION).cuda().float()
        else:
            canvas = torch.from_numpy(self.body.is_inside(self.pts, t)).view(1,1,self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION,self.config.INTENSITIES.shape[1]).cuda().float()
            
        return canvas