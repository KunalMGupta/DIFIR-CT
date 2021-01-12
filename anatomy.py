import numpy as np
from config import Config
import warnings

class Motion:
    def __init__(self, config):
        
        assert isinstance(config, Config), 'config must be an instance of class Config'
        
        self.config = config
        self.func_map = {'const':self.const, 
                        'simple_sin': self.simple_sin,
                        'coupled_sin':self.coupled_sin,
                        'pseudo_heart':self.pseudo_heart,
                        'figure_eight':self.figure_eight,
                        'rectangle':self.rectangle,
                        'const2':self.const2, 
                        'simple_sin2': self.simple_sin2}
        
    def get_motion(self,size, location):
        assert isinstance(size, str) and isinstance(location, str), 'size and location are string parameters'
        assert size in list(self.func_map.keys()) and location in list(self.func_map.keys()), 'Wrong key words for size:{} and location:{}'.format(size, location)
        
        return self.func_map[size], self.func_map[location]
    
    def const(self, val, t):
        '''
        Constant Value
        '''
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        return val
    
    def simple_sin(self, val, t, A=.4, bias = 1.0):
        '''
        simple sinusoid 
        '''
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(A, float), 'A must be a float'
        assert isinstance(bias, float), 'bias must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        
        result = (bias + A*np.sin(2*np.pi*t))*val
        
        if result > 3*val:
            warnings.warn('Resultant value of motion func is more than 3 times the val!!!')
            
        return result
    
    def coupled_sin(self, val, t, k=2.0, A=0.1, bias =1.0):
        '''
        A sin and a cosine of different frequency
        '''
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(A, float), 'A must be a float'
        assert isinstance(bias, float), 'bias must be a float'
        assert isinstance(k, float), 'k must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        
        result = (bias + A*(np.sin(2*np.pi*t) + \
                          np.sin(2*np.pi*t*k))/1.414)*val
        
        if result > 3*val:
            warnings.warn('Resultant value of motion func is more than 3 times the val!!!')
            
        return result
                        
    def pseudo_heart(self, val, t):
        '''
        Model of a heart beat
        '''
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        
        T = np.array([0,0.05,0.1,0.4 ,0.9,1])       # Time axis
        a = np.array([0, 0  ,0.2,-0.4, 0 ,0])       # Corresponding aplitudes
        
        A = np.array([[T[2], T[2]**2, T[2]**3],
                      [T[3], T[3]**2, T[3]**3],     # Matrix for polynomial function computation
                      [0   , 2, 3*(T[2]+T[3])]])
        
        B = np.array([a[2],a[3],0]).T
        
        if t >= T[0] and t <= T[1]:
            # Phase 1: No contraction
            result = val
        
        elif t >= T[1] and t <= T[2]:
            # Pahse 2: Sinusoidal expansion
            t_ = 2*(t-T[1])/(T[2]-T[1]) - 1
            result = val*((np.sin(0.5*np.pi*t_)+1)*a[2]/2+1)

        elif t >= T[2] and t <= T[3]:
            # Phase 3: Polynomial contraction
            Y = np.linalg.inv(A)@B
            y = Y[0]*t + Y[1]*t**2 + Y[2]*t**3
            result = val*(y+1)
                        
        elif t >= T[3] and t <= T[5]:
            # Phase 4: Exponential expansion
            k = 5/(T[4] - T[3]) ** 2
            y = np.exp(-k*(t-T[3])**2)*a[3]
            result = val*(y+1)
        else:
            raise('Something is wrong with pseudo heart')
            
        if result > 3*val:
            warnings.warn('Resultant value of motion func is more than 3 times the val!!!')
            
        return result
            
    def figure_eight(self, val, t):   
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        
        x = np.sin(2*np.pi*t)
        y = np.sin(2*np.pi*t)*np.cos(2*np.pi*t)
        
        return val*x,val*y
    
    def rectangle(self, val, t, a=2.0, b=1.0):
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        assert isinstance(a, float) and a>0, 'a must be a positive float'
        assert isinstance(b, float) and b>0, 'b must be a positive float'
        
        x = np.sign(np.sin(2*np.pi*t))*a*(np.abs(np.sin(2*np.pi*t)))**(1/a)
        y = np.sign(np.cos(2*np.pi*t))*b*(np.abs(np.cos(2*np.pi*t)))**(1/b)
        return val*x,val*y
        
    def const2(self, val, t):
        '''
        Constant Value
        '''
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        return 0, 0
    
    def simple_sin2(self, val, t, A=0.2, bias = 0.0):
        '''
        simple sinusoid 
        '''
        assert isinstance(val, float), 'val must be a float'
        assert isinstance(t, float) and t >= 0 and t <= 1, 't: {} must be a float between 0 and 1'.format(t)
        assert isinstance(A, float), 'A must be a float'
        assert isinstance(bias, float), 'bias must be a float'
        
        return val*(bias + A*np.sin(2*np.pi*t)), 0
#     (bias + A*np.sin(2*np.pi*t*self.config.NUM_HEART_BEATS))
      
class Organ:
    '''
    Main Organ object
    '''
    def __init__(self, config, center, a, b, size, location):
        '''
        Each organ is an ellipse that can change its location and size as a function of time
        '''
        assert isinstance(config, Config), 'config must be an instance of class Config'
        assert isinstance(center, list) and len(center)==2 and all([isinstance(c,float) for c in center]) ,'center must be a list of 2 float coordinates'
        assert isinstance(a, float) and isinstance(b, float) and a>0 and b >0, 'radii a and b should be positive float'
        assert isinstance(size, str) and isinstance(location, str), 'size and location are string parameters'
        
        self.center = center                # Location of center
        self.a = a                          # Major axis 
        self.b = b                          # Minor axis 
        
        self.func_r, self.func_c = Motion(config).get_motion(size, location)
        self.config = config                # Config data
         
    def ellipse(self, pt, a, b):
        assert isinstance(pt, np.ndarray) and len(pt.shape) == 2, 'pt must be a 2D numpy array'
        assert isinstance(a, float) and isinstance(b, float) and a>0 and b >0, 'radii a and b should be positive float'
        
        result = (((pt[:,0]-self.center[0])/a)**2 + ((pt[:,1]-self.center[1])/b)**2 - 1 < 0)*1
        
        assert (((result == 0) + (result == 1)) == True).all(), 'Not all values are either 0 or 1'
        assert len(result.shape) == 2, 'result has incorrect shape of {}'.format(result.shape)
        
        return result
            
    def get_phase(self, t):
        '''
        Calculates the correct phase of the organ motion given time of gantry
        (-TOTAL_CLICKS, +TOTAL_CLICKS)  --> (0,1)
        
        '''
        assert t >= -self.config.TOTAL_CLICKS and t <= self.config.TOTAL_CLICKS, 't = {} is out of range'.format(t)
        
        t = self.config.GANTRY2HEART_SCALE*t
        
        if t <= 0:
            t +=1
            
        assert t >=0 and t <= 1, 'Resultant t = {} is out of range (0,1)'.format(t)
        
        return t
         
    def is_inside(self, pt, t):
        '''
        Checks if a point pt is inside the ellipse at time t
        
        pt: Nx2 numpy array
        t: an integer 
        '''
        assert isinstance(pt, np.ndarray) and len(pt.shape) == 2, 'pt must be a 2D numpy array'
        assert isinstance(t, int) and abs(t) <= self.config.TOTAL_CLICKS, 'Time is out of range: {}'.format(t)
        
        t = self.get_phase(t)
        delta_cx, delta_cy = self.func_c(t)
        
        pt_new = pt.copy()
        pt_new[:,0] -= delta_cx
        pt_new[:,1] -= delta_cy
        
        return self.ellipse(pt_new, self.func_r(self.a, t), self.func_r(self.b, t))
            
          
class Body:
    '''
    Body class is collection of several Organ objects
    '''
    def __init__(self, organs):
        
        assert isinstance(organs, list), 'Provide a valid list of organs'
        assert all([isinstance(organ, Organ) for organ in organs]), 'List elements must be of type organ'
        
        self.organs = organs
        
       
    def is_inside(self, pt, t):
        '''
        returns a one hot encoding of point by querying each organ. returns all zeros if the point is outside the body
        '''
        
        assert isinstance(pt, np.ndarray) and len(pt.shape) == 2, 'Points must be 2D numpy array'
        assert isinstance(t, float), 't must be a single float value'
        
        inside = np.zeros((pt.shape[0], len(self.organs)))
        
        for idx, orgon in enumerate(self.organs):
             inside[:,idx] = orgon.is_inside(pt,t)
                
                
        assert (np.sum(inside, axis=1) < 2).all(), 'Organs are intersecting !! '
        return inside