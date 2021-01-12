import unittest
import numpy as np

from config import Config
from anatomy import Motion, Organ, Body
        
class TestNCT(unittest.TestCase):
    
    def test_config_inputs(self):

        self.assertRaises(TypeError,Config,)
        
        # Test for Intensity
        for intensity in [0.2,[0.2],[[0.2]],[[0.3,0.7]],np.array(0.3),np.array([0.3]), np.array([[[0.3]]]),'High']:
            self.assertRaises(AssertionError,Config,intensity, TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        
        # Test for TYPE
        for type in [-1, 5, 1.0, 'a', [0], [0,1]]:
            self.assertRaises(AssertionError,Config, np.array([[0.3,0.6]]), TYPE=type, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
       
        # Test for NUM_HEART_BEATS
        for num in [-1,2,[2,4],[2,4.0],10.0]:
            self.assertRaises(AssertionError,Config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=num, NUM_SDFS=2)
        
        # Test for NUM_SDFS
        for num in [-1,0,'one',[0,3],0.6,2.0]:
            self.assertRaises(AssertionError,Config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=num)
        
    def test_motion_init(self):
        
        # Test inputs to Motion
        for input in ['a',0.2,[1.0,2.0],np.array([[1,2]]),None]:
            self.assertRaises(AssertionError,Motion,input)
        self.assertRaises(TypeError,Motion,)
        
    def test_motion_get_motion(self):
        
        config = Config(np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        motion = Motion(config)
        
        for input in [1,'a','abc',None,3.0]:
            self.assertRaises(AssertionError, motion.get_motion, input, 'const')
            
        for input in [1,'a','abc',None,3.0]:
            self.assertRaises(AssertionError, motion.get_motion, 'const', input)
            
    def test_motion_models(self):
        
        # Test t input for different motion models
        config = Config(np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        motion = Motion(config)
        
        wrong_time_inputs = [-2.0, -1, -0.1, [0.1,0,5], None, '@', 'abc', 0, 1, 1.1, 10]
        
        print('\n Testing for motion models: ')
        for func in ['const','simple_sin','coupled_sin','pseudo_heart']:
            
            model1, model2 = motion.get_motion(func, func)
            # Test that value at t=0 and t=1 are 0
            print(func)
            self.assertAlmostEqual(model1(1.0,0.0), 1, 2)
            self.assertAlmostEqual(model1(1.0,1.0), 1, 2)
            self.assertAlmostEqual(model2(1.0,0.0), 1, 2)
            self.assertAlmostEqual(model2(1.0,1.0), 1, 2)
            
            for t in wrong_time_inputs:
                self.assertRaises(AssertionError, model1, 1.0, t)
                self.assertRaises(AssertionError, model2, 1.0, t)
                
        for func in ['figure_eight','rectangle','const2','simple_sin2']:
            
            model1, model2 = motion.get_motion(func, func)
            # Test that value at t=0 and t=1 are 0
            print(func)
            self.assertAlmostEqual(model1(1.0,0.0)[0], 0, 2)
            self.assertAlmostEqual(model1(1.0,1.0)[0], 0, 2)
            self.assertAlmostEqual(model2(1.0,0.0)[0], 0, 2)
            self.assertAlmostEqual(model2(1.0,1.0)[0], 0, 2)
            
#             self.assertAlmostEqual(model1(1.0,0.0)[1], 0, 2)
#             self.assertAlmostEqual(model1(1.0,1.0)[1], 0, 2)
#             self.assertAlmostEqual(model2(1.0,0.0)[1], 0, 2)
#             self.assertAlmostEqual(model2(1.0,1.0)[1], 0, 2)
            
            for t in wrong_time_inputs:
                self.assertRaises(AssertionError, model1, 1.0, t)
                self.assertRaises(AssertionError, model2, 1.0, t)
                
        
if __name__ == '__main__':
    unittest.main()