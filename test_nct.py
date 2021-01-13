import unittest
import numpy as np

from config import Config
from anatomy import Motion, Organ, Body
from renderer import SDFGt

import os
        
ALMOST_EQUAL_TOL = 2    
    
class TestNCT(unittest.TestCase):
    
    def setUp(self):
        self.config = Config(np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.body = Body(self.config, [Organ(self.config,[0.5,0.5],0.2,0.2,'simple_sin','const2'),
                                       Organ(self.config,[0.2,0.2],0.2,0.2,'simple_sin','const2')])
        
        if os.path.exists('test_outputs'):
            os.systems('cd test_outputs && rm *')
        
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
        
       
        motion = Motion(self.config)
        
        for input in [1,'a','abc',None,3.0]:
            self.assertRaises(AssertionError, motion.get_motion, input, 'const')
            
        for input in [1,'a','abc',None,3.0]:
            self.assertRaises(AssertionError, motion.get_motion, 'const', input)
            
    def test_motion_models(self):
        
        # Test t input for different motion models
        motion = Motion(self.config)
        
        wrong_time_inputs = [-2.0, -1, -0.1, [0.1,0,5], None, '@', 'abc', 0, 1, 1.1, 10]
        
        for func in ['const','simple_sin','coupled_sin','pseudo_heart']:
            
            model1, model2 = motion.get_motion(func, func)
            # Test that value at t=0 and t=1 are 0
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
                
    def test_organ_init(self):
        
        # Test for config
        for input in ['a',0.2,[1.0,2.0],np.array([[1,2]]),None]:
            self.assertRaises(AssertionError,Organ,input,[0.5,0.5],1.0,1.0,'const','const2')
        self.assertRaises(TypeError,Motion,)
        
        # Test for center
        for input in ['a',['a','b'],0.2,[1,2],[0.5,1.1],np.array([[0.1,0.2]]),None]:
            self.assertRaises(AssertionError,Organ,self.config,input,1.0,1.0,'const','const2')
            
        # Test for radii
        for input in [1,'a','abc',None,-3.0,[0.1,0.3],np.array([0.2,0.1])]:
            self.assertRaises(AssertionError,Organ,self.config,[0.5,0.5],input,1.0,'const','const2')
        for input in [1,'a','abc',None,-3.0,[0.1,0.3],np.array([0.2,0.1])]:
            self.assertRaises(AssertionError,Organ,self.config,[0.5,0.5],1.0,input,'const','const2')
        
        # Test for size and location
        for input in [1,'a','abc',None,-3.0,[0.1,0.3],np.array([0.2,0.1]),['const'],['const','const2']]:
            self.assertRaises(AssertionError,Organ,self.config,[0.5,0.5],1.0,1.0,input,'const2')
        for input in [1,'a','abc',None,-3.0,[0.1,0.3],np.array([0.2,0.1]),['const'],['const','const2']]:
            self.assertRaises(AssertionError,Organ,self.config,[0.5,0.5],1.0,1.0,'const',input)
        
    def test_organ_ellipse(self):
        
        organ = Organ(self.config, [0.0,0.0],1.0,1.0,'const','const2')
        
        for input in [[[0.1,0.2],[0.5,-0.4]],np.array([[0.1,0.2],[0.5,np.nan]]),np.array([0.1,0.2])]:
            self.assertRaises(AssertionError,organ.ellipse,input,1.0,1.0)
        for input in [1,'a','abc',None,-3.0,[0.1,0.3],np.array([0.2,0.1])]:
            self.assertRaises(AssertionError,organ.ellipse,np.array([[0.1,0.2],[0.5,0.1]]),input,1.0)
        for input in [1,'a','abc',None,-3.0,[0.1,0.3],np.array([0.2,0.1])]:
            self.assertRaises(AssertionError,organ.ellipse,np.array([[0.1,0.2],[0.5,0.1]]),1.0,input)
            
        self.assertEqual(len(organ.ellipse(np.array([[0.1,0.2]]), 0.3,0.3).shape), 2)
        self.assertEqual(organ.ellipse(np.array([[0.5,0.5]]), 1.0,1.0), 1)
        self.assertEqual(organ.ellipse(np.array([[-0.5,0.2]]), 1.0,1.0), 1)
        self.assertEqual(organ.ellipse(np.array([[1.1,0.2]]), 1.0,1.0), 0)
        
        
    def test_organ_get_phase(self):
        
        # Testing for Type 0
        config = Config(np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        organ = Organ(config, [0.0,0.0],1.0,1.0,'const','const2')
        
        self.assertAlmostEqual(organ.get_phase(0),0,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(config.THETA_MAX),config.NUM_HEART_BEATS*
                         config.GANTRY_ROTATION_PERIOD/config.HEART_BEAT_PERIOD,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(-config.THETA_MAX),1-config.NUM_HEART_BEATS*
                         config.GANTRY_ROTATION_PERIOD/config.HEART_BEAT_PERIOD,ALMOST_EQUAL_TOL)
        
        # Testing for Type 1
        config = Config(np.array([[0.3,0.6]]), TYPE=1, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        organ = Organ(config, [0.0,0.0],1.0,1.0,'const','const2')
        self.assertAlmostEqual(organ.get_phase(0),0,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(config.THETA_MAX),0,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(-config.THETA_MAX),0,ALMOST_EQUAL_TOL)
        
        # Testing for Type 2
        for num in [1.0,2.0,3.0]:
            config = Config(np.array([[0.3,0.6]]), TYPE=2, NUM_HEART_BEATS=num, NUM_SDFS=2)
            organ = Organ(config, [0.0,0.0],1.0,1.0,'const','const2')
            self.assertAlmostEqual(organ.get_phase(0),0,ALMOST_EQUAL_TOL)
            self.assertAlmostEqual(organ.get_phase(config.THETA_MAX),0,ALMOST_EQUAL_TOL)
            self.assertAlmostEqual(organ.get_phase(int(config.THETA_MAX/num)),0,ALMOST_EQUAL_TOL)
        
    def test_organ_is_inside(self):
        
        config = Config(np.array([[0.3,0.6]]), TYPE=1, NUM_HEART_BEATS=1.0, NUM_SDFS=2)
        organ = Organ(config,[0.0,0.0],0.5,0.5,'simple_sin','const2')
        
        # Test for inputs
        for input in [[0.1,0.2],[[0.1,0.2]], np.array([0.1,0.2]), 'abs']:
            self.assertRaises(AssertionError, organ.is_inside, input, 0)
        
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, organ.is_inside, np.array([[0.5,0.5]]), input)
            
        # Test for functionality
        # Test for different times
        self.assertEqual(organ.is_inside(np.array([[0.4,0.4]]),0), 0)
        self.assertEqual(organ.is_inside(np.array([[0.4,0.4]]),int(0.3*config.THETA_MAX)), 1)
        
        self.assertEqual(organ.is_inside(np.array([[0.7,0.7]]),0), 0)
        self.assertEqual(organ.is_inside(np.array([[0.1,0.1]]),0), 1)
        
        # Change location of center
        organ = Organ(config,[0.5,0.5],0.5,0.5,'simple_sin','const2')
        self.assertEqual(organ.is_inside(np.array([[0.7,0.7]]),0), 1)
        self.assertEqual(organ.is_inside(np.array([[0.1,0.1]]),0), 0)
        
        
    def test_body_init(self):
        
        for input in ['a',0.2,[1.0,2.0],np.array([[1,2]]),None, Organ(self.config,[0.5,0.5],0.5,0.5,'simple_sin','const2'),[Organ(self.config,[0.5,0.5],0.5,0.5,'simple_sin','const2')],[Organ(self.config,[0.5,0.5],0.5,0.5,'simple_sin','const2'),1]]:
            self.assertRaises(AssertionError,Body,self.config,input)
        
    def test_body_is_inside(self):
        
        organs = [Organ(self.config,[0.5,0.5],0.2,0.2,'simple_sin','const2'),
                  Organ(self.config,[0.2,0.2],0.2,0.2,'simple_sin','const2')]
        
        body = Body(self.config,organs)
        
        pt = np.array([[0.1,0.2],
                       [0.4,0.4],
                       [0.9,0.9]])
        
        correct_insides = np.array([[0,1],
                                    [1,0],
                                    [0,0]])
        
        self.assertEqual(np.linalg.norm(body.is_inside(pt,0)-correct_insides), 0)
        
    def test_sdfgt_init(self):
        
        body = Body(self.config, [Organ(self.config,[0.5,0.5],0.2,0.2,'simple_sin','const2'),
                                  Organ(self.config,[0.2,0.2],0.2,0.2,'simple_sin','const2')])
        
        # Test for config
        for input in ['a',0.2,[1.0,2.0],np.array([[1,2]]),None]:
            self.assertRaises(AssertionError,SDFGt, input,body)
            
        # Test for body
        for input in ['a',0.2,[1.0,2.0],np.array([[1,2]]),None]:
            self.assertRaises(AssertionError,SDFGt, self.config, input)
           
        config = Config(np.array([[0.3]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        
        body = Body(config, [Organ(config,[0.5,0.5],0.2,0.2,'simple_sin','const2')])
        self.assertRaises(AssertionError,SDFGt, self.config, body)
        
            
    def test_sdfgt_forward(self):
        
        sdf = SDFGt(self.config, self.body)
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, sdf.forward, input, True)
            
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, sdf.forward, 0, input)
        
        image = sdf.forward(0, True).detach().cpu().numpy().reshape(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)
        np.save('test_outputs/sdfgt_forward_combine_true',image)
        
        image = sdf.forward(0, False).detach().cpu().numpy().reshape(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION,self.config.INTENSITIES.shape[1])
        np.save('test_outputs/sdfgt_forward_combine_false',image)
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    unittest.main()