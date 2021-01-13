import unittest
import numpy as np

from config import Config
from anatomy import Motion, Organ, Body
from renderer import *

import os
        
ALMOST_EQUAL_TOL = 2    
    
class TestNCT(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        ''
#         if os.path.exists('test_outputs'):
#             os.system('cd test_outputs && rm *')
            
#         else:
#             os.system('mkdir test_outputs')
            
    def setUp(self):
        self.config = Config(np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.body = Body(self.config, [Organ(self.config,[0.6,0.6],0.2,0.2,'simple_sin','const2'),
                                       Organ(self.config,[0.2,0.2],0.2,0.2,'simple_sin','const2')])
        
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
        
        self.assertAlmostEqual(organ.get_phase(0.0),0,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(1.0*config.THETA_MAX),config.NUM_HEART_BEATS*
                         config.GANTRY_ROTATION_PERIOD/config.HEART_BEAT_PERIOD,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(-1.0*config.THETA_MAX),1-config.NUM_HEART_BEATS*
                         config.GANTRY_ROTATION_PERIOD/config.HEART_BEAT_PERIOD,ALMOST_EQUAL_TOL)
        
        # Testing for Type 1
        config = Config(np.array([[0.3,0.6]]), TYPE=1, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        organ = Organ(config, [0.0,0.0],1.0,1.0,'const','const2')
        self.assertAlmostEqual(organ.get_phase(0.0),0,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(1.0*config.THETA_MAX),0,ALMOST_EQUAL_TOL)
        self.assertAlmostEqual(organ.get_phase(-1.0*config.THETA_MAX),0,ALMOST_EQUAL_TOL)
        
        # Testing for Type 2
        for num in [1.0,2.0,3.0]:
            config = Config(np.array([[0.3,0.6]]), TYPE=2, NUM_HEART_BEATS=num, NUM_SDFS=2)
            organ = Organ(config, [0.0,0.0],1.0,1.0,'const','const2')
            self.assertAlmostEqual(organ.get_phase(0.0),0,ALMOST_EQUAL_TOL)
            self.assertAlmostEqual(organ.get_phase(1.0*config.THETA_MAX),0,ALMOST_EQUAL_TOL)
            self.assertAlmostEqual(organ.get_phase(config.THETA_MAX/num),0,ALMOST_EQUAL_TOL)
        
    def test_organ_is_inside(self):
        
        config = Config(np.array([[0.3,0.6]]), TYPE=1, NUM_HEART_BEATS=1.0, NUM_SDFS=2)
        organ = Organ(config,[0.0,0.0],0.5,0.5,'simple_sin','const2')
        
        # Test for inputs
        for input in [[0.1,0.2],[[0.1,0.2]], np.array([0.1,0.2]), 'abs']:
            self.assertRaises(AssertionError, organ.is_inside, input, 0)
        
        for input in [-99999, 0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, organ.is_inside, np.array([[0.5,0.5]]), input)
            
        # Test for functionality
        # Test for different times
        self.assertEqual(organ.is_inside(np.array([[0.4,0.4]]),0.0), 0)
        self.assertEqual(organ.is_inside(np.array([[0.4,0.4]]),0.3*config.THETA_MAX), 1)
        
        self.assertEqual(organ.is_inside(np.array([[0.7,0.7]]),0.0), 0)
        self.assertEqual(organ.is_inside(np.array([[0.1,0.1]]),0.0), 1)
        
        # Change location of center
        organ = Organ(config,[0.5,0.5],0.5,0.5,'simple_sin','const2')
        self.assertEqual(organ.is_inside(np.array([[0.7,0.7]]),0.0), 1)
        self.assertEqual(organ.is_inside(np.array([[0.1,0.1]]),0.0), 0)
        
        
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
        
        self.assertEqual(np.linalg.norm(body.is_inside(pt,0.0)-correct_insides), 0.0)
        
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
        for input in [-99999, 0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, sdf.forward, input, True)
            
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, sdf.forward, 0, input)
            
        for t in [0.0,0.3*self.config.THETA_MAX, 1.0*self.config.THETA_MAX]:
            image = sdf.forward(t, True).detach().cpu().numpy().reshape(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)
            np.save('test_outputs/sdfgt_forward_combine_true',image)

            image = sdf.forward(t, False).detach().cpu().numpy().reshape(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION,self.config.INTENSITIES.shape[1])
            np.save('test_outputs/sdfgt_forward_combine_false',image)
        
        # Test for single organ
        config = Config(np.array([[0.3]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=1)
        body = Body(config, [Organ(config,[0.6,0.6],0.2,0.2,'simple_sin','const2')])
        sdf = SDFGt(config, body)
        sdf.forward(0.0, True)
        sdf.forward(0.0, False)
        
        
    def test_sdf_to_occ(self):
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([0]),torch.Tensor([0,1]),torch.Tensor([[0,1]])]:
            self.assertRaises(AssertionError, sdf_to_occ, input)
            
        for input, output in zip([-99,-1,-0.1,-0.01,0.0,0.01,0.1,1,99],
                                 [  0, 0,   0,    0,  0,0.12,  1,1, 1]):
            
            self.assertAlmostEqual(np.linalg.norm(
                sdf_to_occ(torch.Tensor([input]).view(1,1,1)).numpy()-output),0,ALMOST_EQUAL_TOL)
            
#     def test_occ_to_sdf(self):
#         for input in [0.0, None, np.nan, 'abc', [0], np.array([0]), torch.Tensor([[[0,1]]]), np.array([[0,1],[2,3]]), np.random.rand(10,10,3)]:
#             self.assertRaises(AssertionError, occ_to_sdf, input)
            
#         sdf = SDFGt(self.config, self.body).forward(0.0,False)
#         print(sdf.shape)
#         occ = sdf_to_occ(sdf)
            
    def test_renderer_init(self):
        
        sdf = SDF()
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, Renderer, input, sdf)
            
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, Renderer, self.config, input)
            
    def test_renderer(self):
        
        config = Config(np.array([[0.3]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=1)
        body = Body(config, [Organ(self.config,[0.6,0.6],0.2,0.2,'const','const2')])
        
        # Test Snapshot
        sdf = SDFGt(config, body)
        renderer = Renderer(config, sdf)
        for input in [-99999, 0, None, np.nan, 'a', 'abc', [0], np.array([0])]:
            self.assertRaises(AssertionError, renderer.snapshot, input)
            
        for t in [0.0,0.3*config.THETA_MAX, 1.0*self.config.THETA_MAX]:
            renderer.snapshot(t)

        # Test forward inputs
        for input in [-99999, 0.0, None, np.nan, 'a', 'abc', [0], np.array([[0]])]:
            self.assertRaises(AssertionError, renderer.forward, input)
        
        # Test compute_rigid_fbp inputs
        all_thetas = np.linspace(0,config.THETA_MAX, config.TOTAL_CLICKS)
        for input in [0.0, [[0.0,0.1],1,2], np.array([0.0,0.1]), torch.Tensor([[1,2],[3,4]]), 'abc']:
            self.assertRaises(AssertionError, renderer.compute_rigid_fbp, input, all_thetas)
        
        for input in [0.0, [[0.0,0.1],[1,2]], np.array([0.0,0.1]), np.array([[0.0,0.1],[1,2]]),torch.Tensor([[1,2]]), 'abc']:
            self.assertRaises(AssertionError, renderer.compute_rigid_fbp, np.zeros((10,20)), all_thetas)
        
        # Test by visualizing results of forward and rigid fbp
        sinogram = renderer.forward(all_thetas).detach().cpu().numpy()
        fbp = renderer.compute_rigid_fbp(sinogram,all_thetas)
        np.save('test_outputs/renderer_forward',sinogram)
        np.save('test_outputs/renderer_fbp',fbp)
        
        # Test if forward and and fbp are inverse of each other
        sdf = sdf.forward(0.0,True).detach().cpu().numpy().reshape(self.config.IMAGE_RESOLUTION,self.config.IMAGE_RESOLUTION)
        np.save('test_outputs/renderer_sdf',sdf)
        
        A = 1.0*(sdf > 0.01)
        B = 1.0*(fbp > 0.01)
        C = A-B
        
        self.assertAlmostEqual(np.linalg.norm(C)/(C.shape[0]*C.shape[1]), 0, ALMOST_EQUAL_TOL)
        
        
        
        
        
        
if __name__ == '__main__':
    unittest.main()