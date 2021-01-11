import unittest
from config import *
import numpy as np
        
class TestNCT(unittest.TestCase):
    
    def test_config_inputs(self):

        # Test for Intensity
        self.assertRaises(AssertionError,config,0.2, TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config,[0.2], TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config,[[0.2]], TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config,[[0.3,0.7]], TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array(0.3), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array([0.3]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array([[[0.3]]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, 'High', TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        
        # Test for TYPE
        self.assertRaises(AssertionError,config, np.array([[0.3,0.6]]), TYPE=-1, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array([[0.3,0.6]]), TYPE=5, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array([[0.3,0.6]]), TYPE=1.0, NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array([[0.3,0.6]]), TYPE='a', NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array([[0.3,0.6]]), TYPE=[0], NUM_HEART_BEATS=2.0, NUM_SDFS=2)
        self.assertRaises(AssertionError,config, np.array([[0.3,0.6]]), TYPE=[0,1], NUM_HEART_BEATS=2.0, NUM_SDFS=2)
                                            
        # Test for NUM_HEART_BEATS
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=-1, NUM_SDFS=2)
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2, NUM_SDFS=2)
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=[2,4], NUM_SDFS=2)
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=[2,4.0], NUM_SDFS=2)
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=10.0, NUM_SDFS=2)
        
        # Test for NUM_SDFS
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=-1)
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=0)
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS='one')
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=[0,3])
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=0.6)
        self.assertRaises(AssertionError,config,np.array([[0.3,0.6]]), TYPE=0, NUM_HEART_BEATS=2.0, NUM_SDFS=2.0)
        

if __name__ == '__main__':
    unittest.main()