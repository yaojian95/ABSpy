import unittest
import numpy as np
from abspy import abssep

class TestSeparator(unittest.TestCase):
    
    def test_init(self):
        test_ps_total = np.random.random_sample((192,))
        test_ps_noise = np.random.random_sample((192,))
        test_ps_noiseRMS = np.random.random_sample((192,))
        test_nside = 4
        test_lbin = 3
        test_sep = abssep(test_ps_total, test_ps_noise, test_ps_noiseRMS,
                          test_nside, test_lbin)
        self.assertListEqual(list(test_sep.total_ps),list(test_ps_total))
        self.assertListEqual(list(test_sep.noise_ps),list(test_ps_noise))
        self.assertListEqual(list(test_sep.noise_ps_rms),list(test_ps_noiseRMS))
        self.assertEqual(test_sep.nside,test_nside)
        self.assertEqual(test_sep.lmax,3*test_nside-1)
        self.assertEqual(test_sep.lbin,test_lbin)
        self.assertEqual(test_sep.shift,10.0)
        self.assertEqual(test_sep.cut,1.0)
        #
        test_lmax = 10
        test_shift = 23.0
        test_cut = 2.3
        test_sep = abssep(test_ps_total, test_ps_noise, test_ps_noiseRMS,
                          test_nside, test_lbin, test_lmax, test_shift, test_cut)
        self.assertEqual(test_sep.lmax,test_lmax)
        self.assertEqual(test_sep.shift,test_shift)
        self.assertEqual(test_sep.cut,test_cut)
