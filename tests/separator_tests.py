import unittest
import numpy as np
from abspy.methods.abs import abssep

class TestSeparator(unittest.TestCase):
    
    def test_init(self):
        test_ps_total = list(np.random.random_sample((192)))
        test_ps_noise = list(np.random.random_sample((192)))
        test_ps_noiseRMS = list(np.random.random_sample((192)))
        test_lbin = 3
        test_sep = abssep(test_ps_total, test_ps_noise, test_ps_noiseRMS, test_lbin)
        self.assertListEqual(list(test_sep.total_ps), list(test_ps_total))
        self.assertListEqual(list(test_sep.noise_ps), list(test_ps_noise))
        self.assertListEqual(list(test_sep.noise_ps_rms), list(test_ps_noiseRMS))
        self.assertEqual(test_sep.lmax, len(test_ps_total))
        self.assertEqual(test_sep.lbin, test_lbin)
        self.assertEqual(test_sep.shift, 10.0)
        self.assertEqual(test_sep.cut, 1.0)
        #
        test_lmax = 10
        test_shift = 23.0
        test_cut = 2.3
        test_sep = abssep(test_ps_total, test_ps_noise, test_ps_noiseRMS,
                          test_lbin, test_lmax, test_shift, test_cut)
        self.assertEqual(test_sep.lmax, test_lmax)
        self.assertEqual(test_sep.shift, test_shift)
        self.assertEqual(test_sep.cut, test_cut)

    def test_bindl(self):
        test_cl = list(np.random.rand(128))
        test_cl_noise = list(np.random.rand(128))
        test_cl_noiseRMS = list(np.random.rand(128))
        test_sep = abssep(test_cl,
                          test_cl_noise,
                          test_cl_noiseRMS,
                          3,
                          lmax=100)
        test_ell, test_dl = test_sep.bindl(test_cl)
        for i in range(len(test_cl)):
            test_cl[i] *= 0.5*i*(i+1)/np.pi
        check_dl = [np.mean(test_cl[0:34]),
                    np.mean(test_cl[34:67]),
                    np.mean(test_cl[67:100])]
        for i in range(len(check_dl)):
                self.assertAlmostEqual(test_dl[i], check_dl[i])
        self.assertListEqual(test_ell, [17.0,50.5,83.5])


if __name__ == '__main__':
    unittest.main()
