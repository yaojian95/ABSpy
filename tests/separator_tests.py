import unittest
import numpy as np
from abspy.methods.abs import abssep

class TestSeparator(unittest.TestCase):
    
    def test_init(self):
        test_ps_total = np.random.rand(192,3,3)
        test_ps_noise = np.random.rand(192,3,3)
        test_ps_noise_rms = np.random.rand(192,3)
        test_lbin = 3
        test_sep = abssep(test_ps_total,
                          test_ps_noise,
                          test_ps_noise_rms,
                          test_lbin)
        for i in range(test_ps_total.shape[0]):
            self.assertListEqual(list(test_sep.noise_rms[i]), list(test_ps_noise_rms[i]))
            for j in range(test_ps_total.shape[1]):
                self.assertEqual(list(test_sep.total_ps[i,j]), list(test_ps_total[i,j]))
                self.assertEqual(list(test_sep.noise_ps[i,j]), list(test_ps_noise[i,j]))
        self.assertEqual(test_sep.lmax, test_ps_total.shape[0])
        self.assertEqual(test_sep.lbin, test_lbin)
        self.assertEqual(test_sep.shift, 10.0)
        self.assertEqual(test_sep.cut, 1.0)
        #
        test_lmax = 10
        test_shift = 23.0
        test_cut = 2.3
        test_sep = abssep(test_ps_total,
                          test_ps_noise,
                          test_ps_noise_rms,
                          lbin=test_lbin,
                          llist=None,
                          lmax=test_lmax,
                          shift=test_shift,
                          cut=test_cut)
        self.assertEqual(test_sep.lmax, test_lmax)
        self.assertEqual(test_sep.shift, test_shift)
        self.assertEqual(test_sep.cut, test_cut)
        #
        test_llist = [*range(0,16,2)]
        test_lmax = 12
        test_sep = abssep(test_ps_total,
                          test_ps_noise,
                          test_ps_noise_rms,
                          lbin=test_lbin,
                          llist=test_llist,
                          lmax=test_lmax,
                          shift=test_shift,
                          cut=test_cut)
        self.assertListEqual(test_sep.binell, [2, 7, 11])
    
    def test_binning(self):
        test_ccl = np.random.rand(128,3,3)
        test_ccl_noise = np.random.rand(128,3,3)
        test_ccl_noise_rms = np.random.rand(128,3)
        binsize = 3
        test_sep = abssep(test_ccl,
                          test_ccl_noise,
                          test_ccl_noise_rms,
                          binsize,
                          lmax=100)
        test_ell = test_sep.binell
        self.assertListEqual(test_ell, [16.5,50.0,83.0])
        test_cdl = test_sep.bincps(test_ccl)
        test_ndl = test_sep.bincps(test_ccl_noise)
        test_rdl = test_sep.binaps(test_ccl_noise_rms)
        # calculate manually
        check_cdl = np.empty((binsize,test_ccl.shape[1],test_ccl.shape[2]))
        check_ndl = np.empty((binsize,test_ccl.shape[1],test_ccl.shape[2]))
        check_rdl = np.empty((binsize,test_ccl.shape[1]))
        for i in range(test_ccl.shape[0]):
            test_ccl[i,:,:] *= 0.5*i*(i+1)/np.pi
            test_ccl_noise[i,:,:] *= 0.5*i*(i+1)/np.pi
            test_ccl_noise_rms[i,:] *= 0.5*i*(i+1)/np.pi
        for i in range(test_ndl.shape[1]):
            check_rdl[:,i] = [np.mean(test_ccl_noise_rms[0:34,i]),
                     np.mean(test_ccl_noise_rms[34:67,i]),
                     np.mean(test_ccl_noise_rms[67:100,i])]
            for j in range(test_cdl.shape[2]):
                check_cdl[:,i,j] = [np.mean(test_ccl[0:34,i,j]),
                         np.mean(test_ccl[34:67,i,j]),
                         np.mean(test_ccl[67:100,i,j])]
                check_ndl[:,i,j] = [np.mean(test_ccl_noise[0:34,i,j]),
                         np.mean(test_ccl_noise[34:67,i,j]),
                         np.mean(test_ccl_noise[67:100,i,j])]
        for i in range(check_cdl.shape[0]):
            for k in range(check_cdl.shape[2]):
                self.assertAlmostEqual(test_rdl[i,k], check_rdl[i,k])
                for j in range(check_cdl.shape[1]):
                    self.assertAlmostEqual(test_cdl[i,j,k], check_cdl[i,j,k])
                    self.assertAlmostEqual(test_ndl[i,j,k], check_ndl[i,j,k])
    
    def test_abs(self):
        np.random.seed(234)
        test_ccl = np.random.rand(128,3,3)
        test_ccl_noise = np.random.rand(128,3,3)*0.01
        test_ccl_noise_rms = np.random.rand(128,3)*0.001
        binsize = 3
        test_sep = abssep(test_ccl,
                          test_ccl_noise,
                          test_ccl_noise_rms,
                          binsize,
                          lmax=100)
        test_result = test_sep()

if __name__ == '__main__':
    unittest.main()
