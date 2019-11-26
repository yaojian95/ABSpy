"""
the ABS separator class
"""

import logging as log
import numpy as np
from copy import deepcopy
from statistics import mean
from abspy.tools.icy_decorator import icy


@icy
class abssep(object):
    
    def __init__(self, total_ps, noise_ps, noise_ps_rms, lbin, lmax=None, shift=10.0, cut=1.0):
        """
        ABS separator class initialization function
        
        parameters:
        -----------
        
        total_ps:
            list, tuple or numpy.array
            the total power-sepctrum
            
        noise_ps:
            list, tuple or numpy.array
            the noise power-sepctrum
            
        noise_ps_rms:
            list, tuple or numpy.array
            RMS of the noise power-sepctrum
            
        lbin:
            unsigned integer
            the bin width of angular modes
            
        lmax:
            unsigned integer
            the maximal angular modes
            
        shift:
            double
            global shift to the target power-spectrum
            defined in Eq(3) of arXiv:1608.03707
            
        cut:
            positive double
            signal to noise ratio threshold for information extraction
            
        """
        log.debug('@ abs::__init__')
        #
        self.total_ps = total_ps
        self.noise_ps = noise_ps
        self.noise_ps_rms = noise_ps_rms
        #
        self.lmax = lmax
        self.lbin = lbin
        #
        self.shift = shift
        self.cut = cut
        
    @property
    def total_ps(self):
        return self._total_ps
    
    @property
    def noise_ps(self):
        return self._noise_ps
    
    @property
    def noise_ps_rms(self):
        return self._noise_ps_rms
    
    @property
    def lbin(self):
        return self._lbin
    
    @property
    def lmax(self):
        return self._lmax
    
    @property
    def shift(self):
        return self._shift
    
    @property
    def cut(self):
        return self._cut
        
    @total_ps.setter
    def total_ps(self, total_ps):
        assert isinstance(total_ps, (list,tuple))
        self._ps_lmax = len(total_ps)
        self._total_ps = total_ps
        log.debug('total PS read')
        
    @noise_ps.setter
    def noise_ps(self, noise_ps):
        assert isinstance(noise_ps, (list,tuple))
        assert (len(noise_ps) == self._ps_lmax)
        self._noise_ps = noise_ps
        log.debug('noise PS read')
        
    @noise_ps_rms.setter
    def noise_ps_rms(self, noise_ps_rms):
        assert isinstance(noise_ps_rms, (list,tuple))
        assert (len(noise_ps_rms) == self._ps_lmax)
        self._noise_ps_rms = noise_ps_rms
        log.debug('noise PS RMS read')
    
    @lmax.setter
    def lmax(self, lmax):
        if lmax is not None:
            assert isinstance(lmax, int)
            assert (lmax > 0 and lmax <= self._ps_lmax)
            self._lmax = lmax
        else:
            self._lmax = self._ps_lmax
        log.debug('angular mode maximum set as '+str(self._lmax))
    
    @lbin.setter
    def lbin(self, lbin):
        assert isinstance(lbin, int)
        assert (lbin > 0 and lbin <= self._lmax)
        self._lbin = lbin
        log.debug('angular mode bin width set as '+str(self._lbin))
        
    @shift.setter
    def shift(self, shift):
        assert isinstance(shift, float)
        assert (shift > 0)
        self._shift = shift
        log.debug('PS power shift set as '+str(self._shift))
        
    @cut.setter
    def cut(self, cut):
        assert isinstance(cut, float)
        assert (cut > 0)
        self._cut = cut
        log.debug('signal to noise threshold set as '+str(self._cut))

    def bindl(self, cl):
        """
        bin average of power-spectrum Cl and convert it into Dl
        
        parameters
        ----------
        
        cl
            power spectrum
            
        return
        ------
        central angular modes position, list
        and
        Dl with bin average, list
        """
        log.debug('@ abs::bindl')
        assert isinstance(cl, (list,tuple))
        lres = self._lmax%self._lbin
        lmod = self._lmax//self._lbin
        lnew = list()
        result = list()
        _cl = deepcopy(cl)
        for i in range(self._lmax):
            _cl[i] *= 0.5*i*(i+1)/np.pi
        for i in range(self._lbin):
            begin = min(lres,i)+i*lmod
            end = min(lres,i) + (i+1)*lmod + int(i < lres)
            lnew.append(0.5*(begin+end))
            result.append(mean(_cl[begin:end]))
        return lnew, result
