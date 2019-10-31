# -*- coding: utf-8 -*-
"""
the ABS separator class
"""

import logging as log
import numpy as np
from abspy.tools.icy_decorator import icy


@icy
class abssep(object):
    
    def __init__(self, total_ps, noise_ps, noise_ps_rms, nside, lbin, lmax=None, shift=10.0, cut=1.0):
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
            
        nside:
            unsigned integer
            HEALPix resolution of
            
        lbin:
            unsigned integer
            the bin width of angular modes
            
        lmax:
            unsigned integer
            the maximal angular modes
            
        shift:
            double
            global shift to the target power-spectrum
            
        cut:
            positive double
            signal to noise ratio threshold for information extraction
            
        """
        log.debug('instantiating ABS class')
        #
        self.nside = nside
        self.lmax = lmax
        self.lbin = lbin
        #
        self.total_ps = total_ps
        self.noise_ps = noise_ps
        self.noise_ps_rms = noise_ps_rms
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
    def nside(self):
        return self._nside
    
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
        
    @nside.setter
    def nside(self, nside):
        assert isinstance(nside, int)
        assert (nside > 0)
        self._nside = nside
        log.debug('HEALPix Nside set as '+str(self._nside))
        
    @lmax.setter
    def lmax(self, lmax):
        if lmax is not None:
            assert isinstance(lmax, int)
            assert (lmax > 0 and lmax < 3*self._nside - 1)
            self._lmax = lmax
        else:
            self._lmax = 3*self._nside - 1
        log.debug('angular mode maximum set as '+str(self._lmax))
        
    @lbin.setter
    def lbin(self, lbin):
        assert isinstance(lbin, int)
        assert (lbin > 0 and lbin < self._lmax)
        self._lbin = lbin
        log.debug('angular mode bin width set as '+str(self._lbin))
        
    @total_ps.setter
    def total_ps(self, total_ps):
        assert isinstance(total_ps, (list,tuple,np.ndarray))
        assert (len(total_ps) >= self._lmax)
        self._total_ps = total_ps
        log.debug('total PS read')
        
    @noise_ps.setter
    def noise_ps(self, noise_ps):
        assert isinstance(noise_ps, (list,tuple,np.ndarray))
        assert (len(noise_ps) >= self._lmax)
        self._noise_ps = noise_ps
        log.debug('noise PS read')
        
    @noise_ps_rms.setter
    def noise_ps_rms(self, noise_ps_rms):
        assert isinstance(noise_ps_rms, (list,tuple,np.ndarray))
        assert (len(noise_ps_rms) >= self._lmax)
        self._noise_ps_rms = noise_ps_rms
        log.debug('noise PS RMS read')
        
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
