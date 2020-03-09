"""
the ABS separator class.
"""

import logging as log
import numpy as np
from copy import deepcopy
from abspy.tools.icy_decorator import icy


@icy
class abssep(object):
    
    def __init__(self, total_ps, noise_ps=None, noise_rms=None, lbin=None, llist=None, lmax=None, shift=10.0, cut=1.0):
        """
        ABS separator class initialization function.
        
        Parameters:
        -----------
        
        total_ps : numpy.ndarray
            The total CROSS power-sepctrum matrix,
            with global size (N_modes, N_freq, N_freq).
            * N_freq: number of frequency bands
            * N_modes: number of angular modes
            
        noise_ps : numpy.ndarray
            The ensemble averaged (instrumental) noise CROSS power-sepctrum,
            with global size (N_modes, N_freq, N_freq).
            * N_freq: number of frequency bands
            * N_modes: number of angular modes
            
        noise_rms : numpy.ndarray
            The RMS of ensemble (instrumental) noise AUTO power-spectrum,
            with global size (N_modes, N_freq).
            * N_freq: number of frequency bands
            * N_modes: number of angular modes
            
        lbin : unsigned integer
            The bin width of angular modes.
            
        llist : list, tuple
            The list of angular modes of given power spectra.
            
        lmax : unsigned integer
            The maximal angular modes.
            
        shift : double
            Global shift to the target power-spectrum,
            defined in Eq(3) of arXiv:1608.03707.
            
        cut : positive double
            The threshold of signal to noise ratio, for information extraction.
        """
        log.debug('@ abs::__init__')
        #
        self.total_ps = total_ps
        self.noise_ps = noise_ps
        self.noise_rms = noise_rms
        #
        self.lmax = lmax
        self.lbin = lbin
        self.llist = llist
        #
        self.shift = shift
        self.cut = cut
        #
        self.noise_flag = not (self._noise_ps is None or self._noise_rms is None)
        
    @property
    def total_ps(self):
        return self._total_ps
    
    @property
    def noise_ps(self):
        return self._noise_ps
    
    @property
    def noise_rms(self):
        return self._noise_rms
    
    @property
    def lbin(self):
        return self._lbin
        
    @property
    def llist(self):
        return self._llist
    
    @property
    def lmax(self):
        return self._lmax
    
    @property
    def shift(self):
        return self._shift
    
    @property
    def cut(self):
        return self._cut
    
    @property
    def resampling(self):
        return self._resampling
        
    @property
    def noise_flag(self):
        return self._noise_flag
        
    @total_ps.setter
    def total_ps(self, total_ps):
        assert isinstance(total_ps, np.ndarray)
        self._ps_lmax = total_ps.shape[0]
        self._ps_fmax = total_ps.shape[1]
        assert (total_ps.shape[1] == total_ps.shape[2])
        self._total_ps = total_ps
        log.debug('total cross-PS read')
        
    @noise_ps.setter
    def noise_ps(self, noise_ps):
        if noise_ps is None:
            log.debug('without noise cross-PS')
        else:
            assert isinstance(noise_ps, np.ndarray)
            assert (noise_ps.shape[0] == self._ps_lmax)
            assert (noise_ps.shape[1] == self._ps_fmax)
            assert (noise_ps.shape[1] == noise_ps.shape[2])
            log.debug('noise cross-PS read')
        self._noise_ps = noise_ps
        
    @noise_rms.setter
    def noise_rms(self, noise_rms):
        if noise_rms is None:
            log.debug('without noise RMS')
        else:
            assert isinstance(noise_rms, np.ndarray)
            assert (self._ps_lmax == noise_rms.shape[0])
            assert (self._ps_fmax == noise_rms.shape[1])
            log.debug('noise RMS auto-PS read')
        self._noise_rms = noise_rms
    
    @lmax.setter
    def lmax(self, lmax):
        if lmax is not None:
            assert isinstance(lmax, int)
            assert (lmax > 0 and lmax <= self._ps_lmax)
            self._lmax = lmax
        else:
            self._lmax = self._ps_lmax
        log.debug('angular mode maximum set as '+str(self._lmax))
        
    @llist.setter
    def llist(self, llist):
        if llist is not None:
            assert isinstance(llist, (list,tuple))
            assert (len(llist) >= self._lbin)
            self._llist = [x for x in llist if x <= self._lmax]
            self._prebin = True  # with pre binned input
        else:
            self._llist = [*range(self._lmax)]
            self._prebin = False
        log.debug('angular modes list set as '+str(self._llist))
    
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
        
    @resampling.setter
    def resampling(self, resampling):
        assert isinstance(resampling, int)
        assert (resampling > 0)
        self._resampling = resampling
        log.debug('resampling size set as '+str(self._resampling))
        
    @noise_flag.setter
    def noise_flag(self, noise_flag):
        assert isinstance(noise_flag, bool)
        self._noise_flag = noise_flag
        log.debug('ABS with noise? '+str(self._noise_flag))
        
    @property
    def binell(self):
        """
        Central angular modes "ell" of binned average.
        
        Returns
        -------
        
        Central angular modes position : numpy.ndarray.
        """
        log.debug('@ abs::binell')
        lnew = list()
        if (self._prebin):  # binned twice
            _lsample_size = len(self._llist)
        else:
            _lsample_size = self._lmax
        lres = _lsample_size%self._lbin
        lmod = _lsample_size//self._lbin
        # binned average for each single spectrum
        for i in range(self._lbin):
            begin = min(lres,i)+i*lmod
            end = min(lres,i) + (i+1)*lmod + int(i < lres)
            lnew.append(0.5*(self._llist[begin]+self._llist[end-1]))
        return lnew

    def bincps(self, cps):
        """
        Binned average of CROSS-power-spectrum and convert it into CROSS-Dl.
        
        Parameters
        ----------
        
        cps : numpy.ndarray
            cross power spectrum
            
        Returns
        -------
            
        CROSS-Dl with bin average : numpy.ndarray
        """
        log.debug('@ abs::bincps')
        assert isinstance(cps, np.ndarray)
        assert (cps.shape[0] == self._ps_lmax)
        assert (cps.shape[1] == self._ps_fmax)
        assert (cps.shape[1] == cps.shape[2])
        if (self._prebin):  # binned twice
            _lsample_size = len(self._llist)
        else:
            _lsample_size = self._lmax
        lres = _lsample_size%self._lbin
        lmod = _lsample_size//self._lbin
        result = np.empty((self._lbin, cps.shape[1], cps.shape[2]))
        _cps = deepcopy(cps)
        # convert Cl into Dl for each single spectrum
        for i in range(_lsample_size):
            _cps[i,:,:] *= 0.5*i*(i+1)/np.pi
        # binned average for each single spectrum
        for i in range(self._lbin):
            begin = min(lres,i)+i*lmod
            end = min(lres,i) + (i+1)*lmod + int(i < lres)
            result[i,:,:] = np.mean(_cps[begin:end,:,:], axis=0)
        return result
    
    def binaps(self, aps):
        """
        Binned average of AUTO-power-spectrum Cl and convert it into AUTO-Dl.
        
        Parameters
        ----------
        
        aps : numpy.ndarray
            auto power spectrum
        
        Returns
        -------
        
        AUTO-Dl with binned average : numpy.ndarray
        """
        log.debug('@ abs::binaps')
        assert isinstance(aps, np.ndarray)
        assert (aps.shape[0] == self._ps_lmax)
        assert (aps.shape[1] == self._ps_fmax)
        if (self._prebin):  # binned twice
            _lsample_size = len(self._llist)
        else:
            _lsample_size = self._lmax
        lres = _lsample_size%self._lbin
        lmod = _lsample_size//self._lbin
        result = np.empty((self._lbin, aps.shape[1]))
        _aps = deepcopy(aps)
        # convert Cl into Dl for each single spectrum
        for i in range(_lsample_size):
            _aps[i,:] *= 0.5*i*(i+1)/np.pi
        # binned average for each single spectrum
        for i in range(self._lbin):
            begin = min(lres,i)+i*lmod
            end = min(lres,i) + (i+1)*lmod + int(i < lres)
            result[i,:] = np.mean(_aps[begin:end,:], axis=0)
        return result
    
    def __call__(self):
        return self.run()
        
    def run(self):
        """
        ABS separator class call function.
        
        Returns
        -------
        
        angular modes, target angular power spectrum : (list, list)
        """
        log.debug('@ abs::__call__')
        # binned average
        _Dl = self.bincps(self._total_ps)
        if (self._noise_flag):
            _nDl = self.bincps(self._noise_ps)
            _nrmsDl = self.binaps(self._noise_rms)
        # prepare CMB f(ell, freq)
        _f = np.ones((self._lbin,self._total_ps.shape[1]), dtype=np.float64)
        if (self._noise_flag):
            _f /= _nrmsDl  # rescal f according to noise RMS
            # Dl_ij = Dl_ij/sqrt(sigma_li,sigma_lj) + shift*f_li*f_lj
            _Dl -= _nDl
            for i in range(self._ps_fmax):
                for j in range(self._ps_fmax):
                    _Dl[:,i,j] = _Dl[:,i,j]/np.sqrt(_nrmsDl[:,i]*_nrmsDl[:,j]) + self._shift*_f[:,i]*_f[:,j]
        else:
            # Dl_ij = Dl_ij + shift*f_li*f_lj
            for i in range(self._ps_fmax):
                for j in range(self._ps_fmax):
                    _Dl[:,i,j] += self._shift*_f[:,i]*_f[:,j]
        # find eign at each angular mode
        _Dbl = list()
        for ell in range(self._lbin):
            # eigvec[:,i] corresponds to eigval[i]
            # note that eigen values may be complex
            eigval, eigvec = np.linalg.eig(_Dl[ell])
            log.debug('@ abs::__call__, angular mode '+str(self.binell[ell])+' with eigen vals '+str(eigval))
            for i in range(self._ps_fmax):
                eigvec[:,i] /= np.linalg.norm(eigvec[:,i])**2
            _tmp = 0
            for i in range(self._ps_fmax):
                if eigval[i] >= self._cut:
                    _G = np.dot(_f[ell], eigvec[:,i])
                    _tmp += (_G**2/eigval[i])
            _Dbl.append(1.0/_tmp - self._shift)
        return (self.binell, _Dbl)
        
