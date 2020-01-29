"""
The pseudo-PS estimation module,
by default it requires the NaMaster package.
"""
import pymaster as nmt
"""
For using other PS estimators,
please do your own estimation pipeline.
"""
import healpy as hp
import numpy as np
import logging as log
from abspy.tools.icy_decorator import icy

@icy
class pstimator(object):

    def __init__(self):
        pass
        
    def auto_t(maps, mask=None, aposcale=None, binning=None):
        """
        Auto PS,
        apply NaMaster estimator to T map with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : list, tuple
            A list of single T map.
            
        mask : numpy.ndarray
            mask map
            
        Returns
        -------
        
        pseudo-PS results : list
            [ell, TT]
        """
        assert isinstance(maps, (list,tuple))
        assert (len(maps)==1)
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask, aposcale, apotype='Smooth')
        _mapI = maps[0] - np.mean( maps[0][_apd_mask > 0] )
        # assemble NaMaster fields
        _f0 = nmt.NmtField(_apd_mask, [_mapI])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f0, _f0, _b)  # scalar - scalar
        return _b.get_effective_ells(), _cl00[0]
        
    def cross_t(maps, mask=None, aposcale=None, binning=None):
        """
        Cross PS,
        apply NaMaster estimator to T map with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : list, tuple
            A list of two T maps.
            
        mask : numpy.ndarray
            mask map
            
        Returns
        -------
        
        pseudo-PS results : list
            [ell, TT]
        """
        assert isinstance(maps, (list,tuple))
        assert (len(maps)==2)
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask, aposcale, apotype='Smooth')
        _mapI01 = maps[0] - np.mean( maps[0][_apd_mask > 0] )
        _mapI02 = maps[1] - np.mean( maps[1][_apd_mask > 0] )
        # assemble NaMaster fields
        _f01 = nmt.NmtField(_apd_mask, [_mapI01])
        _f02 = nmt.NmtField(_apd_mask, [_mapI02])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f01, _f02, _b)  # scalar - scalar
        return _b.get_effective_ells(), _cl00[0]
    
    def auto_eb(maps, mask=None, aposcale=None, binning=None):
        """
        Auto PS,
        apply NaMaster estimator to QU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : list/tuple
            A list of Q, U maps,
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            mask map
            
        Returns
        -------
        
        pseudo-PS results : list
            [ell, EE, BB]
        """
        assert isinstance(maps, (list,tuple))
        assert (len(maps)==2)
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask, aposcale, apotype='Smooth')
        _mapQ = maps[0] - np.mean( maps[0][_apd_mask > 0] )
        _mapU = maps[1] - np.mean( maps[1][_apd_mask > 0] )
        # assemble NaMaster fields
        _f2 = nmt.NmtField(_apd_mask, [_mapQ, _mapU])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl22 = nmt.compute_full_master(_f2, _f2, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl22[0], _cl22[3]
        
    def cross_eb(maps, mask=None, aposcale=None, binning=None):
        """
        Cross PS,
        apply NaMaster estimator to QU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : list/tuple
            A list of Q, U maps, arranged as {Q1, U1, Q2, U2},
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            mask map
            
        Returns
        -------
        
        pseudo-PS results : list
            [ell, EE, BB]
        """
        assert isinstance(maps, (list,tuple))
        assert (len(maps)==4)
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask, aposcale, apotype='Smooth')
        _mapQ01 = maps[0] - np.mean( maps[0][_apd_mask > 0] )
        _mapU01 = maps[1] - np.mean( maps[1][_apd_mask > 0] )
        _mapQ02 = maps[2] - np.mean( maps[2][_apd_mask > 0] )
        _mapU02 = maps[3] - np.mean( maps[3][_apd_mask > 0] )
        # assemble NaMaster fields
        _f21 = nmt.NmtField(_apd_mask, [_mapQ01, _mapU01])
        _f22 = nmt.NmtField(_apd_mask, [_mapQ02, _mapU02])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl22 = nmt.compute_full_master(_f21, _f22, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl22[0], _cl22[3]
    
    def auto_teb(maps, mask=None, aposcale=None, binning=None):
        """
        Auto PS,
        apply NaMaster estimator to TQU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : list/tuple
            A list of T, Q, U maps,
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            mask map
            
        Returns
        -------
        
        pseudo-PS results : list
            [ell, TT, EE, BB]
        """
        assert isinstance(maps, (list,tuple))
        assert (len(maps)==3)
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask, aposcale, apotype='Smooth')
        _mapI = maps[0] - np.mean( maps[0][_apd_mask > 0] )
        _mapQ = maps[1] - np.mean( maps[1][_apd_mask > 0] )
        _mapU = maps[2] - np.mean( maps[2][_apd_mask > 0] )
        # assemble NaMaster fields
        _f0 = nmt.NmtField(_apd_mask, [_mapI])
        _f2 = nmt.NmtField(_apd_mask, [_mapQ, _mapU])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f0, _f0, _b)  # scalar - scalar
        _cl22 = nmt.compute_full_master(_f2, _f2, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl00[0], _cl22[0], _cl22[3]
        
    def cross_teb(maps, mask=None, aposcale=None, binning=None):
        """
        Cross PS,
        apply NaMaster estimator to TQU maps with(out) masks,
        requires NaMaster, healpy, numpy packages.
        
        Parameters
        ----------
        
        maps : list/tuple
            A list of T, Q, U maps, arranged as {T,Q,U,T,Q,U},
            with polarization in CMB convention.
            
        mask : numpy.ndarray
            mask map
            
        Returns
        -------
        
        pseudo-PS results : list
            [ell, TT, EE, BB]
        """
        assert isinstance(maps, (list,tuple))
        assert (len(maps)==6)
        # fix resolution and apodization
        _nside = hp.get_nside(maps[0])
        # apodization
        if aposcale is None:
            aposcale = 1.0
        _apd_mask = nmt.mask_apodization(mask, aposcale, apotype='Smooth')
        _mapI01 = maps[0] - np.mean( maps[0][_apd_mask > 0] )
        _mapQ01 = maps[1] - np.mean( maps[1][_apd_mask > 0] )
        _mapU01 = maps[2] - np.mean( maps[2][_apd_mask > 0] )
        _mapI02 = maps[3] - np.mean( maps[3][_apd_mask > 0] )
        _mapQ02 = maps[4] - np.mean( maps[4][_apd_mask > 0] )
        _mapU02 = maps[5] - np.mean( maps[5][_apd_mask > 0] )
        # assemble NaMaster fields
        _f01 = nmt.NmtField(_apd_mask, [_mapI01])
        _f21 = nmt.NmtField(_apd_mask, [_mapQ01, _mapU01])
        _f02 = nmt.NmtField(_apd_mask, [_mapI02])
        _f22 = nmt.NmtField(_apd_mask, [_mapQ02, _mapU02])
        # initialize binning scheme with ? ells per bandpower
        if binning is None:
            binning = 16
        else:
            assert isinstance(binning, int)
        _b = nmt.NmtBin(_nside, nlb=binning)
        # MASTER estimator
        _cl00 = nmt.compute_full_master(_f01, _f02, _b)  # scalar - scalar
        _cl22 = nmt.compute_full_master(_f21, _f22, _b)  # tensor - tensor
        return _b.get_effective_ells(), _cl00[0], _cl22[0], _cl22[3]
