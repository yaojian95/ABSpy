# -*- coding: utf-8 -*-
"""
  
"""

#import numpy as np
from abs.tools.icy_decorator import icy

@icy
class ABS(object):
    
    def __init__(self, filename, nside, lmax, bin_width):
        """
        ABS class initialization function
        
        
        parameters:
        -----------
        
        filename:
            string
            path to the input PS file
            
        nside:
            unsigned integer
            HEALPix resolution of
            
        lmax:
            unsigned integer
            the maximal angular modes
            
        bin_width:
            unsigned integer
            the bin width of angular modes
            
        """
        self.filename = filename
        #self.nside = nside
        #self.lmax = lmax
        #self.bin_width = bin_width
        
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, filename):
        assert isinstance(filename, str)
        self._filename = filename