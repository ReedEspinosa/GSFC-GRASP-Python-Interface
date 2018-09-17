#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os.path
import warnings
import yaml

class graspRun(object):
   def __init__(self, pathYAML, dirTMP=tempfile.gettempdir()):
       pixels = [];
       self.pathYAML = pathYAML;
       sdataFN = self.findSDATA_FN();
       self.pathYAML = os.path.join(dirTMP,sdataFN);
       
   def findSDATA_FN(self):
        if not os.path.isfile(self.pathYAML):
            warnings.warn('The file '+self.pathYAML+' does not exist! Returning Generic SDATA filename.', stacklevel=2)
            return 'sdata.dat'
        with open(self.pathYAML, 'r') as stream:
            data_loaded = yaml.load(stream)
        if not ('input'  in data_loaded) or not ('file' in data_loaded['input']):
            warnings.warn('The file '+self.pathYAML+' did not have field input.file! Returning Generic SDATA filename.', stacklevel=2)
            return 'sdata.dat'
        return data_loaded["input"]["file"]
    
    def addPix(self, newPixel): # this is called once for each pixel
        # WE MAY WANT TO ADD SOME BULLET PROOFING HERE
        pixels.append(newPixel)
        
    def writeSDATA(self):
        print('INCOMPLETE!')

    def runGRASP(self):
        print('INCOMPLETE!')
   
    
class pixel(object):
     def __init__(self, ix, iy, lon, lat, masl, land_prct):
         self.ix = ix
         self.iy = iy
         self.lon = lon
         self.lat = lat
         self.masl = masl
         self.land_prct = land_prct
         self.nwl = 0
         self.measVals = []
         
     def addMeas(self, wl, nip, meas_type, nbvm, sza, thetav, phi, measurements): # this is called once for each wavelength of data
         newMeas = meas(wl, nip, meas_type, nbvm, sza, thetav, phi, measurements)
         self.measVals.append(newMeas)
         self.nwl += 1
         
    def genString(self):
        baseStrFrmt = '%2d %2d 1 0 0 %10.5f %6.1f %6.2f %d' # everything up to meas fields
        baseStr = str1Frmt % (self.ix, self.iy, self.lon, self.lat, self.masl, self.land_prct, self.nwl)
        wlStr = " ".join([str(obj.wl) for obj in self.measVals])
        nipStr = " ".join([str(obj.nip) for obj in self.measVals])
        meas_typeStr = " ".join([str(obj.meas_type) for obj in self.measVals])
        nbvmStr = " ".join([str(obj.nbvm) for obj in self.measVals])
        szaStr = " ".join([str(obj.sza) for obj in self.measVals])
        thetavStr = " ".join([str(obj.thetav) for obj in self.measVals])
        phiStr = " ".join([str(obj.phi) for obj in self.measVals])
        measStr = " ".join([str(obj.measurements) for obj in self.measVals])
        settingStr = '0 '*2*len(meas_typeStr.split(" "))
        measStrAll = " ".join((wlStr, nipStr, meas_typeStr, nbvmStr, szaStr, thetavStr, phiStr, measStr))
        return " ".join((baseStr, measStrAll, settingStr))
    
         
class meas(object):
     def __init__(self, wl, nip, meas_type, nbvm, sza, thetav, phi, measurements):
     print('INCOMPLETE!')    
                  
                  
     
     
     
     
     
     
     
                  