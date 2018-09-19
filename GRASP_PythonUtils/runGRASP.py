#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os.path
import warnings
import yaml
from datetime import datetime as dt # we really want datetime.datetime.striptime
from datetime import timedelta
from shutil import copyfile, rmtree
import numpy as np

class graspRun(object):
    def __init__(self, pathYAML, dirGRASP=False, orbHghtKM=700):
        self.pixels = [];
        self.pathYAML = pathYAML;
        self.pathSDATA = False;
        if not dirGRASP:
            self.tempDir = True
            self.dirGRASP = False
        else:
            self.tempDir = False
            self.dirGRASP = dirGRASP
        self.orbHght = orbHghtKM*1000
    
    def addPix(self, newPixel): # this is called once for each pixel
        self.pixels.append(newPixel)
        
    def writeSDATA(self):
        if len(self.pixels) == 0:
            warnings.warn('You must call addPix() at least once before writting SDATA!')
            return False
        if not self.dirGRASP and self.tempDir:
                self.dirGRASP = tempfile.mkdtemp()
        self.pathSDATA = os.path.join(self.dirGRASP, self.findSDATA_FN());
        unqTimes = np.unique([pix.dtNm for pix in self.pixels])
        SDATAstr = self.genSDATAHead(unqTimes)
        for unqTime in unqTimes:
            pixInd = np.nonzero(unqTime == [pix.dtNm for pix in self.pixels])[0]
            SDATAstr += self.genCellHead(pixInd)
            for ind in pixInd:
                SDATAstr += self.pixels[ind].genString()
        with open(self.pathSDATA, 'w') as fid:
            fid.write(SDATAstr)
            fid.close()

    def runGRASP(self, parallel):
        if not self.pathSDATA:
            warnings.warn('You must call writeSDATA() before running GRASP!')
            return False
        pathNewYAML = os.path.join(self.dirGRASP, os.path.basename(self.pathYAML));
        copyfile(self.pathYAML, pathNewYAML) # copy each time so user can update orginal YAML
        if parallel:
            print('Run GRASP async.') # STILL NEED TO IMPLEMENT THIS!
            # return obj from pop and let user wait then read
            # https://stackoverflow.com/questions/636561/how-can-i-run-an-external-command-asynchronously-from-python
        else:
            print('Run GRASP normal') # STILL NEED TO IMPLEMENT THIS!
            # https://stackoverflow.com/questions/4856583/how-do-i-pipe-a-subprocess-call-to-a-text-file
            self.readOutput()          
            
    def readOutput(self):
        if False: # CHANGE THIS TO CHECK IF GRASP HAS BEEN RUN
            warnings.warn('You must call runGRASP() before reading the output!')
            return False
        print('Read output from whatever') # STILL NEED TO IMPLEMENT THIS!
        if self.tempDir:
            rmtree(self.dirGRASP)
           
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
    
    def genSDATAHead(self, unqTimes):
        nx = max([pix.ix for pix in self.pixels])
        ny = max([pix.iy for pix in self.pixels])
        Nstr = ' %d %d %d : NX NY NT' % (nx, ny, len(unqTimes))
        return 'SDATA version 2.0\n%s\n' % Nstr
        
    def genCellHead(self, pixInd):
        nStr = '\n  %d   ' % len(pixInd)
        dtObjDay = dt.fromordinal(np.int(np.floor(self.pixels[pixInd[0]].dtNm)))
        dtObjTime = timedelta(seconds=np.remainder(self.pixels[pixInd[0]].dtNm, 1)*86400)
        dtObj = dtObjDay + dtObjTime
        dtStr = dtObj.strftime('%Y-%m-%dT%H:%M:%SZ')
        endstr = ' %10.2f   0   0\n' % self.orbHght
        return nStr+dtStr+endstr
        
   
class pixel(object):
    def __init__(self, dtNm, ix, iy, lon, lat, masl, land_prct):
        self.dtNm = dtNm
        self.ix = ix
        self.iy = iy
        self.lon = lon
        self.lat = lat
        self.masl = masl
        self.land_prct = land_prct
        self.nwl = 0
        self.measVals = []
         
    def addMeas(self, wl, msTyp, nbvm, sza, thtv, phi, msrmnts): # this is called once for each wavelength of data
        newMeas = dict(wl=wl, nip=len(msTyp), meas_type=msTyp, nbvm=nbvm, sza=sza, thetav=thtv, phi=phi, measurements=msrmnts)
        self.measVals.append(newMeas)
        self.nwl += 1
         
    def genString(self):
        baseStrFrmt = '%2d %2d 1 0 0 %10.5f %10.5f %6.1f %6.2f %d' # everything up to meas fields
        baseStr = baseStrFrmt % (self.ix, self.iy, self.lon, self.lat, self.masl, self.land_prct, self.nwl)
        wlStr = " ".join(['%6.4f' % obj['wl'] for obj in self.measVals])
        nipStr = " ".join(['%d' % obj['nip'] for obj in self.measVals])        
        allVals = np.block([obj['meas_type'] for obj in self.measVals])
        meas_typeStr = " ".join(['%d' % n for n in allVals])        
        allVals = np.block([obj['nbvm'] for obj in self.measVals])
        nbvmStr = " ".join(['%d' % n for n in allVals])        
        szaStr = " ".join(['%6.2f' % obj['sza'] for obj in self.measVals])        
        allVals = np.block([obj['thetav'] for obj in self.measVals])
        thetavStr = " ".join(['%6.2f' % n for n in allVals])        
        allVals = np.block([obj['phi'] for obj in self.measVals])
        phiStr = " ".join(['%6.2f' % n for n in allVals])
        allVals = np.block([obj['measurements'] for obj in self.measVals])
        measStr = " ".join(['%10.6f' % n for n in allVals])
        settingStr = '0 '*2*len(meas_typeStr.split(" "))
        measStrAll = " ".join((wlStr, nipStr, meas_typeStr, nbvmStr, szaStr, thetavStr, phiStr, measStr))
        return " ".join((baseStr, measStrAll, settingStr, '\n'))
    
         
                  
                  
     
     
     
     
     
     
     
                  