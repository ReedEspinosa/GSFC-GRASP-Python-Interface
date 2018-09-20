#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os.path
import warnings
import yaml
import re
from datetime import datetime as dt # we really want datetime.datetime.striptime
from datetime import timedelta
from shutil import copyfile, rmtree
from subprocess import Popen
import numpy as np

class graspRun(object):
    def __init__(self, pathYAML, orbHghtKM=700, dirGRASP=False):
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
        self.pObj = False
    
    def addPix(self, newPixel): # this is called once for each pixel
        self.pixels.append(newPixel)
        
    def writeSDATA(self):
        if len(self.pixels) == 0:
            warnings.warn('You must call addPix() at least once before writting SDATA!')
            return False
        if not self.dirGRASP and self.tempDir:
                self.dirGRASP = tempfile.mkdtemp()
        self.pathSDATA = os.path.join(self.dirGRASP, self.findSDATA_FN());
        assert (not self.pathSDATA), 'Failed to read SDATA filename from '+self.pathYAML
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

    def runGRASP(self, parallel=False, pathGRASP='grasp'):
        if not self.pathSDATA:
            warnings.warn('You must call writeSDATA() before running GRASP!')
            return False
        pathNewYAML = os.path.join(self.dirGRASP, os.path.basename(self.pathYAML));
        copyfile(self.pathYAML, pathNewYAML) # copy each time so user can update orginal YAML
        self.pObj = Popen([pathGRASP, pathNewYAML])
        if not parallel:
            self.pObj.wait()
            self.readOutput()          
        return self.pObj # returns Popen object, (PopenObj.poll() is not None) == True when complete
            
    def readOutput(self):
        if not self.pObj:
            warnings.warn('You must call runGRASP() before reading the output!')
            return False
        if self.pObj.poll() is None:
            warnings.warn('GRASP has not yet terminated, output can only be read after retrieval is complete.')
            return False
        outputFN = self.findSDATA_FN()
        assert (not outputFN), 'Failed to read stream filename from '+self.pathYAML
        with open(os.path.join(self.dirGRASP, outputFN)) as fid:
            contents = fid.readlines()
        rsltDict = self.parseOutContent(contents)
        if self.tempDir and rsltDict:
            rmtree(self.dirGRASP)
        return rsltDict
           
    def parseOutContent(self, contents):
        results = []
        numericLn = re.compile('^[ ]*[0-9]+')
        ptrnDate = re.compile('^[ ]*Date:[ ]+')
        ptrnTime = re.compile('^[ ]*Time:[ ]+')
        ptrnHdEnd = re.compile('^[ ]*\*\*\*[ ]*[A-z ]+[ ]*\*\*\*[ ]*$')
        ptrnPSD = re.compile('^[ ]*Size Distribution dV\/dlnr \(normalized to 1\)')
        inHeader = True
        i = 0;
        while i<len(contents):
            line = contents[i]
            if not ptrnHdEnd.match(line) is None: inHeader=False # start of DETAILED PARAMETERS
            if not ptrnDate.match(line) is None and inHeader: # Date
                dtStrCln = line[ptrnDate.match(line).end():-1].split()
                dates_list = [dt.strptime(date, '%Y-%m-%d').date() for date in dtStrCln]
            if not ptrnTime.match(line) is None and inHeader: # Time (should come after Date in output)
                dtStrCln = line[ptrnTime.match(line).end():-1].split()
                times_list = [dt.strptime(time, '%H:%M:%S').time() for time in dtStrCln]
                for j in range(len(times_list)): 
                    dtNow = dt.combine(dates_list[j], times_list[j])
                    results.append(dict(datetime=dtNow))
            if not ptrnPSD.match(line) is None: # binned PSD
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                for dataRow in contents[i+1:lastLine]:
                    for k in range(len(results)):
                        rVal = np.float64(dataRow.split()[0])
                        results[k]['r'] = np.append(results[k]['r'], rVal) if 'r' in results[k] else rVal
                        psdVal = np.float64(dataRow.split()[k+1])
                        results[k]['dVdlnr'] = np.append(results[k]['dVdlnr'], psdVal) if 'dVdlnr' in results[k] else psdVal
                i = lastLine - 1
            i+=1
        return results
    
    def findSDATA_FN(self):
        if not os.path.isfile(self.pathYAML):
            warnings.warn('The file '+self.pathYAML+' does not exist! Can not get SDATA filename.', stacklevel=2)
            return False
        with open(self.pathYAML, 'r') as stream:
            data_loaded = yaml.load(stream)
        if not ('input'  in data_loaded) or not ('file' in data_loaded['input']):
            warnings.warn('The file '+self.pathYAML+' did not have field input.file! Can not get SDATA filename.', stacklevel=2)
            return False
        return data_loaded["input"]["file"]
    
    def findStream_FN(self):
        if not os.path.isfile(self.pathYAML):
            warnings.warn('The file '+self.pathYAML+' does not exist! Can not get stream filename.', stacklevel=2)
            return False
        with open(self.pathYAML, 'r') as stream:
            data_loaded = yaml.load(stream)
        if not ('input'  in data_loaded) or not ('file' in data_loaded['input']):
            warnings.warn('The file '+self.pathYAML+' did not have field output.segment.stream! Can not get stream filename.', stacklevel=2)
            return False
        return data_loaded["output"]["segment"]["stream"]
    
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
    
         
                  
                  
     
     
     
     
     
     
     
                  