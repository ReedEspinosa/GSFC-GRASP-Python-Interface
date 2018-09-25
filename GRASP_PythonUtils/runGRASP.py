#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os.path
import warnings
import yaml
import re
import time
from datetime import datetime as dt # we want datetime.datetime
from datetime import timedelta
from shutil import copyfile, rmtree
from subprocess import Popen
import numpy as np

class graspDB(object):
    def __init__(self, listGraspRunObjs):
        self.grObjs = listGraspRunObjs
        
    def processData(self, maxCPUs=1, binPathGRASP='/usr/local/bin/grasp'):
        usedDirs = []
        rslts = []
        for grObj in self.grObjs:
            assert (not grObj.dirGRASP in usedDirs), "Each graspRun instance must use a unique directory!"
            grObj.writeSDATA()
        i = 0
        Nobjs = len(self.grObjs)
        pObjs = []
        while i < Nobjs:
            if sum([pObj.poll() is None for pObj in pObjs]) < maxCPUs:
                print('Starting a new thread for graspRun index %d/%d' % (i,Nobjs-1))
                pObjs.append(self.grObjs[i].runGRASP(True, binPathGRASP))
                i+=1
            time.sleep(0.1)
        while any([pObj.poll() is None for pObj in pObjs]): time.sleep(0.1)
        for grObj in self.grObjs:
            rslts.append(grObj.readOutput())
        return rslts
    
    
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
        assert (self.pathSDATA), 'Failed to read SDATA filename from '+self.pathYAML
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

    def runGRASP(self, parallel=False, binPathGRASP='/usr/local/bin/grasp'):
        if not self.pathSDATA:
            warnings.warn('You must call writeSDATA() before running GRASP!')
            return False
        pathNewYAML = os.path.join(self.dirGRASP, os.path.basename(self.pathYAML));
        copyfile(self.pathYAML, pathNewYAML) # copy each time so user can update orginal YAML
        self.pObj = Popen([binPathGRASP, pathNewYAML])
        if not parallel:
            self.pObj.wait()
            self.invRslt = self.readOutput()          
        return self.pObj # returns Popen object, (PopenObj.poll() is not None) == True when complete
            
    def readOutput(self, customOUT=False): # customSDATA is full path of unrelated SDATA file to read
        if not customOUT and not self.pObj:
            warnings.warn('You must call runGRASP() before reading the output!')
            return False
        if not customOUT and self.pObj.poll() is None:
            warnings.warn('GRASP has not yet terminated, output can only be read after retrieval is complete.')
            return False
        assert (customOUT or self.findStream_FN()), 'Failed to read stream filename from '+self.pathYAML
        outputFN = customOUT if customOUT else os.path.join(self.dirGRASP, self.findStream_FN())
        with open(outputFN) as fid:
            contents = fid.readlines()
        rsltAeroDict = self.parseOutAerosol(contents)
        rsltSurfDict = self.parseOutSurface(contents)
        rsltDict = [{**aero, **surf} for aero, surf in zip(rsltAeroDict, rsltSurfDict)]
        if self.tempDir and rsltDict and not customOUT:
            rmtree(self.dirGRASP)
        return rsltDict
    
    def parseOutDateTime(self, contents):
        results = []
        ptrnDate = re.compile('^[ ]*Date:[ ]+')
        ptrnTime = re.compile('^[ ]*Time:[ ]+')
        i = 0
        while i<len(contents) and len(results)==0:
            line = contents[i]
            if not ptrnDate.match(line) is None: # Date
                dtStrCln = line[ptrnDate.match(line).end():-1].split()
                dates_list = [dt.strptime(date, '%Y-%m-%d').date() for date in dtStrCln]
            if not ptrnTime.match(line) is None: # Time (should come after Date in output)
                dtStrCln = line[ptrnTime.match(line).end():-1].split()
                times_list = [dt.strptime(time, '%H:%M:%S').time() for time in dtStrCln]
                for j in range(len(times_list)): 
                    dtNow = dt.combine(dates_list[j], times_list[j])
                    results.append(dict(datetime=dtNow))
            i+=1
        return results

    def parseOutAerosol(self, contents):
        results = self.parseOutDateTime(contents)
        numericLn = re.compile('^[ ]*[0-9]+')
        singNumeric = re.compile('^[ ]*[0-9]+[ ]*$')
        ptrnPSD = re.compile('^[ ]*(Radius \(um\),)?[ ]*Size Distribution dV\/dlnr \(normalized')
        ptrnLN = re.compile('^[ ]*Parameters of lognormal SD')
        ptrnVol = re.compile('^[ ]*Aerosol volume concentration')
        ptrnSPH = re.compile('^[ ]*% of spherical particles')
        ptrnAOD = re.compile('^[ ]*Wavelength \(um\),[ ]+(Total_AOD|AOD_Total)')
        ptrnAODmode = re.compile('^[ ]*Wavelength \(um\),[ ]+AOD_Particle_mode')
        ptrnSSA = re.compile('^[ ]*Wavelength \(um\),[ ]+(SSA_Total|Total_SSA)')
        ptrnSSAmode = re.compile('^[ ]*Wavelength \(um\),[ ]+SSA_Particle_mode')
        ptrnRRI = re.compile('^[ ]*Wavelength \(um\), REAL Ref\. Index')
        ptrnIRI = re.compile('^[ ]*Wavelength \(um\), IMAG Ref\. Index')
        i = 0
        nsd = 0
        while i<len(contents):
            if not ptrnPSD.match(contents[i]) is None: # binned PSD
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                for dataRow in contents[i+1:lastLine]:
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        results[k]['r'] = np.append(results[k]['r'], dArr[0]) if 'r' in results[k] else dArr[0]
                        results[k]['dVdlnr'] = np.append(results[k]['dVdlnr'], dArr[k+1]) if 'dVdlnr' in results[k] else dArr[k+1]
                i = lastLine - 1
                nsd+=1
            if not ptrnLN.match(contents[i]) is None: # lognormal PSD
                mtch = re.search('[ ]*rv \(um\):[ ]*', contents[i+1])
                rvArr = np.array(contents[i+1][mtch.end():-1].split(), dtype='float64')
                mtch = re.search('[ ] ln\(sigma\):[ ]*', contents[i+2])
                sigArr = np.array(contents[i+2][mtch.end():-1].split(), dtype='float64')
                for k in range(len(results)):
                    results[k]['rv'] = np.append(results[k]['rv'], rvArr[k]) if 'rv' in results[k] else rvArr[k]
                    results[k]['sigma'] = np.append(results[k]['sigma'], sigArr[k]) if 'sigma' in results[k] else sigArr[k]
                i+=2
            if not ptrnVol.match(contents[i]) is None: # Volume Concentration
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                for dataRow in contents[i+1:lastLine]:
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        results[k]['vol'] = np.append(results[k]['vol'], dArr[k+1]) if 'vol' in results[k] else dArr[k+1]
                i = lastLine - 1
            if not ptrnSPH.match(contents[i]) is None: # spherical fraction
                sphVal = np.array(contents[i+1].split(), dtype='float64')
                for k in range(len(results)):
                    results[k]['sph'] = np.append(results[k]['sph'], sphVal[k+1]) if 'sph' in results[k] else sphVal[k+1]
                i+=1
            if not ptrnAOD.match(contents[i]) is None: # AOD (must come before all of the following fields)
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                for dataRow in contents[i+1:lastLine]:
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        results[k]['lambda'] = np.append(results[k]['lambda'], dArr[0]) if 'lambda' in results[k] else dArr[0]
                        results[k]['aod'] = np.append(results[k]['aod'], dArr[k+1]) if 'aod' in results[k] else dArr[k+1]
                nwl = lastLine - (i + 1)
                i = lastLine - 1
            if not ptrnAODmode.match(contents[i]) is None: # AOD by aerosol size mode
                for dataRow in contents[i+1:i+nwl+1]:
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        results[k]['aodMode'] = np.append(results[k]['aodMode'], dArr[k+1]) if 'aodMode' in results[k] else dArr[k+1]
                i = i + nwl
            if not ptrnSSA.match(contents[i]) is None: # SSA
                for dataRow in contents[i+1:i+nwl+1]:
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        results[k]['ssa'] = np.append(results[k]['ssa'], dArr[k+1]) if 'ssa' in results[k] else dArr[k+1]
                i = i + nwl
            if not ptrnSSAmode.match(contents[i]) is None: # SSA by aersol size mode
                for dataRow in contents[i+1:i+nwl+1]:
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        results[k]['ssaMode'] = np.append(results[k]['ssaMode'], dArr[k+1]) if 'ssaMode' in results[k] else dArr[k+1]
                i = i + nwl
            if not ptrnRRI.match(contents[i]) is None: # RRI by aersol size mode
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                for dataRow in contents[i+1:lastLine]:
                    if not singNumeric.match(dataRow):
                        dArr = np.array(dataRow.split(), dtype='float64')
                        for k in range(len(results)):
                            results[k]['n'] = np.append(results[k]['n'], dArr[k+1]) if 'n' in results[k] else dArr[k+1]
            if not ptrnIRI.match(contents[i]) is None: # RRI by aersol size mode
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                for dataRow in contents[i+1:lastLine]:
                    if not singNumeric.match(dataRow):
                        dArr = np.array(dataRow.split(), dtype='float64')
                        for k in range(len(results)):
                            results[k]['k'] = np.append(results[k]['k'], dArr[k+1]) if 'k' in results[k] else dArr[k+1]
                i = lastLine - 1
            i+=1
        if nsd > 1:
            for k in range(len(results)): # seperate aerosol modes 
                results[k]['r'] = results[k]['r'].reshape(nsd,-1)
                results[k]['dVdlnr'] = results[k]['dVdlnr'].reshape(nsd,-1)
                results[k]['aodMode'] = results[k]['aodMode'].reshape(nsd,-1)
                results[k]['ssaMode'] = results[k]['ssaMode'].reshape(nsd,-1)
                if len(results[k]['n']) == len(results[k]['aodMode']): # RI not always mode dependent
                    results[k]['n'] = results[k]['n'].reshape(nsd,-1)
                    results[k]['k'] = results[k]['k'].reshape(nsd,-1)
        return results
    
    def parseOutSurface(self, contents):
        results = self.parseOutDateTime(contents)
        numericLn = re.compile('^[ ]*[0-9]+')
        singNumeric = re.compile('^[ ]*[0-9]+[ ]*$')
        ptrnALB = re.compile('^[ ]*Wavelength \(um\),[ ]+Surface ALBEDO')
        ptrnBRDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BRDF parameters')
        ptrnBPDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BPDF parameters')
        ptrnWater = re.compile('^[ ]*Wavelength \(um\),[ ]+Water surface parameters')
        i = 0
        while i<len(contents):
            if not ptrnALB.match(contents[i]) is None: # Surface Albedo 
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                for dataRow in contents[i+1:lastLine]:
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        results[k]['albedo'] = np.append(results[k]['albedo'], dArr[k+1]) if 'albedo' in results[k] else dArr[k+1]
                i = lastLine - 1
            if not ptrnBRDF.match(contents[i]) is None: # RRI by aersol size mode
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                Nparams = 0
                for dataRow in contents[i+1:lastLine]:
                    if not singNumeric.match(dataRow):
                        dArr = np.array(dataRow.split(), dtype='float64')
                        for k in range(len(results)):
                            results[k]['brdf'] = np.append(results[k]['brdf'], dArr[k+1]) if 'brdf' in results[k] else dArr[k+1]
                    else:
                        Nparams += 1
                for k in range(len(results)): # seperate parameters from wavelengths
                    results[k]['brdf'] = results[k]['brdf'].reshape(Nparams,-1)
                i = lastLine - 1
            if not ptrnBPDF.match(contents[i]) is None: # RRI by aersol size mode
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                Nparams = 0
                for dataRow in contents[i+1:lastLine]:
                    if not singNumeric.match(dataRow):
                        dArr = np.array(dataRow.split(), dtype='float64')
                        for k in range(len(results)):
                            results[k]['bpdf'] = np.append(results[k]['bpdf'], dArr[k+1]) if 'bpdf' in results[k] else dArr[k+1]
                    else:
                        Nparams += 1
                for k in range(len(results)): # seperate parameters from wavelengths
                    results[k]['bpdf'] = results[k]['bpdf'].reshape(Nparams,-1)
                i = lastLine - 1
            if not ptrnWater.match(contents[i]) is None: # RRI by aersol size mode
                lastLine = i+1
                while not numericLn.match(contents[lastLine]) is None: lastLine+=1
                Nparams = 0
                for dataRow in contents[i+1:lastLine]:
                    if not singNumeric.match(dataRow):
                        dArr = np.array(dataRow.split(), dtype='float64')
                        for k in range(len(results)):
                            results[k]['wtrSurf'] = np.append(results[k]['wtrSurf'], dArr[k+1]) if 'wtrSurf' in results[k] else dArr[k+1]
                    else:
                        Nparams += 1
                for k in range(len(results)): # seperate parameters from wavelengths
                    results[k]['wtrSurf'] = results[k]['wtrSurf'].reshape(Nparams,-1)
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
    
         
                  
                  
     
     
     
     
     
     
     
                  