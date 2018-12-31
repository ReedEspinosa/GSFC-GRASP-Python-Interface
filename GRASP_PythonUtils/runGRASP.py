#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os.path
import warnings
import yaml
import re
import time
import pickle
from datetime import datetime as dt # we want datetime.datetime
from datetime import timedelta
from shutil import copyfile, rmtree
from subprocess import Popen
import numpy as np

class graspDB(object):
    def __init__(self, listGraspRunObjs=[]):
        self.grObjs = listGraspRunObjs
        
    def processData(self, maxCPUs=1, binPathGRASP='/usr/local/bin/grasp', savePath=False):
        usedDirs = []
        for grObj in self.grObjs:
            assert (not grObj.dirGRASP in usedDirs), "Each graspRun instance must use a unique directory!"
            grObj.writeSDATA()
        i = 0
        Nobjs = len(self.grObjs)
        pObjs = []
        while i < Nobjs:
            if sum([pObj.poll() is None for pObj in pObjs]) < maxCPUs:
                print('Starting a new thread for graspRun index %d/%d' % (i+1,Nobjs))
                pObjs.append(self.grObjs[i].runGRASP(True, binPathGRASP))
                i+=1
            time.sleep(0.1)
        while any([pObj.poll() is None for pObj in pObjs]): time.sleep(0.1)
        self.rslts = []
        [self.rslts.extend(grObj.readOutput()) for grObj in self.grObjs] # TODO: note if readOutput returns [] (i.e. error occured w/ that run)
        if savePath: 
            with open(savePath, 'wb') as f:
                pickle.dump(self.rslts, f, pickle.HIGHEST_PROTOCOL)
        return self.rslts
    
    def loadResults(self, loadPath):
        print('loadpath is '+loadPath)
        try:
            with open(loadPath, 'rb') as f:
                self.rslts = pickle.load(f)
            return self.rslts
        except EnvironmentError:
            warnings.warn('Could not load valid pickle data from %s.' % loadPath)
            return []  

# can add self.AUX_dict[Npixel] dictionary list to instance w/ additional fields to port into rslts
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
        try:
            with open(outputFN) as fid:
                contents = fid.readlines()
        except:
            warnings.warn('Could not open %s\n   Returning empty list in place of output data...' % outputFN)
            return []
        rsltAeroDict = self.parseOutAerosol(contents)
        if not rsltAeroDict:
            warnings.warn('Aerosol data could not be read from %s\n  Returning empty list in place of output data...' % outputFN)
            return []
        rsltSurfDict = self.parseOutSurface(contents)
        rsltFitDict = self.parseOutFit(contents)
        try:
            rsltDict = [{**aero, **surf, **fit, **aux} for aero, surf, fit, aux in zip(rsltAeroDict, rsltSurfDict, rsltFitDict, self.AUX_dict)] 
        except AttributeError: # the user did not create AUX_dict
            rsltDict = [{**aero, **surf, **fit} for aero, surf, fit in zip(rsltAeroDict, rsltSurfDict, rsltFitDict)]
        if self.tempDir and rsltDict and not customOUT:
            rmtree(self.dirGRASP)
        return rsltDict
    
    def parseOutDateTime(self, contents):
        results = []
        ptrnDate = re.compile('^[ ]*Date[ ]*:[ ]+')
        ptrnTime = re.compile('^[ ]*Time[ ]*:[ ]+')
        ptrnLon = re.compile('^[ ]*Longitude[ ]*:[ ]+')
        ptrnLat = re.compile('^[ ]*Latitude[ ]*:[ ]+')
        i = 0
        LatLonLinesFnd = 0 # GRASP prints these twice, assume lat/lon are last lines
        while i<len(contents) and LatLonLinesFnd<2:
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
            if not ptrnLon.match(line) is None: # longitude
                lonVals = np.array(line[ptrnLon.match(line).end():-1].split(), dtype=np.float)
                for k,lon in enumerate(lonVals):
                    results[k]['longitude'] = lon
                LatLonLinesFnd += 1
            if not ptrnLat.match(line) is None: # longitude
                latVals = np.array(line[ptrnLat.match(line).end():-1].split(), dtype=np.float)
                for k,lat in enumerate(latVals):
                    results[k]['latitude'] = lat
                LatLonLinesFnd += 1
            i+=1
        if not len(results[-1].keys())==3:
            warnings.warn('Failure reading date/lat/lon from GRASP output!')
            return []
        return results

    def parseOutAerosol(self, contents):
        results = self.parseOutDateTime(contents)
        if len(results)==0:
            return []
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
        ptrnReff = re.compile('^[ ]*reff total[ ]*([0-9Ee.+\- ]+)[ ]*$')
        i = 0
        nsd = 0
        while i<len(contents):
            if not ptrnLN.match(contents[i]) is None: # lognormal PSD, these fields have unique form
                mtch = re.search('[ ]*rv \(um\):[ ]*', contents[i+1])
                rvArr = np.array(contents[i+1][mtch.end():-1].split(), dtype='float64')
                mtch = re.search('[ ]*ln\(sigma\):[ ]*', contents[i+2])
                sigArr = np.array(contents[i+2][mtch.end():-1].split(), dtype='float64')
                for k in range(len(results)):
                    results[k]['rv'] = np.append(results[k]['rv'], rvArr[k]) if 'rv' in results[k] else rvArr[k]
                    results[k]['sigma'] = np.append(results[k]['sigma'], sigArr[k]) if 'sigma' in results[k] else sigArr[k]
                i+=2
            if not ptrnReff.match(contents[i]) is None: # Reff, field has unique form
                Reffs = np.array(ptrnReff.match(contents[i]).group(1).split(), dtype='float64')
                for k,Reff in enumerate(Reffs):
                    results[k]['rEff'] = Reff     
            self.parseMultiParamFld(contents, i, results, ptrnAOD, 'aod', 'lambda')
            self.parseMultiParamFld(contents, i, results, ptrnPSD, 'dVdlnr', 'r')
            self.parseMultiParamFld(contents, i, results, ptrnVol, 'vol')
            self.parseMultiParamFld(contents, i, results, ptrnSPH, 'sph')            
            self.parseMultiParamFld(contents, i, results, ptrnAODmode, 'aodMode')
            self.parseMultiParamFld(contents, i, results, ptrnSSA, 'ssa')
            self.parseMultiParamFld(contents, i, results, ptrnSSAmode, 'ssaMode')
            self.parseMultiParamFld(contents, i, results, ptrnRRI, 'n')
            self.parseMultiParamFld(contents, i, results, ptrnIRI, 'k')
            i+=1
        if not results:
            warnings.warn('No aerosol data found, returning empty dictionary...')
            return results
        if 'aodMode' in results[0]:
            Nwvlth = 1 if np.isscalar(results[0]['aod']) else results[0]['aod'].shape[0]
            nsd = int(results[0]['aodMode'].shape[0]/Nwvlth)
            for k in range(len(results)): # seperate aerosol modes 
                results[k]['r'] = results[k]['r'].reshape(nsd,-1)
                results[k]['dVdlnr'] = results[k]['dVdlnr'].reshape(nsd,-1)
                results[k]['aodMode'] = results[k]['aodMode'].reshape(nsd,-1)
                results[k]['ssaMode'] = results[k]['ssaMode'].reshape(nsd,-1)
        return results
    
    def parseOutSurface(self, contents):
        results = self.parseOutDateTime(contents)
        ptrnALB = re.compile('^[ ]*Wavelength \(um\),[ ]+Surface ALBEDO')
        ptrnBRDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BRDF parameters')
        ptrnBPDF = re.compile('^[ ]*Wavelength \(um\),[ ]+BPDF parameters')
        ptrnWater = re.compile('^[ ]*Wavelength \(um\),[ ]+Water surface parameters')
        i = 0
        while i<len(contents):
            self.parseMultiParamFld(contents, i, results, ptrnALB, 'albedo')
            self.parseMultiParamFld(contents, i, results, ptrnBRDF, 'brdf')
            self.parseMultiParamFld(contents, i, results, ptrnBPDF, 'bpdf')
            self.parseMultiParamFld(contents, i, results, ptrnWater, 'wtrSurf')            
            i+=1
        return results
    
    def parseOutFit(self, contents):
        results = self.parseOutDateTime(contents)
        ptrnFIT = re.compile('^[ ]*[\*]+[ ]*FITTING[ ]*[\*]+[ ]*$')
        ptrnPIX = re.compile('^[ ]*pixel[ ]*#[ ]*([0-9]+)[ ]*wavelength[ ]*#[ ]*([0-9]+)')
        numericLn = re.compile('^[ ]*[0-9]+')
        i = 0
        skipFlds = 4 # the 1st 4 fields aren't interesting
        FITfnd = False
        while i<len(contents):
            if not ptrnFIT.match(contents[i]) is None: # We found fitting data
                FITfnd = True
            pixMatch = ptrnPIX.match(contents[i]) if FITfnd else None
            if not pixMatch is None:  # We found a single pixel & wavelength group
                pixInd = int(pixMatch.group(1))-1
                wvlInd = int(pixMatch.group(2))-1
                while not re.search('^[ ]*#[ ]*sza[ ]*vis', contents[i+2]) is None: # loop over measurement types
                    flds = [s.replace('/','o') for s in contents[i+2].split()[skipFlds:]]
                    lastLine = i+3 
                    while (lastLine < len(contents)) and not (numericLn.match(contents[lastLine]) is None): 
                        lastLine+=1 # lastNumericInd+1
                    for ang,dataRow in enumerate(contents[i+3:lastLine]): # loop over angles
                        dArr = np.array(dataRow.split(), dtype='float64')[skipFlds:]
                        for j,fld in enumerate(flds):
                            if fld not in results[pixInd]: results[pixInd][fld] = np.array([]).reshape(0,1)
                            if results[pixInd][fld].shape[1] == wvlInd: # need another column
                                nanCol = np.full((results[pixInd][fld].shape[0],1),np.nan)
                                results[pixInd][fld] = np.block([results[pixInd][fld],nanCol])
                            if results[pixInd][fld].shape[0] == ang: # need another angle row
                                nanRow = np.full((1,results[pixInd][fld].shape[1]),np.nan)
                                results[pixInd][fld] = np.block([[results[pixInd][fld]],[nanRow]])
                            results[pixInd][fld][ang,wvlInd] = dArr[j]
                    i=min(lastLine-2, len(contents)-3)
            i+=1
        return results
    
    def parseMultiParamFld(self, contents, i, results, ptrn, fdlName, fldName0=False):
        if not ptrn.match(contents[i]) is None: # RRI by aersol size mode
            singNumeric = re.compile('^[ ]*[0-9]+[ ]*$')
            numericLn = re.compile('^[ ]*[0-9]+')
            lastLine = i+1
            while not numericLn.match(contents[lastLine]) is None: lastLine+=1
            Nparams = 0
            for dataRow in contents[i+1:lastLine]:
                if not singNumeric.match(dataRow):
                    dArr = np.array(dataRow.split(), dtype='float64')
                    for k in range(len(results)):
                        if fldName0:
                            results[k][fldName0] = np.append(results[k][fldName0], dArr[0]) if fldName0 in results[k] else dArr[0]
                        results[k][fdlName] = np.append(results[k][fdlName], dArr[k+1]) if fdlName in results[k] else dArr[k+1]
                else:
                    Nparams += 1
            if Nparams > 1:
                for k in range(len(results)): # seperate parameters from wavelengths
                    results[k][fdlName] = results[k][fdlName].reshape(Nparams,-1)
            i = lastLine - 1
        
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
        msrmnts = np.array(msrmnts)
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
    
         
                  
                  
     
     
     
     
     
     
     
                  