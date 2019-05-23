#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:26:35 2019

@author: wrespino
"""
import yaml
import numpy as np

pathYAML = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/YAML_settingsFiles/settings_HARP_16bin_2lambda.yml'
pathYAMLout = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/YAML_settingsFiles/settings_HARP_16bin_1lambdaTEST.yml'
lambdaTypes = ['surface_water_CxMnk_iso_noPol', 
               'surface_land_brdf_ross_li', 
               'surface_land_polarized_maignan_breon', 
               'real_part_of_refractive_index_spectral_dependent',
               'imaginary_part_of_refractive_index_spectral_dependent']
indKeep = slice(0,1) # (start(0->1st), stop+1; ex. (0,2) takes 1st and 2nd), index_of_wavelength_involved's should have all wavelenghts

iowi = np.r_[1:(indKeep.stop-indKeep.start+1)].tolist()
with open(pathYAML, 'r') as stream:
    dl = yaml.load(stream)
for char in dl["retrieval"]["constraints"]:
    if dl["retrieval"]["constraints"][char]["type"] in lambdaTypes:
        for mod in dl["retrieval"]["constraints"][char]:
            if 'mode' in mod:
                dl["retrieval"]["constraints"][char][mod]["initial_guess"]["value"] = \
                    dl["retrieval"]["constraints"][char][mod]["initial_guess"]["value"][indKeep]
                dl["retrieval"]["constraints"][char][mod]["initial_guess"]["min"] = \
                    dl["retrieval"]["constraints"][char][mod]["initial_guess"]["min"][indKeep]
                dl["retrieval"]["constraints"][char][mod]["initial_guess"]["max"] = \
                    dl["retrieval"]["constraints"][char][mod]["initial_guess"]["max"][indKeep]
                dl["retrieval"]["constraints"][char][mod]["initial_guess"]["index_of_wavelength_involved"] = np.array(iowi).tolist() # prevents yaml.dump from regonizing repeat and using aliases (which GRASP does not support)
for noise in dl["retrieval"]["noises"]:
    for ms in dl["retrieval"]["noises"][noise]:
        if 'measurement_type' in ms:
            lmbs = dl["retrieval"]["noises"][noise][ms]["index_of_wavelength_involved"]
            lmbs = np.intersect1d(lmbs,iowi)
            dl["retrieval"]["noises"][noise][ms]["index_of_wavelength_involved"] = lmbs.tolist()
    
with open(pathYAMLout, 'w') as outfile:
    yaml.dump(dl, outfile, default_flow_style=None, indent=4, width=1000)