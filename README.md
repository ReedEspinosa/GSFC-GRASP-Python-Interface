# GSFC-GRASP-Python-Interface
A set of Python 3 tools for interfacing with the GRASP retrieval code

**The core of the code is in ``runGRASP.py`` which contains several classes:**

``graspDB`` – This takes many ``pixel`` or ``graspRun`` objects and runs them all though a series of individual calls to GRASP. The class achieves two ends: (1) distribute large batches of pixels into chunks that are manageable from a memory perspective and (2) allow the individual calls to the GRASP binary to occur asynchronously in accordance with user specified parameters (e.g., max number of CPUs to use at a given time).  

``graspRun`` – This represents a collection of pixels, generally designed to be called in a single run. This class handles input/output to GRASP through the SDATA and output text files, respectively. The output data is stored in a list of ``rslt`` dicts, which are defined below.

``pixel`` – This is class holds most of the information that would be in a typical row (entry for a single pixel) of an SDATA file.

``graspYAML`` – This class has a variety of methods for changing settings in the YAML files.

## ``rslt`` dict keys and values
```
for key,val in rslt[0].items(): 
  print("%23s: %27s %11s" % (("rslt['%s']" % key), str(type(val)), (str(val.shape) if 'numpy' in str(type(val)) else '()')))
```
```
       rslt['datetime']: <class 'datetime.datetime'>          ()
      rslt['longitude']:     <class 'numpy.float64'>          ()
       rslt['latitude']:     <class 'numpy.float64'>          ()
        rslt['costVal']:             <class 'float'>          ()
              rslt['r']:     <class 'numpy.ndarray'>     (N_modes, N_radii)
         rslt['dVdlnr']:     <class 'numpy.ndarray'>     (N_modes, N_radii)
             rslt['rv']:     <class 'numpy.ndarray'>        (N_modes,)
          rslt['sigma']:     <class 'numpy.ndarray'>        (N_modes,)
           rslt['rEff']:     <class 'numpy.float64'>          ()
       rslt['rEffCalc']:     <class 'numpy.float64'>          ()
            rslt['vol']:     <class 'numpy.ndarray'>        (N_modes,)
            rslt['sph']:     <class 'numpy.ndarray'>        (N_modes,)
         rslt['height']:     <class 'numpy.ndarray'>        (N_modes,)
      rslt['heightStd']:     <class 'numpy.ndarray'>        (N_modes,)
         rslt['lambda']:     <class 'numpy.ndarray'>        (N_lambda,)
            rslt['aod']:     <class 'numpy.ndarray'>        (N_lambda,)
        rslt['aodMode']:     <class 'numpy.ndarray'>      (N_modes, N_lambda)
            rslt['ssa']:     <class 'numpy.ndarray'>        (N_lambda,)
        rslt['ssaMode']:     <class 'numpy.ndarray'>      (N_modes, N_lambda)
              rslt['n']:     <class 'numpy.ndarray'>      (N_modes, N_lambda)
              rslt['k']:     <class 'numpy.ndarray'>      (N_modes, N_lambda)
         rslt['albedo']:     <class 'numpy.ndarray'>        (N_lambda,)
        rslt['wtrSurf']:     <class 'numpy.ndarray'>      (3, N_lambda)
           rslt['brdf']:     <class 'numpy.ndarray'>      (3, N_lambda)
           rslt['bpdf']:     <class 'numpy.ndarray'>        (N_lambda,)
              rslt['g']:     <class 'numpy.ndarray'>        (N_lambda,)
          rslt['gMode']:     <class 'numpy.ndarray'>      (N_modes, N_lambda)
 rslt['LidarRatioMode']:     <class 'numpy.ndarray'>      (N_modes, N_lambda)
 rslt['LidarDepolMode']:     <class 'numpy.ndarray'>      (N_modes, N_lambda)
     rslt['LidarRatio']:     <class 'numpy.ndarray'>        (N_lambda,)
     rslt['LidarDepol']:     <class 'numpy.ndarray'>        (N_lambda,)
            rslt['sza']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
            rslt['vis']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
            rslt['fis']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
        rslt['sca_ang']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
         rslt['meas_I']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
          rslt['fit_I']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
         rslt['meas_Q']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
          rslt['fit_Q']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
         rslt['meas_U']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
          rslt['fit_U']:     <class 'numpy.ndarray'>     (N_views, N_lambda)
     rslt['RangeLidar']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
        rslt['meas_DP']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
         rslt['fit_DP']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
      rslt['meas_VExt']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
       rslt['fit_VExt']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
       rslt['meas_VBS']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
        rslt['fit_VBS']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
           rslt['βext']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
          rslt['range']:     <class 'numpy.ndarray'>     (N_layers, N_lambda)
          rslt['angle']:     <class 'numpy.ndarray'> (N_ang, N_modes, N_lambda)
            rslt['p11']:     <class 'numpy.ndarray'> (N_ang, N_modes, N_lambda)
            rslt['p12']:     <class 'numpy.ndarray'> (N_ang, N_modes, N_lambda)
            rslt['p22']:     <class 'numpy.ndarray'> (N_ang, N_modes, N_lambda)
            rslt['p33']:     <class 'numpy.ndarray'> (N_ang, N_modes, N_lambda)
            rslt['p34']:     <class 'numpy.ndarray'> (N_ang, N_modes, N_lambda)
            rslt['p44']:     <class 'numpy.ndarray'> (N_ang, N_modes, N_lambda)
 ```
 