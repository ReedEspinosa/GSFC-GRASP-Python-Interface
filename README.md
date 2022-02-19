# GSFC-GRASP-Python-Interface
A set of Python 3 tools for interfacing with the GRASP retrieval code

**The core of the code is in ``runGRASP.py`` which contains several classes:**

``graspDB`` – This takes many ``pixel`` or ``graspRun`` objects and runs them all though a series of individual calls to GRASP. The class achieves two ends: (1) distribute large batches of pixels into chunks that are manageable from a memory perspective and (2) allow the individual calls to the GRASP binary to occur asynchronously in accordance with user specified parameters (e.g., max number of CPUs to use at a given time).  

``graspRun`` – This represents a collection of pixels, generally designed to be called in a single run. This class handles input/output to GRASP through the SDATA and output text files, respectively. 

``pixel`` – This is class holds most of the information that would be in a typical row (entry for a single pixel) of an SDATA file.

``graspYAML`` – This class has a variety of methods for changing settings in the YAML files.

**Note that this repository also includes several scripts for plotting the outputs of GRASP runs, including runs involving retrieval simulations of synthetic data performed by the ``simulation`` class.**
