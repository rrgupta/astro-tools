# astro-tools

_Some useful code for astro applications_

For example:
- redshift or de-redshift a spectrum given an initial and final redshift, assuming some cosmology
- convert from AB magnitude to flux density and vice versa
- get the redshift given the age of the universe and some cosmology
- integrate a spectrum through a (sncosmo-registered) bandpass to obtain an AB magnitude

_Also contains some basic statistical routines useful for data analysis_

For example:
- addition in quadrature
- true root-mean-square (RMS), distinct from standard deviation
- normalized median absolute deviation (NMAD)
- Chauvenet's criterion for outlier rejection

Python
------

**Requirements**

The following Python packages are needed:
- numpy
- astropy
- scipy
- sncosmo (https://github.com/sncosmo/sncosmo)

