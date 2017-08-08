#!/usr/bin/env python
"""
Useful functions such as the ability to take a spectrum at some redshift, 
shift it to some different redshift and output the new wavelength and flux.
"""
import math as m
import numpy as np
import astropy.cosmology as cosmo
import astropy.units as u
import astropy.constants as const
from scipy import interpolate
import sncosmo

def redshift_to(zi, zf, wi, fi, cos, adjust_f=True, in_Hz=False):
    """
    Redshift or de-redshift a spectrum
    zi = initial redshift (redshift of input data)
    zf = final redshift (redshift of output data)
    wi = initial wavelength vector (Angstroms)
    fi = intial flux vector (Angstroms)
    cos = astropy.cosmology object
    adjust_f = whether or not to modify the flux by the appropriate (1+z) factors
    in_Hz = whether or not the input spectrum is in units of Hz^-1 (vs. AA^-1)
    Outputs flux density in flux per AA
    """
    # First convert wavelengths
    wf = wi * (1 + zf)/(1 + zi)
    # Check if flux density is per Hz. If so, convert to per AA
    if in_Hz:
        c = const.c.to('AA/s').value
        fi = (c/wi**2) * fi
    if not adjust_f:
        ff = fi # do not rescale flux (for cases where this is irrelevant)
    else: # otherwise, rescale flux
        # Check z=0 cases
        if zi == 0:
            D_Lzi = 1./(2*m.sqrt(np.pi))*u.cm
        else:
            D_Lzi = cos.luminosity_distance(zi).to('cm') # luminosity distance
        if zf == 0:
            D_Lzf = 1./(2*m.sqrt(np.pi))*u.cm
        else:
            D_Lzf = cos.luminosity_distance(zf).to('cm')
        ff = np.array(fi, dtype=np.float64) * (D_Lzi/D_Lzf)**2 * (1 + zi)/(1 + zf)
    return wf, ff

def get_z_at_age(A, cos, zmax=2):
    """
    Get the redshift at a given age of the universe, assuming some cosmology, 
    by linearly interpolating the age(z) relation
    A = age at which to get redshift
    cos = astropy.cosmology object
    """
    z = np.linspace(0, zmax, num=10000)
    age = cos.age(z).value # age of universe in Gyr
    f = interpolate.interp1d(age, z)
    return f(A)


def integrate_ABmag(wave, flux, bandname, funit=(u.erg / (u.s * u.cm**2 * u.AA))):
    """
    Compute AB magnitude given a spectrum (wave, flux) and a transmission 
    filter (band) using sncosmo utilities
    wave = wavelength (Angstroms)
    flux = flux (default in erg/s/cm^2/Ang)
    bandname = sncosmo band name (string)
    """
    spectrum = sncosmo.Spectrum(wave, flux, unit=funit)
    band = sncosmo.get_bandpass(bandname)
    f = spectrum.bandflux(band)
    ab = sncosmo.get_magsystem('ab')
    m = ab.band_flux_to_mag(f, band)
    return m

def ABmag_to_flux(m, ZP=-48.6):
    """
    Convert AB magnitude to flux density, f_nu
    assuming ZP = -48.6, unless specified otherwise
    """
    f_nu = np.ma.power(10, (m - ZP)/-2.5)
    return f_nu

def flux_to_ABmag(f, ZP=-48.6):
    """
    Convert flux density to AB magnitude 
    assuming ZP = -48.6, unless specified otherwise
    """
    m = -2.5*np.ma.log10(f) + ZP # zeropoint from data files
    return m

def ABmagerr_to_fluxerr(merr, f):
    """
    Convert AB magnitude errors to flux errors
    """
    ferr = (m.log(10)/2.5)*merr*f
    return np.fabs(ferr)

def fluxerr_to_ABmagerr(ferr, f):
    """
    Convert flux errors to AB magnitude errors
    """
    merr = (ferr/f)*(2.5/m.log(10))
    return np.fabs(merr)
    
def add_quad(a, b):
    """
    Add 2 quantities (a and b) in quadrature
    """
    s = np.sqrt(np.square(a) + np.square(b))
    return s

def row_to_array(r):
    """
    Convert astropy row to masked numpy array
    """
    #a = np.array(r.as_void().tolist())
    a = np.ma.array([i for i in r.as_void()])
    return a

def rms(x):
    """ 
    Compute root-mean-square of an array
    """
    rms = np.sqrt(np.nansum(np.square(x))/np.float(np.sum(~np.isnan(x))))
    return rms

def nmad(x):
    """
    Compute normalized median absolute deviation
    """
    k = 1.4826 # normalization
    m = np.nanmedian(x)
    nmad = k * np.nanmedian(np.absolute(x - m))
    return nmad

