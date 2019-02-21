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

def redshift_to(zi, zf, wi, fi, cos, adjust_f=True, in_Hz=False):
    """
    Redshift or de-redshift a spectrum
    zi = initial redshift (redshift of input data)
    zf = final redshift (redshift of output data)
    wi = initial wavelength vector in Angstroms (AA)
    fi = intial flux vector in units of s^-1 cm^-2 AA^-1 (or Hz^-1) [see in_Hz]
    cos = astropy.cosmology object
    adjust_f = whether or not to modify the flux by the appropriate (1+z) factors
    in_Hz = whether or not the input spectrum is in units of Hz^-1 (vs. AA^-1)
    Outputs flux density in flux s^-1 cm^-2 AA^-1
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
            D_Lzi = 1. / (2*m.sqrt(np.pi)) * u.cm
        else:
            D_Lzi = cos.luminosity_distance(zi).to('cm') # luminosity distance
        if zf == 0:
            D_Lzf = 1. / (2*m.sqrt(np.pi)) * u.cm
        else:
            D_Lzf = cos.luminosity_distance(zf).to('cm')
        ff = np.array(fi, dtype=np.float64) * (D_Lzi/D_Lzf)**2 * (1 + zi)/(1 + zf)
        ff = ff.value
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
    import sncosmo
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
    # RuntimeWarning for f<=0, but preserves masking and NaNs
    # np.ma.log10 suppresses RuntimeWarning but masks NaNs
    m = -2.5 * np.log10(f) + ZP
    return m

def ABmagerr_to_fluxerr(merr, f):
    """
    Convert AB magnitude errors to flux errors
    """
    ferr = (m.log(10)/2.5) * merr * f
    return np.fabs(ferr)

def fluxerr_to_ABmagerr(ferr, f):
    """
    Convert flux errors to AB magnitude errors
    """
    merr = (ferr/f) * (2.5/m.log(10))
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
    a = np.ma.array([i for i in r.as_void()])
    return a

def rms(x):
    """ 
    Compute root-mean-square of x
    """
    rms = np.sqrt(np.nansum(np.square(x))/np.float(np.sum(~np.isnan(x))))
    return rms

def nmad(x):
    """
    Compute normalized median absolute deviation of x
    """
    k = 1.4826 # normalization
    m = np.nanmedian(x)
    nmad = k * np.nanmedian(np.absolute(x - m))
    return nmad

def b_n(n):
    """
    Compute approximation of b_n (constant in Sersic profile).
    For 1 <~ n < 10, use Ciotti & Bertin (1999) approximation which has a 
    relative error < 10^-6. For n <= 0.36, use MacArthur, Courteau, & Holtzman 
    (2003) which is accurate to 0.002.
    """
    if n <= 0.36: # MCH03
        ei  = np.array([0, 1, 2, 3, 4])
        ai  = np.array([0.01945, -0.8902, 10.95, -19.67, 13.43])
    else: # CB99
        ei  = np.array([1, 0, -1, -2])
        ai  = np.array([2, -1./3, 4./405, 46./25515])
    return np.sum(ai * np.power(float(n), ei))

def RKron_from_Sersic(R, Re, n):
    """
    Compute the Kron radius given Sersic parameters using Eq. 32 in 
    Graham & Driver (2005). 
    R  = radius to integrate out to
    Re = effective radius (in Sersic profile definition)
    n  = Sersic index
    Kron magnitudes (SExtractor mag_auto) use 2.5*RKron.
    Output units are same units as input R, Re
    """
    from scipy.special import gammainc, gamma
    b = b_n(n)
    x = b * np.power(float(R)/Re, 1./n)
    norm = gamma(3*n) / gamma(2*n) # scipy gammainc has 1/Gamma(a) prefactor
    R_K = (Re/b**n) * gammainc(3*n, x)/gammainc(2*n, x) * norm 
    return R_K

def chauvenet(x):
    """
    Use Chauvenet's criterion to determine which elements of array x are 
    outliers and output a mask where for each element, T=keep and F=reject.
    NOTE: This method assumes that x is drawn from a Gaussian distribution.
    This function has been tested using examples from Ch. 6 of John R. Taylor's 
    book Error Analysis, 2nd edition
    """
    from scipy.stats import norm
    x = np.array(x)
    n = len(x)
    mean = np.mean(x)
    sigma = np.std(x, ddof=1) # sample standard deviation (divide by N-1)
    dev = np.abs(x - mean) / sigma # normalized deviation
    crit = 1. / (2*n) # Chauvenet's criterion
    prob = 2 * norm.sf(dev) # probability of obtaining deviations > dev
    mask = prob >= crit # reject if prob is less than criterion
    return mask
