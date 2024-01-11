#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:26:32 2024

@author: tga
"""

import astropy.units as u
import numpy as np

from astropy import constants as const
from matplotlib import pyplot as plt


def ic_bE(Ee, photon_field):
    """
    ### This function calculates the energy loss rate of inverse Compton mechanism.
    >>> Ee: energy of electrons (unit: eV) while 
    >>> photon_field: energy density (erg cm**-3) and temperature (K) of scattered photons.
    ### return energy loss rate(erg s**-1) 
    """
    U_ph, T_ph = photon_field            
### define some constants
    sigma_T = const.sigma_T.cgs.value
    c = const.c.cgs.value
    ev2k = 1*u.eV.to(u.J) / const.k_B.value
    a1 = 2.82 / (0.511e6)**2 / ev2k
    a2 = 4/3 *sigma_T *c / (0.511e6)**2
    bE = a2*Ee**2*U_ph / (1 + (a1*T_ph*Ee)**0.6)**(1.9/0.6) # erg s**-1
    return bE

def syn_bE(Ee, B):
    ### This function calculates the energy loss rate of synchrotron mechanism.
    >>> Ee: energy of electrons (unit: eV) while 
    >>> B: magnetic field strength
    ### return energy loss rate(erg s**-1) 
    sigma_T = const.sigma_T.cgs.value
    c = const.c.cgs.value
    a2 = 4/3 *sigma_T *c / (0.511e6)**2
    bE = a2*Ee**2*B**2/8/np.pi # erg s**-1
    return bE

