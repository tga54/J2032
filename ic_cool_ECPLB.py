#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:46:20 2023

@author: tga
"""

#from matplotlib import pyplot as plt
import astropy.units as u
import naima
import numpy as np
from multiprocessing import Pool
import emcee
import corner
from astropy import constants as const
import os
import csv
from matplotlib import pyplot as plt
############### particle and photon energy ############### 
electron_energy = np.logspace(0,3,100)*u.TeV # TeV
photon_energy = np.logspace(0,3,100)*u.TeV # TeV

############### constants #############################
sigma_T = const.sigma_T.cgs.value
m_e = const.m_e.cgs.value
c = const.c.cgs.value
ev2k = 11600
a1 = 2.82 / (0.511e6)**2 / ev2k
a2 = -4/3 *sigma_T *c / (0.511e6)**2
yr2s = 1*u.yr.to(u.s)

################# photon field ##################
U_cmb, T_cmb = [4.2e-13, 2.73  ]  # erg cm**-3, eV
U_fir, T_fir = [7.8e-13, 25.23 ]  # erg cm**-3, eV
U_nir, T_nir = [3.0e-13, 500   ]  # erg cm**-3, eV
U_vis, T_vis = [6.7e-13, 5034  ]  # erg cm**-3, eV

#################### MCMC settings ##############
labels = ["log10(norm)", "alpha", "cutoff","beta", "B", "Age" ]
# labels = ["log10(norm)", "alpha", "cutoff", "B", "Age" ]

pl = np.array([25, 1, 1, 0,  1, 0])
pu = np.array([35, 5, 4, 5, 10, 6])


ndim = len(pl)
nwalkers = 2 *ndim
nstep = 5000
nburn = nstep // 5
# par = np.array([30.15, 2.7, 2.3, 22, 10])*(1 + 0.001*np.random.randn(nwalkers,ndim))

par = np.array([29.4, 1.5, 2.2, 1.3, 3.6, 3.3])*(1 + 0.001*np.random.randn(nwalkers,ndim))
# par = np.random.uniform(pl,pu,(nwalkers,ndim))

################### MCMC functions ###################

path= "/home/tga/Downloads/J2032/two_sources/ic_cool/"
chainfile = "ic_ecplb_cool.h5"
backend = emcee.backends.HDFBackend(path + chainfile)
backend.reset(nwalkers, ndim)

def ElectronIC(pars):
    amplitude = 10**pars[0]
    alpha = pars[1]
    cutoff = 10**pars[2]
    beta = pars[3]
    B = pars[4]*1e-6
    U_B = B**2 / 8 / np.pi  
    age = 10**pars[5]  # yr
    ECPL = naima.models.ExponentialCutoffPowerLaw(
        amplitude / u.eV, 30.0 * u.TeV, alpha, cutoff * u.TeV, beta
    )
    IC = naima.models.InverseCompton(ECPL)
    Ed = IC.particle_distribution(electron_energy)
    Ee = electron_energy.to(u.eV).value
    dEe = np.diff(Ee)

    NumT = 1000
    dt = age / NumT * yr2s
    
    for i in range(NumT):
        U_ph = U_cmb / (1 + (a1*T_cmb*Ee)**0.6)**(1.9/0.6) \
             + U_fir / (1 + (a1*T_fir*Ee)**0.6)**(1.9/0.6) \
             + U_nir / (1 + (a1*T_nir*Ee)**0.6)**(1.9/0.6) \
             + U_vis / (1 + (a1*T_vis*Ee)**0.6)**(1.9/0.6) 
        bE = a2*Ee**2*(U_B + U_ph)
        Ee = Ee + dt*bE*1e12/1.6
        
    Ee1 = Ee
    dEe1 = np.diff(Ee1)
    Ed1 = Ed[:-1]*dEe/dEe1
    E_cool = naima.models.TableModel(Ee1[:-1]*u.eV, Ed1)

    IC1 = naima.models.InverseCompton(E_cool, seed_photon_fields=[
        ["CMB", T_cmb * u.K, U_cmb * u.erg / u.cm ** 3],
        ["FIR", T_fir * u.K, U_fir * u.erg / u.cm ** 3],
        ["NIR", T_nir * u.K, U_nir * u.erg / u.cm ** 3],
        ["VIS", T_vis * u.K, U_vis * u.erg / u.cm ** 3]
    ])
    SED = IC1.sed(data_energy*u.TeV, distance=1.4 * u.kpc)
#    We = IC.compute_We(Eemin=1 * u.TeV)
    return SED



def log_likelihood(theta):
#    model = PionDecay_ECPL(theta, photon_energy).value
    model = ElectronIC(theta).value
    likelihood = -0.5*np.sum((data_flux - model)**2/data_flux_err**2)
    if np.isnan(likelihood):
        likelihood = -100
    return likelihood

def log_prior(theta):
    if np.all(theta>pl) and np.all(theta<pu):         
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

### J2032 data ### 
E_2032 = np.array([5.22803, 5.9491, 7.71755, 12.0335, 22.2169, 32.6012, 49.1058, \
                   74.7376, 113.262, 170.812])
flux_2032 = np.array([2.10871e-14, 1.43858e-14, 7.7256e-15, 4.50327e-15, 1.25287e-15, \
                      5.68982e-16, 2.07714e-16, 5.71098e-17, 1.10936e-17, 8.67472e-19])
flux_err_2032 = np.array([6.29059e-15, 3.33203e-15, 1.24504e-15, 4.87993e-16, 1.40316e-16,\
                          3.76089e-17, 1.45144e-17, 4.91621e-18, 1.73155e-18, 5.15633e-19])
data_energy = E_2032
data_flux = flux_2032*E_2032**2*1.6
data_flux_err = flux_err_2032*E_2032**2*1.6
E_ul_J2032 = np.array([ 253.192,])
flux_ul_J2032 = np.array([ 4.39709e-19, ])
itst_ul_J2032 = flux_ul_J2032*E_ul_J2032**2*1.6


with Pool(nwalkers) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,ndim,log_probability,args=(),pool=pool, backend=backend
    )
    sampler.run_mcmc(par, nstep,progress=True)

flat_samples = sampler.get_chain(discard=nburn, flat=True)
###  save data ###
fig = corner.corner(flat_samples, labels=labels,quantiles=[0.16, 0.5, 0.84],\
                    show_titles=True)
fit_par = np.percentile(flat_samples,50, axis=0)

