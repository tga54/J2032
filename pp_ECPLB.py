#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:07:08 2023

@author: tga
"""

from matplotlib import pyplot as plt
import astropy.units as u
import naima
import numpy as np
from multiprocessing import Pool
import emcee
import corner
from astropy import constants as const
from scipy import special
import os
import csv

PionDecay_ECPL_labels = ["log10(norm)", "index", "cutoff", "beta"]

# pl = np.array([30, 1])
# pu = np.array([40, 4])
pl = np.array([40, 0, 100, 0])
pu = np.array([50, 5, 900, 20])

nwalkers, ndim = 8, 4
nstep = 10000
nburn = nstep // 5

par = np.random.uniform(pl, pu, (nwalkers, ndim))
# par = np.array([44, 1, 400, 10])*(1 + 0.001*np.random.randn(nwalkers,ndim))
# par = np.array([45, 2.1, 414])*(1 + 0.001*np.random.randn(nwalkers,ndim))

# def PionDecay_PL(pars):
#     amplitude = 10 ** pars[0] / u.TeV
#     alpha = pars[1]
#     PL = naima.models.PowerLaw(
#         amplitude, 30 * u.TeV, alpha
#     )
#     PP = naima.models.PionDecay(PL, nh=1e8 * u.cm ** -3)

#     model = PP.sed(data_energy*u.TeV, distance=1.4 * u.kpc)
#     return model

def PionDecay_ECPL(pars):
    amplitude = 10 ** pars[0] / u.TeV
    alpha = pars[1]
    e_cutoff = pars[2] * u.TeV
    beta = pars[3]
    ECPL = naima.models.ExponentialCutoffPowerLaw(
        amplitude, 30 * u.TeV, alpha, e_cutoff, beta
    )
    PP = naima.models.PionDecay(ECPL, nh=1.0 * u.cm ** -3)

    model = PP.sed(data_energy*u.TeV, distance=1.4 * u.kpc)
    return model

path = "/home/tga/Downloads/J2032/two_sources/pp/"
# chainfile = "pp_ecplb_veritas.h5"
chainfile = "pp_ecplb1.h5"
backend = emcee.backends.HDFBackend(path + chainfile)
backend.reset(nwalkers, ndim)

def log_likelihood(theta):
    model = PionDecay_ECPL(theta).value
#    model = ElectronIC(theta, photon_energy).value
    likelihood = -0.5*np.sum((data_flux - model)**2/data_flux_err**2)
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

# E = np.array([0.25, 0.4, 0.63, 1, 1.6, 2.5])
# itst = np.array([3.41e-12, 3.29e-12, 2.05e-12, 1.98e-12, 1.08e-12, 7.89e-13])
# itst_ul = np.array([4.57e-12, 3.87e-12, 2.36e-12, 2.4e-12, 1.4e-12, 1.07e-12])
# itst_ll = np.array([2.3e-12, 2.76e-12, 1.61e-12, 1.6e-12, 7.8e-13, 5.10e-13])
# E1 = np.array([4, 6.3])
# itst1 = np.array([9.8e-13, 1.2e-12])
# itst_err = [itst_ul - itst, itst - itst_ll]
# data_energy = E 
# data_flux = itst*1.6
# data_flux_err = (itst_err[0] + itst_err[1])/2

E_2032 = np.array([5.22803, 5.9491, 7.71755, 12.0335, 22.2169, 32.6012, 49.1058, \
                   74.7376, 113.262, 170.812])
flux_2032 = np.array([2.10871e-14, 1.43858e-14, 7.7256e-15, 4.50327e-15, 1.25287e-15, \
                      5.68982e-16, 2.07714e-16, 5.71098e-17, 1.10936e-17, 8.67472e-19])
flux_err_2032 = np.array([6.29059e-15, 3.33203e-15, 1.24504e-15, 4.87993e-16, 1.40316e-16,\
                          3.76089e-17, 1.45144e-17, 4.91621e-18, 1.73155e-18, 5.15633e-19])
E_ul_J2032 = np.array([ 253.192,])
flux_ul_J2032 = np.array([ 4.39709e-19, ])
itst_ul_J2032 = flux_ul_J2032*E_ul_J2032**2*1.6
data_energy = E_2032
data_flux = flux_2032*E_2032**2*1.6
data_flux_err = flux_err_2032*E_2032**2*1.6


with Pool(nwalkers) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,ndim,log_probability,args=(),pool=pool, backend=backend
    )
    sampler.run_mcmc(par, nstep,progress=True)

flat_samples = sampler.get_chain(discard=nburn, flat=True)
###  save data ###
fig = corner.corner(flat_samples, labels=PionDecay_ECPL_labels,quantiles=[0.16, 0.5, 0.84],show_titles=True)
fit_par = np.percentile(flat_samples,50, axis=0)



