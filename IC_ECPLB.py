#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:42:11 2023

@author: tga
"""


from matplotlib import pyplot as plt
import astropy.units as u
import naima
import numpy as np
from multiprocessing import Pool
import emcee
import corner
import os
import csv

IC_lpb_labels = ["log10(norm)", "alpha", "cutoff", "beta"]
pl = np.array([25, 1,   1, 0])
pu = np.array([35, 5, 500, 9])
nwalkers, ndim = 8, 4
nstep = 10000
nburn = nstep // 5
# par = np.array([29.4, 2.0, 173, 2])*(1 + 0.001*np.random.randn(nwalkers,ndim))
par = np.random.uniform(pl, pu, (nwalkers, ndim))

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


path = "/home/tga/Downloads/J2032/two_sources/ic/"
chainfile = "ic_ecplb.h5"
backend = emcee.backends.HDFBackend(path + chainfile)
backend.reset(nwalkers, ndim)

def ElectronIC(pars, data_energy):
    """
    Define particle distribution model, radiative model, and return model flux
    at data energy values
    """
    amplitude = 10**pars[0]
    alpha = pars[1]
    cutoff = pars[2]
    beta = pars[3]
    ECPL = naima.models.ExponentialCutoffPowerLaw(
        amplitude / u.eV, 30.0 * u.TeV, alpha, cutoff * u.TeV, beta
    )
    IC = naima.models.InverseCompton(ECPL, seed_photon_fields=[
        ["CMB", 2.73 * u.K, 4.2e-13 * u.erg / u.cm ** 3],
        ["FIR", 25.23 *u.K, 7.8e-13 * u.erg / u.cm ** 3],
        ["NIR", 500 * u.K,  3.0e-13 * u.erg / u.cm ** 3],
        ["VIS", 5034 * u.K, 6.7e-13 * u.erg / u.cm ** 3]
    ])
    SED = IC.sed(data_energy*u.TeV, distance=1.4 * u.kpc)
#    We = IC.compute_We(Eemin=1 * u.TeV)
    return SED


def log_likelihood(pars):
    model = ElectronIC(pars, data_energy).value
    likelihood = -0.5*np.sum((data_flux - model)**2/data_flux_err**2)
    return likelihood

def log_prior(pars):
    if np.all(pars>pl) and np.all(pars<pu):         
        return 0.0
    return -np.inf

def log_probability(pars):
    lp = log_prior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(pars)


with Pool(nwalkers) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,ndim,log_probability,args=(),pool=pool, backend=backend
    )
    sampler.run_mcmc(par, nstep,progress=True)

flat_samples = sampler.get_chain(discard=nburn, flat=True)
###  save data ###
fig = corner.corner(flat_samples, labels=IC_lpb_labels,quantiles=[0.16, 0.5, 0.84],\
                    show_titles=True)

