#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:24:05 2023

@author: tga
"""

########## data ##############
import numpy as np
import emcee 
from multiprocessing import Pool
import corner 
from matplotlib import pyplot as plt
import h5py

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

######### MCMC settings ############
pl = [-15, -3,  10, 0]
pu = [-10,  3, 500, 5]
ndim = len(pl)
nwalkers = 2*ndim
nstep = 10000
# par = np.array([-12, -0.3, 60, 1.3])*(1 + 0.001*np.random.randn(nwalkers,ndim))
par = np.random.uniform(pl, pu, (nwalkers, ndim))

nburn = nstep // 5
labels = ["log10(norm)", "index", "cutoff", "beta"]
path= "/home/tga/Downloads/J2032/two_sources/photon/"
chainfile = "photon_ecplb1.h5"
backend = emcee.backends.HDFBackend(path + chainfile)
backend.reset(nwalkers, ndim)
########### MCMC functions ##############

def ECPLB(pars):
    amplitude = 10**pars[0]
    index = pars[1]
    cutoff = pars[2]
    beta = pars[3]
    return amplitude*data_energy**(-index)*np.exp(-(data_energy/cutoff)**beta)

##################################################
def log_likelihood(pars):
    model = ECPLB(pars)
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
    sampler.run_mcmc(par, nstep, progress=True)


flat_samples = sampler.get_chain(discard=nburn, flat=True)
fig = corner.corner(flat_samples, labels=labels,quantiles=[0.16, 0.5, 0.84],show_titles=True)
