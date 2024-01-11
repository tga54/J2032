import numpy as np
from astropy import units as u
from astropy import constants as const

def GammaGammaAbsorption(frequency_ph, E_abs, n_abs, R):
    """
    Gamma-gamma absorption due to the soft photon.
    :param frequency_ph: radiative photon frequency [Hz]
    :param E_abs: soft photon energy [eV]
    :param n_abs: soft photon number density distribution [eV-1 cm-3]
    :param R: size of the region [cm]
    :return: optical depth
    """
    pi = np.pi
    R = R.to(u.cm).value
    E_ph = (const.h.cgs*frequency_ph).to(u.eV).value
    E_abs = E_abs.to(u.eV).value
    n_abs = n_abs.to(u.eV**-1 * u.cm**-3).value
    
    epsilon_ph = E_ph / (const.m_e.cgs * const.c.cgs**2).to(u.eV).value
    epsilon_abs = E_abs / (const.m_e.cgs * const.c.cgs**2).to(u.eV).value
    n_abs_epsilon = n_abs * (const.m_e.cgs * const.c.cgs**2).to(u.eV).value

    r_e = (2.8179e-13*u.cm).value
    dtau = (np.zeros(len(epsilon_ph)) * u.cm**-1).value
    for i in range(len(epsilon_ph)):
        temp = 0
        for j in range(len(epsilon_abs)-1):
            if epsilon_abs[j] < 1/epsilon_ph[i]:
                continue
            else:
                s0 = epsilon_abs[j]*epsilon_ph[i]
                phibar = np.select([s0 >= 3, s0 >= 0.1, s0 < 0.1],
                                    [2 * s0 * (np.log(4 * s0) - 2) + np.log(4 * s0) * (np.log(4 * s0) - 2) - (
                                                pi ** 2 - 9) / 3 + (np.log(4 * s0) + 8 / 9) / s0,
                                     2.2151050 - 5.6254697 * s0 + 4.2940799 * s0 ** 2 - 0.98922329 * s0 ** 3 + 0.095506958 * s0 ** 4,
                                     4 / 3 * (s0 - 1) ** (3 / 2) + 6 / 5 * (s0 - 1) ** (5 / 2) - 253 / 70 * (
                                                 s0 - 1) ** (7 / 2)])
                temp += n_abs_epsilon[j] / epsilon_abs[j]**2 * phibar * (epsilon_abs[j+1] - epsilon_abs[j])
        dtau[i] =  pi * r_e**2 / epsilon_ph[i]**2 * temp
    tau = dtau * R
    return tau
