# ----------------------------------------
# pseudo isothermic profile (pISO)
# this version of the code assumes that we take r_s from rho(r) and use it in V(r) calculation as x = r/r_s; we calculate M200 from paper derived profile 

import numpy
from scipy.optimize import curve_fit as fit
def pISO(radii, den, vel):
    initial_guess = [10**10, 0.01]
    param, covar = fit(rho_pISO, radii, den, p0 = initial_guess)
    #new list for r/r_s
    x = []
    for r in radii:
        x.append(r/param[1])
    param1, covar1 = fit(V_pISO, x, vel)
    return M_pISO(param1[0]*param[1], param[0], param[1]), param1[0]
    # param = [rho_s, r_s], param1 = [C200, V200]
    
def rho_pISO(r, rho_s,r_s): 
    # r200 is the radius inside of which the average halo (bg paper)
    return rho_s/(1+(r/r_s)**2)

def M_pISO(r, rho_s, r_s):
    # we define M200 and N200 as the mass and the number of particles within r200 (Maccio 2008)
    return 4*numpy.pi*rho_s*(r_s**3)*(r-numpy.arctan(r))**0.5
    
def V_pISO(r, C_200, V200):
     return V200*((1-numpy.arctan(r)/r)/(1-numpy.arctan(C_200)/C_200))**0.5