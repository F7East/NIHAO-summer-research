# ----------------------------------------
# pseudo isothermic profile (pISO)
# this version of the code assumes that we take r_s from rho(r) and use it in V(r) calculation as x = r/r_s; we calculate M200 from paper derived profile 

import numpy
from scipy.optimize import curve_fit as fit
import pynbody as pyn

class DM_Profile:
    
    def __init__(self, radii, den, vel, snapshot):
        # takes array of radii of shells, density and velocity at each shell, and snapshot itself 
        self.radii = radii
        self.den = den
        self.vel = vel
        self.param = [] 
        self.param1 = []
        # param gives [rho_s, r_s] and param1 gives [C200]
        self.s = snapshot
        self.H = float(pyn.analysis.cosmology.H(self.s))
    
    def fits_pISO(self):
        # fits rho_pISO and V_pISO with their parameters
        initial_guess = [10**10, 0.01]
        self.param, covar = fit(self.rho_pISO, self.radii, self.den, p0 = initial_guess, bounds = (0,numpy.inf))
        self.param1, covar1 = fit(self.V_pISO, self.radii, self.vel, bounds = (0, numpy.inf))
        return self.param, self.param1 # for debugging purposes
    
    def pISO(self):
        # returns enclosed mass accoring to paper profile, the concentration and parameter arrays
        self.fits_pISO()
        return self.M_pISO(self.param1[0]*self.param[1], self.param[0], self.param[1]), self.param1[0], self.param, self.param1

    def rho_pISO(self, r, rho_s,r_s): 
        # rho profile from the paper
        # r200 is the radius inside of which the average halo (bg paper)
        return rho_s/(1+(r/r_s)**2)

    def M_pISO(self, r, rho_s, r_s):
        # velocity profile from the paper
        # we define M200 and N200 as the mass and the number of particles within r200 (Maccio 2008)
        return 4*numpy.pi*rho_s*(r_s**3)*(r-numpy.arctan(r))**0.5

    def V_pISO(self, r, C_200, r_s):
#         if self.param == []: raise CustomError('rho_pISO has not been fitted yet') # have to implement this later
#         return 10*H*self.param[1]*C_200*((1-numpy.arctan((r/self.param[1]))/(r/self.param[1]))/(1-numpy.arctan(C_200)/C_200))**0.5 # worse fit
        return 10*self.H*r_s*C_200*((1-numpy.arctan((r/r_s))/(r/r_s))/(1-numpy.arctan(C_200)/C_200))**0.5
    