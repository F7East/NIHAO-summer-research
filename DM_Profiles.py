# ----------------------------------------
# pseudo isothermic profile (pISO)
# this version of the code assumes that we take r_s from rho(r) and use it in V(r) calculation as x = r/r_s; we calculate M200 from paper derived profile 

import numpy
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare
import pynbody as pyn

class DM_Profile:
    
    def __init__(self, radii, den, vel, snapshot, number):
        # takes array of radii of shells, density and velocity at each shell, and snapshot itself 
        self.radii = radii
        self.den = den
        self.vel = vel
        self.param = [] 
        self.param1 = []
        self.number = number
        # param gives [rho_s, r_s] and param1 gives [C200]
        self.s = snapshot
#         self.s.physical_units()
        self.H = float(pyn.analysis.cosmology.H(self.s))
#         self.den_chisq = 0.0
#         self.vel_chisq = 0.0
        self_zeros = []
        for i in range((len(self.radii)-1)):
            if self.den[i] == 0.0:
                self_zeros.append(i)
        self.den  =  numpy.delete(self.den, self_zeros)
        self.radii = numpy.delete(self.radii, self_zeros)
        self.vel = numpy.delete(self.vel, self_zeros)
        self.den = numpy.log10(self.den)
    
    def fits_pISO(self):
        # fits rho_pISO and V_pISO with their parameters
        initial_guess = [self.den[0], 10]
        self.param, covar = fit(self.rho_pISO, self.radii, self.den, p0  = initial_guess, bounds = ( 0 ,numpy.inf))
        self.param1, covar1 = fit(self.V_pISO, self.radii, self.vel, bounds = (0, numpy.inf))

        return self.param, self.param1 # for debugging purposes
    
#     def chisq_pISO(self):
#         self.fits_pISO()
#         self.den_chisq = chisquare(self.den, f_exp = self.rho_pISO(self.radii, *self.param) )
#         self.vel_chisq = chisquare(self.vel, f_exp = self.V_pISO(self.radii, *self.param1)  )
        
    def pISO(self):
        # returns enclosed mass accoring to paper profile, the concentration and parameter arrays
        self.fits_pISO()
        return self.M_pISO(self.param1[0]*self.param[1], self.param[0], self.param[1]), self.param1[0], self.param, self.param1

    def rho_pISO(self, r, log_rho_s,r_s): 
        # rho profile from the paper
        # r200 is the radius inside of which the average halo (bg paper)
        return log_rho_s/numpy.log10((1+(r/r_s)**2))

    def M_pISO(self, r, log_rho_s, r_s):
        # velocity profile from the paper
        # we define M200 and N200 as the mass and the number of particles within r200 (Maccio 2008)
        return 4*numpy.pi*(10**log_rho_s)*(r_s**3)*((r/r_s)-numpy.arctan((r/r_s)))**0.5

    def V_pISO(self, r, C_200, V_200):
#         if self.param == []: raise CustomError('rho_pISO has not been fitted yet') # have to implement this later
#         return 10*self.H*self.param[1]*C_200*((1-numpy.arctan((r/self.param[1]))/(r/self.param[1]))/(1-numpy.arctan(C_200)/C_200))**0.5 # worse fit
        return V_200*((1-numpy.arctan((r/self.param[1]))/(r/self.param[1]))/(1-numpy.arctan(C_200)/C_200))**0.5
    