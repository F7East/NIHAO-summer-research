# ----------------------------------------
# pseudo isothermic profile (pISO)
# this version of the code assumes that we take r_s from rho(r) and use it in V(r) calculation as x = r/r_s; we calculate M200 from paper derived profile 

import numpy
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare
import pynbody as pyn

class DM_Profile:
    
    def __init__(self, profile, snapshot, number, r_200):
        # takes array of radii of shells, density and velocity at each shell, and snapshot itself 
        self.radii = profile['rbins']
        self.den = numpy.log10(profile['density'])
        self.vel = profile['v_circ']
        self.M_200 = profile['mass_enc'][-1]
        self.param = []
        self.r_200 = r_200
#         self.param1 = []
        self.number = number
        # param gives [rho_s, r_s] and param1 gives [C200]
        self.s = snapshot
#         self.s.physical_units()
        self.H = float(pyn.analysis.cosmology.H(self.s))/1000
        self.C_200 = 0
        self.V_200 = 0
        self.den_error = []
        self.log_den_error = []
        for i in range(len(self.den)):
            self.log_den_error.append(1./(numpy.sqrt(profile['n'][i])*numpy.log(10)))
            self.den_error.append(profile['density'][i]/(numpy.sqrt(profile['n'][i])))
#         self.den_chisq = 0.0
#         self.vel_chisq = 0.0
#         So zeroing was useless 
#         self_zeros = []
#         for i in range((len(self.radii))):
#             if self.den[i] == 0.0:
#                 self_zeros.append(i)
#         self.den  =  numpy.delete(self.den, self_zeros)
#         self.radii = numpy.delete(self.radii, self_zeros)
#         self.vel = numpy.delete(self.vel, self_zeros)
#         self.den = numpy.log10(self.den)
    
    def fits_pISO(self):
        # fits rho_pISO and V_pISO with their parameters
        initial_guess = [self.den[0], 0.001]
        self.param, covar = fit(self.rho_pISO, self.radii, self.den, sigma = self.log_den_error, p0  = initial_guess, bounds = ( [self.den[-1], 0] , numpy.inf), maxfev = 10000)
        self.C_200 = self.r_200 / self.param[1]
        self.V_200 = 10*self.H*self.r_200
        return covar
#         self.param1, covar1 = fit(self.V_pISO, self.radii, self.vel, bounds = (0, numpy.inf))  # no need for this

        return self.param #, self.param1 # for debugging purposes
    
#     def chisq_pISO(self):
#         self.fits_pISO()
#         self.den_chisq = chisquare(self.den, f_exp = self.rho_pISO(self.radii, *self.param) )
#         self.vel_chisq = chisquare(self.vel, f_exp = self.V_pISO(self.radii, *self.param1)  )
        
    def pISO(self):
        # returns enclosed mass accoring to paper profile, the concentration and parameter arrays
        covar = self.fits_pISO()
        return self.M_200, float(self.C_200), self.param, covar

    def rho_pISO(self, r, log_rho_s, r_s): 
        # rho profile from the paper
        # r200 is the radius inside of which the average halo (bg paper)
        return log_rho_s - numpy.log10((1+(r/r_s)**2))
#         return log_rho_s - numpy.log10((1+(r*(10**k))**2))

    def M_pISO(self, r, log_rho_s, r_s):
        # velocity profile from the paper
        # we define M200 and N200 as the mass and the number of particles within r200 (Maccio 2008)
        return 4*numpy.pi*(10**log_rho_s)*(r_s**3)*((r/r_s)-numpy.arctan((r/r_s)))**0.5

    def V_pISO(self, r):
#         if self.param == []: raise CustomError('rho_pISO has not been fitted yet') # have to implement this later
#         return 10*self.H*self.param[1]*C_200*((1-numpy.arctan((r/self.param[1]))/(r/self.param[1]))/(1-numpy.arctan(C_200)/C_200))**0.5 # worse fit
        return self.V_200*((1-numpy.arctan((r/self.param[1]))/(r/self.param[1]))/(1-numpy.arctan(self.C_200)/self.C_200))**0.5
    