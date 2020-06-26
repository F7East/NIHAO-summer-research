import numpy
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare
import pynbody as pyn

class model:
    
    '''
    A model class stores stuff needed for my graphs
    
    **Input**
    
    *profile* a analysis.profile.Profile of already centered halo
    
    *name* for graphing and tracking purposes
    
    *h* hubble constant that is not H0
    
    *stellar_mass* total stellar mass of the halo
    
    *p_model* specify which model is used to fit the data
    
    
    **Stores**
    
    *radii* rbins
    
    *log_den* log10 density bins
    
    *den_errors* according to Poisson distribution
    
    *log_den_error* using error propagation
    
    *r_200*, *M_200*, *C_200*
    
    *params* parameters for fit
    
    *covar* covariance matrix for fit parameters
    
    **
    
    '''
    
    def __init__(self, profile, name, h  =  0.1, stellar_mass = 0.1, halo_mass = 0.1, pmodel = 'pISO'):

       
        # radii and density
        self.radii = profile['rbins']
        self.log_den = numpy.log10(profile['density'])
        
        #errors in density
        self.den_error = []
        self.log_den_error = []
        for i in range(len(self.log_den)):
            self.log_den_error.append(1./(numpy.sqrt(profile['n'][i])*numpy.log(10)))
            self.den_error.append(profile['density'][i]/(numpy.sqrt(profile['n'][i])))
        
        # 200's
        self.r_200 = self.radii[-1]
        self.M_200 = profile['mass_enc'][-1]
        self.C_200 = 0.
        
        #for parameters and covariance
        self.params = []
        self.covar  = []
        
        #name and model
        self.name = name
        self.pmodel = pmodel
        
        #stellar and halo mass, hubble
        
        self.stellar_mass = stellar_mass
        self.halo_mass = halo_mass
        self.h = h
        
        #Einasto coefficients
        if self.pmodel == 'Einasto_ae':
            m = numpy.log10(halo_mass * h /(10**12 * pyn.array.SimArray(1., units = 'Msol') ))
            ν = 10**(-0.11 + 0.146*m + 0.0138*m**2 + 0.00123*m**3)
            self.alpha_e = 0.0095*ν**2 + 0.155
            
        #DC14 coefficients
        if self.pmodel == 'DC14':
            X = numpy.log10(stellar_mass/halo_mass)
            self.alpha = 2.94 - numpy.log10(10**((X+2.33)*(-1.08)) + 10**((X+2.33)*2.29))
            self.beta  = 4.23 + 1.34*X + 0.26*X**2
            self.gamma = -0.06 + numpy.log10(10**((X+2.56)*(-0.68)) + 10**(X+2.56))
        
        
        #fitting with different number of parameters
        if self.pmodel == 'Einasto':
            self.initial_guess = [self.log_den[0], 0.001, 1]
            self.bounding = ([self.log_den[-1], 0, -numpy.inf] , numpy.inf)
        else:
            self.initial_guess = [self.log_den[0], 0.001]
            self.bounding = ([self.log_den[-1], 0] , numpy.inf)
        
        self.params, self.covar = fit(self.log_rho, self.radii, self.log_den, sigma = self.log_den_error, absolute_sigma =  True, p0  = self.initial_guess, bounds = self.bounding, maxfev = 600)
        
        self.C_200 = self.r_200 / self.params[1]
        
        #additional arguments for Einasto
        
        
        
    def log_rho(self, r, log_rho_s, r_s, *args):
        '''
        Function that is being fitted as a density profile that come from the paper
        
        **Input**
        
        *r* radii
        
        *log_rho_s* is log10 of rho_s which is easier to fit this way
        
        *r_s* 
        
        '''
        
        if self.pmodel == 'pISO':
            return log_rho_s - numpy.log10((1+(r/r_s)**2))
        if self.pmodel == 'Burket':
            return log_rho_s - numpy.log10((1+(r/r_s)**2)) - numpy.log10(1+(r/r_s))
        if self.pmodel == 'NFW':
            return log_rho_s - numpy.log10(1+(r/r_s))*2 - numpy.log10(r/r_s)
        if self.pmodel == 'Lucky13':
            return log_rho_s - numpy.log10(1+(r/r_s))**3
        if self.pmodel == 'Einasto_ae':
            return log_rho_s - 2/(numpy.log(10)*self.alpha_e)*((r/r_s)**self.alpha_e-1)
        if self.pmodel == 'Einasto':
            return log_rho_s - 2/(numpy.log(10)*args[0])*((r/r_s)**args[0])
        if self.pmodel == 'DC14':
            return log_rho_s - self.gamma*numpy.log10(r/r_s) - (self.beta - self.gamma)/self.alpha*numpy.log10(1+(r/r_s)**self.alpha)
        
    def output(self):
        '''
        Returns most needed things for futre plots
        '''
        
        return (self.M_200, self.C_200), self.params, self.covar
        
def models():
    '''
    This is a way of keeping all profile names here
    '''
    return ('pISO', 'Burket', 'NFW', 'Einasto', 'DC14', 'coreNFW', 'Lucky13', 'Einasto_ae')


class DM_Profile:
    
    """
    This is needed in case somethign does not work out
    """
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
#         self.H = float(pyn.analysis.cosmology.H(self.s))/1000
        self.C_200 = 0
        self.V_200 = 0
        self.den_error = []
        self.log_den_error = []
        for i in range(len(self.den)):
            self.log_den_error.append(1./(numpy.sqrt(profile['n'][i])*numpy.log(10)))
            self.den_error.append(profile['density'][i]/(numpy.sqrt(profile['n'][i])))
#         self.den_chisq = 0.0
#         self.vel_chisq = 0.0

    def fits_pISO(self):
        # fits rho_pISO and V_pISO with their parameters
        initial_guess = [self.den[0], 0.001]
        self.param, covar = fit(self.rho_pISO, self.radii, self.den, sigma = self.log_den_error, absolute_sigma =  True, p0  = initial_guess, bounds = ( [self.den[-1], 0] , numpy.inf), maxfev = 10000)
        self.C_200 = self.r_200 / self.param[1]
#         self.V_200 = 10*self.H*self.r_200
        return covar
#         self.param1, covar1 = fit(self.V_pISO, self.radii, self.vel, bounds = (0, numpy.inf))  # no need for this

#         return self.param #, self.param1 # for debugging purposes
    
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

#     def V_pISO(self, r):
# #         if self.param == []: raise CustomError('rho_pISO has not been fitted yet') # have to implement this later
# #         return 10*self.H*self.param[1]*C_200*((1-numpy.arctan((r/self.param[1]))/(r/self.param[1]))/(1-numpy.arctan(C_200)/C_200))**0.5 # worse fit
#         return self.V_200*((1-numpy.arctan((r/self.param[1]))/(r/self.param[1]))/(1-numpy.arctan(self.C_200)/self.C_200))**0.5
    