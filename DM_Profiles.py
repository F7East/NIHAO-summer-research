import numpy
from scipy.optimize import curve_fit as fit
from scipy.stats import chisquare
from pynbody import units
import pynbody as pyn
from numpy import log10 as lg

def model_prep(halo, l_r200, p_r200):
    
    """
    This prepares profile for model to take in. It is run separately because this is a heavy function
    
    **Input**
    
    *halo* is just halo object
    
    *l_r200* minimum distance from profiling, if > 1 is in units of lenght  if <1 is portion of r_200
    
    **Output**
    
    *profile* which includes
    
    *sm* total stellar mass
    
    *shm_radius* stellar half-mass radius
    
    *r_200* virial radius over 200 times the critical density of snapshot
    
    *t_sf* run time
    
    *M_200
    
    """
    
    # centering to generate variables and placing particles back 
    with pyn.analysis.angmom.faceon(halo, cen_size  =  '1 kpc'):
        
        r_200 = float(pyn.analysis.halo.virial_radius(halo.d, overden = 200, rho_def =  'critical'))
        
        # zero-min proof
        eps = 1.5*(halo['eps'][0])
        
        if l_r200 > 1.0:
            minimum = l_r200
            
        elif l_r200 >= 0.0:
            minimum = l_r200 * r_200
            
        else:
            minimum = 0.01 * r_200
          
        if minimum < eps:
            minimum = 1.5* eps
         
        # a choice between kpc or portion of r200
        if p_r200 > 1.0:
            maximum = p_r200
        elif p_r200 > 0.0:
            maximum = r_200 * p_r200
        else:
            maximum = 0.2*r_200
            
            
        profile = pyn.analysis.profile.Profile(halo.d, min = minimum, max = maximum, ndim = 3, type = 'log', nbins = 50)
        stellar_profile = pyn.analysis.profile.Profile(halo.s, min = 0.01, max = r_200, ndim = 2, type = 'equaln', nbins = 10000)
        shm_radius = stellar_profile['rbins'][len(stellar_profile['rbins'])//2]

    # calculating steallar and halo mass
    
    sm = halo.s['mass'].sum()
    t_sf = halo.properties['time'].in_units('Gyr')
    
    profile =  { 'rbins':profile['rbins'], 'density':profile['density'], 'n':profile['n'], 'M_200': pyn.array.SimArray(profile['mass_enc'][-1], units = profile['mass'].units), 'stellar_mass': sm, 'shm_radius':shm_radius, 'r_200':r_200, 'h': pyn.analysis.cosmology.H(halo).in_units(100*  units.km * units.s**-1 * units.Mpc**-1), 't_sf':t_sf}
        
    return profile

class model:
    
    '''
    A model class stores stuff needed for my graphs
    
    **Input**
    
    *see model_prep output
    
    
    **Stores**
    
    *radii* rbins
    
    *log_den* log10 density bins
    
    *den_errors* according to Poisson distribution
    
    *log_den_error* using error propagation
    
    *halo_mass*, *stellar_mass*
    
    *r_200*, *M_200*, *C_200*
    
    *params* parameters for fit
    
    *covar* covariance matrix for fit parameters
    
    **
    
    '''
    
    def __init__(self, profile, pmodel = 'pISO'):

       
        # radii and density
        self.radii = profile['rbins']
        self.log_den = lg(profile['density'])
        
        #errors in density
        self.den_error = []
        self.log_den_error = []
        for i in range(len(self.log_den)):
            self.log_den_error.append(1./(numpy.sqrt(profile['n'][i])*numpy.log(10)))
            self.den_error.append(profile['density'][i]/(numpy.sqrt(profile['n'][i])))
        
        # 200's
        self.r_200 = profile['r_200']
        self.M_200 = profile['M_200']
        self.C_200 = 0.
        
        #for parameters and covariance
        self.params = []
        self.covar  = []
        
        #model
        self.pmodel = pmodel
        
        #stellar and halo mass, hubble
        
        self.stellar_mass = profile['stellar_mass']
        
        #stellar half-mass radius
        self.shm_radius = profile['shm_radius']
        
        #Einasto coefficients
        if self.pmodel == 'Einasto_ae':
            HM = float(self.M_200.in_units(units.Msol*10**12/profile['h']))
            m = lg(HM)
            v = 10**(-0.11 + 0.146*m + 0.0138*m**2 + 0.00123*m**3)
            self.alpha_e = (0.0095*v**2 + 0.155)
            
        #DC14 coefficients
        if self.pmodel == 'DC14':
            X = lg(self.stellar_mass/self.M_200)
            if X > -1.3:
                X = -1.3
            self.alpha = 2.94 - lg(10**((X+2.33)*(-1.08)) + 10**((X+2.33)*2.29))
            self.beta  = 4.23 + 1.34*X + 0.26*X**2
            self.gamma = -0.06 + lg(10**((X+2.56)*(-0.68)) + 10**(X+2.56))            
            
        #coreNFW coefficients
        
        if self.pmodel in ('coreNFW', 'coreNFW_ek'):
            self.G = float(units.G.ratio(profile['rbins'].units**3*profile['M_200'].units**-1*units.Gyr**-2))
            self.t_sf = profile['t_sf']
        
        
        #fitting with different number of parameters
        
        self.rs_guess = numpy.power(10, lg(self.r_200) - 0.83+0.98*lg(self.M_200.in_units(units.Msol*10**12/profile['h'])))
        
        if self.pmodel == 'Einasto':
            self.initial_guess = [self.log_den[0], self.rs_guess, 1]
            self.bounding = ([self.log_den[-1], 0, -numpy.inf] , numpy.inf)
            
        elif self.pmodel == 'coreNFW':
            self.initial_guess = [self.log_den[0], self.rs_guess, 100, 1]
            self.bounding = ([self.log_den[-1], 0, 0, 0] , numpy.inf)
            
        else:
            self.initial_guess = [self.log_den[0], self.rs_guess]
            self.bounding = ([self.log_den[-1], 0] , numpy.inf)
        
        self.fitting()
        
        
    def fitting(self):
        """
        This function does the fitting procedure, takes no input, gives no output
        """
        self.params, self.covar = fit(self.log_rho, self.radii, self.log_den, sigma = self.log_den_error, absolute_sigma =  True, p0  = self.initial_guess, bounds = self.bounding, maxfev = 1000)
        
        self.C_200 = self.r_200 / self.params[1]
        
        
    def log_rho(self, r, log_rho_s, r_s, *args):
        '''
        Function that is being fitted as a density profile that come from the paper
        
        **Input**
        
        *r* radii
        
        *log_rho_s* is log10 of rho_s which is easier to fit this way
        
        *r_s* 
        
        '''
        
        if self.pmodel == 'pISO':
            return log_rho_s - lg((1+(r/r_s)**2))
        if self.pmodel == 'Burket':
            return log_rho_s - lg((1+(r/r_s)**2)) - lg(1+(r/r_s))
        if self.pmodel == 'NFW':
            return self.NFW(r, log_rho_s, r_s)
        if self.pmodel == 'Lucky13':
            return log_rho_s - 3*lg(1+(r/r_s))
        if self.pmodel == 'Einasto_ae':
            return log_rho_s - 2/(self.alpha_e)*((r/r_s)**self.alpha_e-1)*lg(numpy.e)
        if self.pmodel == 'Einasto':
            return log_rho_s - 2/(args[0])*((r/r_s)**args[0])*lg(numpy.e)
        if self.pmodel in ('DC14', 'DC14_X-1.3'):
            return log_rho_s - self.gamma*lg(r/r_s) - (self.beta - self.gamma)/self.alpha*lg(1+(r/r_s)**self.alpha)
        if self.pmodel == 'coreNFW':
            return self.coreNFW(r, log_rho_s, r_s, args[0], args[1])
        if self.pmodel == 'coreNFW_ek':
            return self.coreNFW_ek(r, log_rho_s, r_s)
            
        
    def output(self):
        '''
        Returns most needed things for futre plots
        '''
        
        return (self.M_200, self.C_200), self.params, self.covar
    
    #Functions that are needed for coreNFW fit
    def r_c(self, etta):
        return etta*self.shm_radius
    
    def NFW(self, r, log_rho_s, r_s ):
        return log_rho_s - lg(1+(r/r_s))*2 - lg(r/r_s)
    
    def M_NFW(self, r, log_rho_s, r_s):
        return 4*numpy.pi*(10**log_rho_s)*(r_s**3)*(numpy.log(1+r/r_s)+(r/r_s)/(1+r/r_s))
    
    def f(self, r, etta):
        return numpy.tanh(r/self.r_c(etta))
    
    def n(self, log_rho_s, r_s, k):
        return numpy.tanh(k*self.t_sf/self.t_dyn(log_rho_s, r_s))
    
    def t_dyn(self, log_rho_s, r_s):
        return 2*numpy.pi*numpy.sqrt(r_s**3/(self.G*self.M_NFW(r_s, log_rho_s, r_s)))
        # question: M_NFW(r_s) means M_NFW(r_s, log_rho_s, r_s)? probably yes

    def coreNFW(self, r, log_rho_s, r_s, etta, k):
        return  (2*self.n( log_rho_s, r_s, k)-1)*lg(self.f(r,etta)) + self.NFW(r, log_rho_s, r_s) + lg(self.M_NFW(r, log_rho_s, r_s)) +  lg(self.n( log_rho_s, r_s, k)) + lg(1-self.f(r, etta)**2) - lg(numpy.pi*4*self.r_c(etta)) - 2*lg(r)
    
    def coreNFW_ek(self, r, log_rho_s, r_s):
        etta = 1.75
        k = 0.04
        return self.coreNFW(r, log_rho_s, r_s, etta, k)

    
def models():
    '''
    This is a way of keeping all profile names here
    '''
    return ('pISO', 'Burket', 'NFW', 'Einasto', 'Einasto_ae', 'DC14', 'coreNFW', 'Lucky13')
    
