import time, DM_Profiles, numpy, scipy
import pynbody as pyn
from pynbody import units
from matplotlib import pyplot as plt

class galaxy:
    
    """
    This class stores and provides all the necessary information and methods that are done with the galaxy in this project
    
    **Input**
    
    *galaxy_path* path to the snapshot file
    
    *minimum*, *maximum* is a distance range for pynbody profile to take
    
    
    **Stores**
    
    *variables* see "output" of DM_Profies.model_prep
    
    *dmprofiles* are "model" objects from DM_Profiles
    
    *C200s* is a dicionary of C200 values for different profiles for the galaxy
    
    """
    
    def __init__(self, galaxy_path, minimum, maximum):
        
        self.galaxy_path = galaxy_path
        snapshot = pyn.load(galaxy_path)
        snapshot.physical_units()
        
        halo = snapshot.halos()[1]
        variables = DM_Profiles.model_prep(halo, minimum, maximum)
        pynprofile = variables[0]
        self.variables = (1, variables[1], variables[2], variables[3], variables[4], variables[5])
        self.dmprofiles = self.create_profiles(pynprofile)
        self.C200s = self.comp_C200s()
        
    def create_profiles(self, pynprofile):
        
        """
        This function takes variables from model_prep and outputs dark matter profile objects list

        """

        a, sm, hm, shm_radius, r_200, t_sf = self.variables #this is because  python2
        
        dmprofiles = []
        for model in DM_Profiles.models():
            dmprofiles.append(DM_Profiles.model(pynprofile, sm, hm, shm_radius, r_200, t_sf, self.galaxy_path, pmodel = model))

        return dmprofiles
    
    def comp_C200s(self):
        
        """
        This function returns a dictionary with all C200 values and their errors for each profile including Maccio2008
        """
        
        C200s = {}
        
        for hp in self.dmprofiles:
            C_200, r_s, e_r_s = hp.output()[0][1], hp.output()[1][1], numpy.sqrt(hp.output()[2][1][1])
            e_C_200 = hp.r_200/r_s**2*e_r_s
            C200s.update({hp.pmodel : (C_200, e_C_200)})
        
        #for Maccio 2008 NFW, coreNFW, Lucky (a question: should i do it for different kinds of a and b from the paper)
        
        M200 = float(self.variables[2].in_units(units.Msol*10**12/units.h))
        a = 0.830
        b = 0.098
        C200 = 10**a/M200**b
        CNFW = C200
        eC200 = 10**0.11 #have to figure this out later
        
        C200s.update({"M_NCL" : (C200, eC200)})
        
        #for Einasto
        a = 0.977
        b = 0.130
        C200 = 10**a/M200**b
        eC200 = 10**0.11 #have to figure this out later
        
        C200s.update({"M_Einasto" : (C200, eC200)})
        
        #for DC14
        
        X = numpy.log10(self.variables[1]/self.variables[2])
        C200 = CNFW*(1.0+0.00003*numpy.exp(3.4*(X+4.5)))
        eC200 = 10**0.11 #same shit here
        C200s.update({"M_DC14" : (C200, eC200)})
        
        return C200s
        
        
        
        
        
    
    def panel_plot(self, rows, columns, to_save = False):
        
        '''
        This function plots a panel of all density profiles for the galaxy which can be saved

        **Input**

        *rows*, *columns* are panel dimensions

        *profiles* is an array of dark matter profile objects

        *to_save* 

        '''
        #to make a 1d counter in 2d array
        counter = 0 

        #clearing out plt from  previous possible graphs and creating new figure
        plt.clf()
        fig, ax = plt.subplots(rows,columns, sharex = True, sharey = True, figsize = (15, 13))


        for a in range(rows):
            for b in range(columns):

                hp = self.dmprofiles[counter]
                ax[a][b].plot(hp.radii, numpy.power(10, hp.log_den), 'g--')
                ax[a][b].plot(hp.radii, numpy.power(10, hp.log_rho(numpy.array(hp.radii), *hp.params)), 'r-')
                ax[a][b].grid(b=True, which='major', color='b', linestyle='-', linewidth = 0.5)
                ax[a][b].grid(b=True, which='minor', color='b', linestyle='-', linewidth = 0.1)
                ax[a][b].set_title(hp.pmodel)
                ax[a][b].errorbar(hp.radii, numpy.power(10, hp.log_den) , yerr = numpy.array(hp.den_error), fmt = 'none')
                ax[a][b].set_yscale('log')
                ax[a][b].set_xscale('log')

                #creating labels only to the left
                if b == 0:
                    ax[a][b].set_ylabel(r'$\rho$ [M$_{\odot}$ /kpc$^{3}$]')

                #when there are more panel slots than models just breaks
                if counter == len(DM_Profiles.models()) - 1:
                    break
                counter += 1

        # creating labels only at the bottom
        for b in range(columns):
            ax[rows-1][b].set_xlabel('$R$ [kpc]')

        fig.suptitle("Panel plot for " + self.galaxy_path.split('/')[-1] + ' density profiles', fontsize = 16)
        fig.show()

        #saving the file
        if to_save:
            fig.savefig('./Graphs/' + self.galaxy_path.split('/')[-1] + 'density_plot.png')
    
    def C200_plot(self, to_save = False):
        
        '''
        This creates a bar plot of C200 values which can be saved
        '''
        plt.clf()
        fig = plt.figure(figsize = (7,5))
        C_200list = []
        e_C_200list = []
        lables = self.C200s.keys()
        for label in lables:
            C_200list.append(self.C200s[label][0])
            e_C_200list.append(self.C200s[label][1])
        x = range(1,len(C_200list)+1)
        fig.suptitle('C$_{200}$ ' + "values for " + self.galaxy_path.split('/')[-1])
        plt.bar(x, C_200list, color = 'blue')
        plt.errorbar(x, C_200list, yerr = e_C_200list, color = 'red', fmt = 'none')
        plt.xticks(x, lables)
        plt.yscale('linear')
        plt.xticks(rotation = 30)
        if to_save:
            fig.savefig('./Graphs/' + self.galaxy_path.split('/')[-1] + 'C200.png')
        fig.show()
