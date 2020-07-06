"""
This file is simply for running on the server
"""

import time, DM_Profiles, numpy, scipy
import pynbody as pyn
from pynbody import units
from matplotlib import pylab as plt
s = pyn.load('/scratch/database/nihao/nihao_classic/g5.05e10/g5.05e10.00016')
s.physical_units()
h = s.halos()

def model_prep(halo):
    """
    This prepares profile for model to take in
    """
    # centering to generate profile and placing particles back 
    with pyn.analysis.angmom.faceon(halo, cen_size  =  '1 kpc'):
        r_200 = float(pyn.analysis.halo.virial_radius(halo.d, overden = 200))
        stellar_profile = pyn.analysis.profile.Profile(halo.s, min = 0.01, max = r_200, ndim = 2, type = 'equaln', nbins = 10000)
        shm_radius = stellar_profile['rbins'][len(stellar_profile['rbins'])//2]
#         profile = pyn.analysis.profile.Profile(halo.d, min = 2*max(halo.d['eps']), max = 0.2*r_200, ndim = 3, type = 'log', nbins = 50)
        profile = pyn.analysis.profile.Profile(halo.d, min = 10, max = 0.2*r_200, ndim = 3, type = 'log', nbins = 70)
    
    # calculating steallar and halo mass
    
    sm = halo.s['mass'].sum()
    hm = halo['mass'].sum()
    t_sf = halo.properties['time'].in_units('Gyr')
        
    return profile, sm, hm, shm_radius, r_200, t_sf

def panel_plot(rows, columns, to_save = False):
    '''
    This function plots a panel of all density profiles for a halo
    '''
    counter = 0 #you could make a panel function
    fig, ax = plt.subplots(rows,columns, sharex = True, sharey = True, figsize = (17, 15))
    for a in range(rows):
        for b in range(columns):
            hp = DM_Profiles.model(*variables, 'halo_' +  str(i), pmodel = DM_Profiles.models()[counter])
            ax[a][b].plot(hp.radii, numpy.power(10, hp.log_den), 'g--')
            ax[a][b].plot(hp.radii, numpy.power(10, hp.log_rho(numpy.array(hp.radii), *hp.params)), 'r-')
            ax[a][b].grid()
        #     ax.legend(('data','fit'))
            ax[a][b].set_title(hp.name + ' ' + hp.pmodel + ' density profile')
            ax[a][b].errorbar(hp.radii, numpy.power(10, hp.log_den) , yerr = numpy.array(hp.den_error), fmt = 'none')
            ax[a][b].set_yscale('log')
            ax[a][b].set_xscale('log')
            if b == 0:
                ax[a][b].set_ylabel(r'$\rho$ [M$_{\odot}$ /kpc$^{3}$]')   
            if counter == len(DM_Profiles.models()) - 1:
                break
            counter += 1
    for b in range(columns):
        ax[rows-1][b].set_xlabel('$R$ [kpc]')
    plt.show()
    if to_save:
        fig.savefig('/Graphs/density_plot.jpg')
    return fig


i = 1
halo = h[i]
start_time = time.time()
variables = model_prep(halo) 
columns = 3
rows = 3
panel_plot(rows, columns, to_save = True)


C_200list = []
e_C_200list = []
for k in DM_Profiles.models():
    hp = DM_Profiles.model(*variables, 'halo_' +  str(i), pmodel = k)
    C_200, r_s, e_r_s = hp.output()[0][1], hp.output()[1][1], numpy.sqrt(hp.output()[2][1][1])
    e_C_200 = hp.r_200/r_s**2*e_r_s
    C_200list.append(C_200)
    e_C_200list.append(e_C_200)
    print('for ' + k + ' profile \t' + 'C_200 = ' + str(C_200) + '\t +/- \t' + str(e_C_200))

e_C_200list = e_C_200list[1:]
lables = DM_Profiles.models()[1:]
x = range(1,len(C_200list)+1)
plt.bar(x, C_200list, color = 'blue')
plt.errorbar(x, C_200list, yerr = e_C_200list, color = 'red', fmt = 'none')
plt.xticks(x, lables)
plt.yscale('linear')
plt.show()
plt.safefit('/Graphs/C200.jpg')
