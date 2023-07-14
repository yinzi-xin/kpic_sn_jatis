from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob
import kpicdrp.data as data
from scipy import signal
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14) 

# Wavelength solution
wvsfl = '../first_guess_wvs_20200928_HIP_81497.fits'
wvs_sol = data.Wavecal(filepath=wvsfl)
wavels = wvs_sol.wvs[0]
wavels = np.ravel(wavels)

flatlist = glob.glob('speckle_flat/fluxes/*.fits')
flatdata = []
for file in flatlist:
	flatdata.append(fits.getdata(file))

flat_mean = np.squeeze(np.mean(flatdata,axis=0))
flat_means = np.nanmean(flat_mean,axis=1)
flat_mean_o5 = np.nanmean(flat_mean[6])
flat_mean = np.ravel(flat_mean)
flat_medfilt = signal.medfilt(flat_mean,kernel_size=201)


nulledlist = glob.glob('speckle_m7/fluxes/*.fits')
nulleddata = []
for file in nulledlist:
	nulleddata.append(fits.getdata(file))

nulled_mean = np.squeeze(np.mean(nulleddata,axis=0))
nulled_means = np.nanmean(nulled_mean,axis=1)
nulled_mean_o5 = np.nanmean(nulled_mean[6])
nulled_mean = np.ravel(nulled_mean)
nulled_medfilt = signal.medfilt(nulled_mean,kernel_size=201)

o5_ratio = flat_mean_o5/nulled_mean_o5
print(o5_ratio)

ratios = flat_means/nulled_means
print(ratios)

plt.figure(dpi=400)
plt.subplot(2,1,1)
plt.scatter(wavels,flat_medfilt,s=5,label='flat map',color='k',alpha=0.5)
plt.scatter(wavels,nulled_medfilt,s=5,label='solution map', color='dodgerblue',alpha=0.5)
plt.ylim([0,900])
plt.fill_betweenx([0,900],2.29,2.34,alpha=0.25)
plt.xlabel('Wavelength (um)',fontsize=16)
plt.ylabel('Flux',fontsize=16)
plt.legend(loc='upper left',fontsize=12)
plt.grid(True)
plt.title('October 11, 2022',fontsize=16)

plt.subplot(2,1,2)
plt.scatter(wavels,nulled_medfilt/flat_medfilt,s=1,color='k')
plt.xlabel('Wavelength (um)',fontsize=16)
plt.ylabel('Ratio (Nulled/Flat)',fontsize=16)
plt.ylim([0,1.1])
plt.fill_betweenx([0,1.1],2.29,2.34,alpha=0.25)
plt.grid(True)
plt.tight_layout()
plt.savefig('null_shape_221011.pdf')