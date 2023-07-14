import os
from glob import glob
import kpicdrp.data as data
import kpicdrp.throughput as throughput
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import signal
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14) 

# Wavelength solution
wvsfl = '../first_guess_wvs_20200928_HIP_81497.fits'

def process_spec_throughtput(specfls, kmag, teff):

    spec = data.Dataset(filelist=specfls, dtype=data.Spectrum)

    wvs_sol = data.Wavecal(filepath=wvsfl)
    for spec_samp in spec:
        spec_samp.calibrate_wvs(wvs_sol)

    # Get absolute throughput
    thpts = []
    for spec_samp in spec:
        spec_throughput = throughput.calculate_peak_throughput(spec_samp, kmag, bb_temp=teff,return_spec=True)
        thpts.append(spec_throughput)
    thpts = np.array(thpts)

    return thpts


kmag = 5.593
teff = 6486

#onaxis final
datadir = "onaxis_final/fluxes/"
specfls = glob(datadir+"*.fits")

thpts_soln = np.ravel(np.mean(process_spec_throughtput(specfls,kmag,teff),axis=0))
thpts_soln = signal.medfilt(thpts_soln,kernel_size=201)

#onaxis final flatmap
datadir = "onaxis_final_flat/fluxes/"
specfls = glob(datadir+"*.fits")

thpts_flat = np.ravel(np.mean(process_spec_throughtput(specfls,kmag,teff),axis=0))
thpts_flat = signal.medfilt(thpts_flat,kernel_size=201)

wvs_sol = data.Wavecal(filepath=wvsfl)
wavels = wvs_sol.wvs[0]
wavels = np.ravel(wavels)

plt.figure(dpi=400)
plt.scatter(wavels,thpts_flat*100,s=5,label='flat map',color='k',alpha=0.5)
plt.scatter(wavels,thpts_soln*100,s=5,label='solution map', color='dodgerblue',alpha=0.5)
plt.xlabel('Wavelength (um)',fontsize=16)
plt.ylabel('Throughput (%)',fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.title('Throughputs (Oct 11, 2022)',fontsize=16)
plt.tight_layout()
plt.savefig('throughputs_221011.pdf')