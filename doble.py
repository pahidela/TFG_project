
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from time import time
from skimage.metrics import structural_similarity as ssim
from snr_tools import set_snr, snr_cal, snr_cal1
from random import randint
from sklearn.decomposition import SparseCoder
from patches_1D import extract_patches_1d_Alex, reconstruct_from_patches_1d_Alex
from sklearn.metrics import mean_squared_error as mse 
from scipy.interpolate import interp1d
from numpy.fft import fft
from numpy.fft import fftfreq
from numpy.fft import fftshift
from gs import gs1
from scipy.optimize import minimize

def norm(x):
    if abs(max(x, key = abs)) == 0.0:
        return(x) 
    else: 
        return(x/abs(max(x, key = abs)))

def pplot(noised, original, denoised):
    #esta funcion es solo de VISUALIZACION
    denoised = norm(denoised); original = norm(original)
    x = np.linspace(0, 1, 16384)
    plt.subplot(2,1,1)
    down = 3600; up = 4600
    plt.plot(x[down:up], norm(noised[down:up]), '-', label = 'measured', linewidth = .85)
    plt.plot(x[down:up], denoised[down:up], '-r', label = 'denoised', linewidth = .85)
    plt.xlim([.22, .27])
    plt.legend(loc = 3)
    plt.subplot(2,1,2)
    
    plt.plot(x[down:up], original[down:up], '-', label = 'original', linewidth = .85)
    plt.plot(x[down:up], denoised[down:up], '-r', label = 'denoised', linewidth = .85)
    plt.legend(loc = 3)
    plt.xlim([.22, .27])
    plt.xlabel('Time (s)', fontsize = 16)
    plt.show()
def ppplot(noised, original, denoised):
    #esta funcion es solo de VISUALIZACION
    x = np.linspace(0, 1, 16384)

    plt.plot(x[3700:4600], norm(noised[3700:4600]), '-', label = 'measured', linewidth = .25)    
    plt.plot(x[3700:4600], norm(original[3700:4600]), '-k', label = 'original', linewidth = .85)
    plt.plot(x[3700:4600], norm(denoised[3700:4600]), '-r', label = 'denoised', linewidth = .65)
    plt.legend(loc = 4)
    plt.xlim([.23, .27])
    plt.xlabel('Time (s)', fontsize = 16)
    plt.show()
    
def pplot2(ESCALADA, SUMA, DENOISED, ORIGINAL_MOD):
    x = np.linspace(0, 1, 16384); down = 3700; up = 4600

    
    
    plt.subplot(212)
    plt.plot(x[down:up], norm(ORIGINAL_MOD[down:up]), '-', linewidth = 0.85, label = 'original')
    plt.plot(x[down:up], norm(DENOISED[down:up]), '-r', linewidth = 0.65, label = 'denoised')
    plt.legend(loc = 4)
    
    plt.xlabel('Time (s)', fontsize = 16)
    plt.xlim([.23, .27]);
    
    plt.subplot(211)
    plt.plot(x[down:up], SUMA[down:up], '-', linewidth = 0.65, label = 'señal+ruido')
    plt.plot(x[down:up], ESCALADA[down:up], '-r', linewidth = 1.05, label = 'original')
    plt.legend(loc = 4)
    plt.tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom = False)

    plt.xlim([.23, .27]); plt.ylim(-5e-21,5e-21)
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.show()
def double_pam(params, signal):
    
    t0 = time()
    lam, alpha = params

    #cargamos los datos
    timev, noise = np.loadtxt('noise.dat', delimiter='	', usecols=(0, 1), unpack=True)
    freq, asd = np.loadtxt('ruido_psd.dat', delimiter='	', usecols=(0, 1), unpack=True)
    psd = asd**2
    s = np.load('datagood.npy'); ORIGINAL = s[signal]
    
    #elegimos un segundo de tiempo del vector noise
    index = int(len(noise)/2); RUIDO = noise[0+index:index+int(len(noise)/2)]
    
    #escalamos la señal a la SNR correspondiente
    SNR = 5
    ESCALADA = set_snr(ORIGINAL, psd, SNR, freq, 16384, 1)[1]
    SUMA = ESCALADA + RUIDO
    
    #ejecutamos el código ROF
    #gs1(f, h, beta, lam, tol)
    DENOISED = gs1(SUMA, 1, .01, lam, .001)
    ORIGINAL_MOD = gs1(ESCALADA, 1, .01, lam, .001)
    
    #ppplot(norm(SUMA), norm(ORIGINAL_MOD), norm(DENOISED))
    SUMA = norm(SUMA); ORIGINAL_MOD = norm(ORIGINAL_MOD); DENOISED = norm(DENOISED)
    
    #parametros del metodo
    number_atoms = 300; number_samples = 512; length = 128
    
    #extract noisy patches from cleansed signal
    data = extract_patches_1d_Alex(DENOISED, length); data = np.transpose(data)
    data2 = extract_patches_1d_Alex(ORIGINAL_MOD, length); data2 = np.transpose(data2)
    
    #cargamos un diccionario
    #dic_contents = sio.loadmat('D_burst_128np300_short')
    #Da = dic_contents['D']; Da = np.transpose(Da)
    D = np.load("DICT.npy") 
    #atomos en filas!!
    
    #la función SparseCoder recibe los parámetros para ejecutar la reconstrucción
    coder = SparseCoder(dictionary=D, transform_algorithm='lasso_lars', transform_alpha = alpha)   
    code = coder.transform(data)
    patches = np.dot(code, D)
    
    code2 = coder.transform(data2)
    patches2 = np.dot(code2, D)
    
    #reconstruimos la señal a partir de los patches
    DENOISED = reconstruct_from_patches_1d_Alex(patches, 16384)
    ORIGINAL = reconstruct_from_patches_1d_Alex(patches2, 16384)

    upper = 4600; lower = 3700
    pplot2(ESCALADA, SUMA, DENOISED, ORIGINAL_MOD)
    
    print('SIGNAL =', signal), print('lambda = ', lam); print('alpha = ', alpha); print('\n')
    print('SSIM = ', ssim(norm(DENOISED)[lower:upper], norm(ORIGINAL)[lower:upper]))
    print('MSE = ', mse(norm(DENOISED)[lower:upper], norm(ORIGINAL)[lower:upper]))
    #print('\n')
    dt = time() - t0
    print('done in %.2fs.' % dt); print('\n')
    return(mse(norm(DENOISED)[lower:upper], norm(ORIGINAL)[lower:upper]))


initial_guess = [20, 1]

signals = np.arange(20,21, dtype = int)+100;

alphav = []; msev = []; lambdav = []
for signal in signals:
    result = minimize(double_pam, initial_guess, args = (signal), method='SLSQP', bounds = ((0.05, 25), (0.05, 2.5)))
    alphav.append(result.x[1]); lambdav.append(result.x[0]); msev.append(result.fun)

#np.save('M3_SSIM_alpha_snr20', alphav); np.save('M3_SSIM_snr20', msev); np.save('M3_SSIM_lambda_snr20', lambdav)


