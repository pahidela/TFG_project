
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
from plotear import plot_25_random

def norm(x):
    return(x/abs(max(x, key = abs)))
def norm2(data):
    data -= np.mean(data)
    data /= np.std(data)
    return(data)
    
#función que extrae number_patches de longitud=length  
#genera aleatoriamente un número para designar la señal aleatoria de las 100 
#genera otro número para determinar qué parte de la señal extrae  
def extract_patches_1D(A, length, number_patches, number_of_samples):
    flag = True
    #este while es para asegurarme de que no se genera ningún patch que sea
    #todo ceros 
    while flag == True: 
        end = randint(length, number_of_samples-1)
        start = end - length
        signal = randint(0, 99)
        if abs(max(A[signal][start:end], key = abs)) == 0.0:
            flag = True
        else: 
            flag = False
    M = np.atleast_2d(A[signal][start:end]) #añadimos el primer patch de la matriz

    for i in range(0, number_patches-1):
        flag = True
	#este while es para asegurarme de que no se genera ningún patch que sea todo ceros 
        while flag == True: 
            end = randint(length, number_of_samples-1)
            start = end - length
            signal = randint(0, 99)
            if abs(max(A[signal][start:end], key = abs)) == 0.0:
                flag = True
            else: 
                flag = False
        M = np.concatenate((M, np.atleast_2d(A[signal][start:end])), axis = 0) #añadimos el resto de patches
    return(M) 
    
#funcion para plotear visualmente 100 atomos en una grid cuadrada
#importante: la funcion da por hecho que el numero es de una red cuadrada
#ejemplo 10x10, 5x5...
def plot_number_atoms(D, length, number_atoms_to_plot):
    xa = np.linspace(0, 1, length)
    for i in range(0, number_atoms_to_plot):
        grid_size = int(number_atoms_to_plot**.5)
        ax = plt.subplot(grid_size, grid_size, i+1)
        plt.plot(xa, D[i], '-') 
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    plt.show()

#esta función recorta las señales a la parte significativa
def cut_signals(A, number_of_samples):
    #recortamos 2048 elementos alrededor del minimo de la señal
    index = 4000
                
    if number_of_samples == 512:
        up = 450; down = 62
    elif number_of_samples == 1024:
        up = 900; down = 124
    elif number_of_samples == 768:
        up = 600; down = 168
    elif number_of_samples == 1536:
        up = 1200; down = 336
    else:
        up = 1248; down = 800
        
    d = np.atleast_2d(A[0][int(index-down):int(index+up)])
    for i in range(1, 100):
        d = np.concatenate((d, np.atleast_2d(A[i][int(index-down):int(index+up)])), axis = 0)
    return(d)


        
def generar_diccionario(s, number_atoms, number_samples, length):    
    print('Generando el diccionario...')
    
    #recortamos las señales a 2048 muestras 
    s = cut_signals(s, number_samples)
        
    #creamos la matriz A de number_of_patches filas y length columnas
    A = extract_patches_1D(s, length, number_atoms, number_samples)
    
    #normalizamos los patches
    for i in range(0, number_atoms):
        A[i] -= np.mean(A[i]); A[i] /= np.std(A[i])
    
    #entrenamos el diccionario
    print('Learning the dictionary...') 
    t0 = time()
    dico = MiniBatchDictionaryLearning(n_components=number_atoms, 
              alpha = 1.2/np.sqrt(number_atoms), n_iter=1000, batch_size = 4, 
              fit_algorithm = 'lars', transform_algorithm = 'lasso_lars')


    V = dico.fit(A).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)

    #guardo el diccionario para cargarlo directamente
    np.save("DICT", V) 
    return(A, V)


#cargamos los datos en una matriz cuyas filas son señales 
mat_contents = sio.loadmat('H_dim')
data_dim = mat_contents['H_dim']; s = np.transpose(data_dim) 
#s = np.load('data.npy')

#declaramos el valor de los parámetros del diccionario y lo generamos  
number_atoms = 30; number_samples = 512; length = 64 ;
A, V = generar_diccionario(s, number_atoms, number_samples, length)  

plot_25_random(V, 30, 64)
