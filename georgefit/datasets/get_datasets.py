import numpy as np 
import os

thisdir = os.path.dirname(__file__)

def get_MaunaLoa():
    p = os.path.join(thisdir, 'MaunaLoa_dataset.dat')
    return np.loadtxt(p, unpack=True)

def get_simulated():
    p = os.path.join(thisdir, 'simulated_dataset.dat')
    return np.loadtxt(p, unpack=True)
