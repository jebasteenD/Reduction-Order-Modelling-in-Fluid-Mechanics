# -*- coding: utf-8 -*-
"""
load data """
#%% Import libraries
import numpy as np
from sklearn.preprocessing import minmax_scale
#%% Define Functions
def tecplot_reader(file):
    """Tecplot reader."""
    arrays = []
    with open(file, 'r') as a:
        for idx, line in enumerate(a.readlines()):
            if (idx+1)%(MP+NL) < (NL+1) and (idx+1)%(MP+NL)>0:
                continue
            else:
                arrays.append([float(s) for s in line.split()])
    return arrays

#%% Load data:
IMAX= 90 
JMAX= 500 
MP =IMAX*JMAX
NP =90         # number of snapshots
nr=10 		   # number of modes to use for ROM
NL=5
a= tecplot_reader('FIELD_TIMEascit.txt') 
a = np.asarray(a)
X=a[:,0]
Y=a[:,1]
U=a[:,4]
S=a[:,5]   
del a
X=np.reshape(X, (MP, NP),order='F')
Y=np.reshape(Y, (MP, NP),order='F')
U=np.reshape(U, (MP, NP),order='F')
S=np.reshape(S, (MP, NP),order='F')
U = minmax_scale(U, axis = 0)
muU = U.mean(axis=0)

S = minmax_scale(S, axis = 0)
muS = S.mean(axis=0)
Um=U.mean(axis=1)
Sm=S.mean(axis=1)
#%% POD
UU,sU,VU = np.linalg.svd(U - muU, full_matrices=False)
ZpcaU = np.dot(U - muU, VU.transpose())  

US,sS,VS = np.linalg.svd(S - muS, full_matrices=False)
ZpcaS = np.dot(S - muS, VS.transpose())  

ZpcaU=ZpcaU[:,:nr]
ZpcaS=ZpcaS[:,:nr]
#%% Save
np.save('Wsnapshots.npy',U)
np.save('MeanW.npy',muU)
np.save('WM.npy',Um)
np.save('Wbasis.npy',ZpcaU)

np.save('Ssnapshots.npy',S)
np.save('MeanS.npy',muS)
np.save('SM.npy',Sm)
np.save('Sbasis.npy',ZpcaS)



