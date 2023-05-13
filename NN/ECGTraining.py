import sys
sys.path.append("src")
import os
import datetime
import pysindy as ps
import pandas as pd
import numpy as np
import scipy.io
from scipy.io import loadmat
from pysindy.differentiation import SmoothedFiniteDifference
from sindy_utils import library_size
from training import train_network
from tensorflow.python.framework import ops
import tensorflow as tf


def get_hankel(x, dimension, delays, skip_rows=1):
    # if skip_rows>1:
    #     delays = len(x) - delays * skip_rows
    H = np.zeros((dimension, delays))
    for j in range(delays):
        H[:, j] = x[j*skip_rows:j*skip_rows+dimension]
    return H

def get_hankel_svd(H, reduced_dim):
    U, s, VT = np.linalg.svd(H, full_matrices=False)
    rec_v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))
    return U, s, VT, rec_v

# generate training, validation, testing data
noise_strength = 1e-6
# training_data = get_lorenz_data(1024, noise_strength=noise_strength)
# validation_data = get_lorenz_data(20, noise_strength=noise_strength)

    # data['t'] = t
    # data['x'] = x.reshape((n_ics*t.size, -1))
    # data['dx'] = dx.reshape((n_ics*t.size, -1))
    # data['ddx'] = ddx.reshape((n_ics*t.size, -1))
    # data['z'] = z.reshape((n_ics*t.size, -1))[:,0:1]
    # data['dz'] = z.reshape((n_ics*t.size, -1))[:,1:2]

   # reshape((-1,input_dim))
hankel = np.zeros((10000,30))
dhankel = np.zeros((10000,30))
vhankel = np.zeros((5000,30))
vdhankel = np.zeros((5000,30))
mat = scipy.io.loadmat('f2o09m.mat') 
hnk = scipy.io.loadmat('h.mat') #Hankelmatrix mit SVD, bessere res als np.linalg
hankel10000 = hnk['H'].T 
for j in range(1,30):
    for k in range(1,10000) :
        hankel[k,j] = hankel10000[k,j]

for j in range(1,30):
    for k in range(1000,1500) :
        vhankel[k-1000,j] = hankel10000[k-1000,j]  

t0 = np.arange(0, 240, .02) 
tv0 = np.arange(0, 160, .02)
t = np.arange(0, 200, .02) #10000
tv = np.arange(0, 100, .02) #5000

sfd = SmoothedFiniteDifference()
for j in range(1,30):
    dhankel[:,j] = sfd._differentiate(hankel[:,j],t)
sfd = SmoothedFiniteDifference()
for j in range(1,30):
    vdhankel[:,j] = sfd._differentiate(vhankel[:,j],tv)


values = mat['val'] #index1: resp, index2:ecg, index3: bp
ecg = values[2]
data_end = 12000
valdata_end = 20000
ecg10000= ecg[0:data_end]
valecg10000= ecg[data_end:valdata_end]
tau = 1
n_delays = 30
H_ecg = get_hankel(ecg10000,10000,n_delays,tau)

sfd = SmoothedFiniteDifference()
decg10000 = sfd._differentiate(ecg10000,t0)

sfd = SmoothedFiniteDifference()
dvalecg10000 = sfd._differentiate(valecg10000,tv0)

dH_ecg = get_hankel(decg10000,10000,n_delays,tau)
H_ecg_validation = get_hankel(valecg10000,5000,n_delays-10,tau)
dH_ecg_validation = get_hankel(dvalecg10000,5000,n_delays-10,tau)

data = {}
# data['t'] = t
# data['x'] = H_ecg
# data['dx'] = dH_ecg

# validata = {}
# validata['t'] = tv
# validata['x'] = H_ecg_validation
# validata['dx'] = dH_ecg_validation

data['t'] = t
data['x'] = hankel
data['dx'] = dhankel

print(hankel.shape)

validata = {}
validata['t'] = tv
validata['x'] = vhankel
validata['dx'] = vdhankel

params = {}

params['input_dim'] = 30
params['latent_dim'] = 6
params['model_order'] = 1
params['poly_order'] = 1
params['include_sine'] = False
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.3
params['threshold_frequency'] = 20
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# loss function weighting
params['loss_weight_rekon'] = 0.95
params['loss_weight_sindy_z'] = 0.01 
params['loss_weight_sindy_x'] = 0.1
params['loss_weight_cons'] = 0.95
params['loss_weight_z1'] = 0.01

params['activation'] = 'sigmoid'
params['widths'] = [64,32]

# training parameters
params['epoch_size'] = data['x'].shape[0]
params['batch_size'] = 30
params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 100

# training time cutoffs
params['max_epochs'] = 101
params['refinement_epochs'] = 201

num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
    params['save_name'] = 'model_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    ops.reset_default_graph()

    results_dict = train_network(data, data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

# df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')
df.to_pickle('model_'+ datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")+'.pkl')