import sys
sys.path.append("src")
import os
import numpy as np
import pickle
import scipy.io
from scipy.io import loadmat
from autoencoder import full_network
from training import create_feed_dictionary
from sindy_utils import sindy_simulate
from sindy_utils import sindy_model
import tensorflow as tf
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import pysindy as ps
from pysindy.differentiation import SmoothedFiniteDifference

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

hankel = np.zeros((10000,30))
dhankel = np.zeros((10000,30))
vhankel = np.zeros((5000,30))
vdhankel = np.zeros((5000,30))
mat = scipy.io.loadmat('f2o09m.mat') 
hnk = scipy.io.loadmat('h.mat') 
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


t0 = np.arange(0, 240, .02)
tv0 = np.arange(0, 160, .02)
t = np.arange(0, 200, .02)
tv = np.arange(0, 100, .02)

data = {}
data['t'] = t
data['x'] = hankel
data['dx'] = dhankel

validata = {}
validata['t'] = tv
validata['x'] = vhankel
validata['dx'] = vdhankel


# data = {}
# data['t'] = t
# data['x'] = H_ecg
# data['dx'] = dH_ecg


data_path = os.getcwd() + '\\'
#save_name = 'modelecg'
save_name = 'model'
params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))
params['save_name'] = data_path + save_name

autoencoder_network = full_network(params)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

tensorflow_run_tuple = ()
for key in autoencoder_network.keys():
    tensorflow_run_tuple += (autoencoder_network[key],)

    
# t = np.arange(0,20,.01)
# z0 = np.array([[-8,7,27]])

# data['z'] = data['z'].reshape((-1,params['latent_dim']))
# data['dz'] = data['dz'].reshape((-1,params['latent_dim']))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, data_path + save_name)
    test_dictionary = create_feed_dictionary(data, params)
    tf_results = sess.run(tensorflow_run_tuple, feed_dict=test_dictionary)

test_set_results = {}
for i,key in enumerate(autoencoder_network.keys()):
    test_set_results[key] = tf_results[i]

z_sim = sindy_simulate(test_set_results['z'][0],t,
                       params['coefficient_mask']*test_set_results['sindy_coefficients'],
                       params['poly_order'], params['include_sine'])

threshold = 0.0
model = ps.SINDy(
        differentiation_method=ps.SmoothedFiniteDifference(),
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=params['poly_order'], interaction_only =True))
model.fit(z_sim, t=t, ensemble=True)
model.print()

decoder_x_error = np.mean((data['x'] - test_set_results['x_decode'])**2)/np.mean(data['x']**2)
decoder_dx_error = np.mean((data['dx'] - test_set_results['dx_decode'])**2)/np.mean(data['dx']**2)
sindy_dz_error = np.mean((test_set_results['dz'] - test_set_results['dz_predict'])**2)/np.mean(test_set_results['dz']**2)

print('Rekon relative error: %f' % decoder_x_error)
print('Decoder relative SINDy error: %f' % decoder_dx_error)
print('SINDy reltive error, z: %f' % sindy_dz_error)

Xi_plot = (params['coefficient_mask']*test_set_results['sindy_coefficients'])
Xi_plot[Xi_plot==0] = np.inf
plt.figure(figsize=(1,2))
plt.imshow(Xi_plot, interpolation='none')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.clim([-10,30])


plt.show()

