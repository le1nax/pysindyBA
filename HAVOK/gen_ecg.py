import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import StateSpace
from scipy.linalg import svd
from scipy.signal import savgol_filter

figpath = './figures/'
data = sio.loadmat('./data/f2o09m.mat')
V2 = sio.loadmat('./data/V2.mat')
V2 = V2['V']
S2 = sio.loadmat('./data/S.mat')
S2 = S2['S']
values = data['val'] #index1: resp, index2:ecg, index3: bp
ecg = values[2]
data_start = 40000
data_end = 15000
ecg45000= ecg[data_start:data_end+data_start]

tspan = np.arange(0, 60, .004)
xdat = ecg45000 - np.mean(ecg45000)

dt = tspan[1] - tspan[0]

# plt.plot(xdat, 'k')
# plt.show()

stackmax = 10
rmax = 5

H = np.zeros((stackmax, data_end-stackmax))
for j in range(0, stackmax):
    H[j,:] = xdat[j:j+data_end-stackmax]                              

U, s, Vt = svd(H)   
Vt = V2 #SVD kann alternativ in Matlab gemacht werden (bessere Auflösung der U moden)
s = S2
sigs = s
beta = stackmax / data_end
threshold = np.sqrt(2 * beta) * np.median(sigs)
r = len(sigs[sigs>threshold])
r = min(rmax, r)
r=5
print(r)


#Zustandsraumdarstellugn des HAVOK erstellen
x = Vt[0:data_end-stackmax-1,0:r]
xprime = Vt[1:data_end-stackmax, 0:r]
Xi = np.linalg.lstsq(x, xprime, rcond=None)[0]
B = Xi[:-1,-1]
A = Xi[:-1, :-1]
b0= Xi[0, -1]
b1= Xi[1, -1]
b2= Xi[2, -1]
b3= Xi[3, -1]
#b4= Xi[4, -1]
# b5= Xi[5, -1]
# b6= Xi[6, -1]
# b7= Xi[7, -1]
# b8= Xi[8, -1]
# b9= Xi[9, -1]

B1 = np.array([[b0], [b1], [b2], [b3]])#, [b4], [b5], [b6], [b7], [b8]])


C = np.eye(r-1)
D = np.array([[0], [0], [0], [0]])#,[0], [0], [0], [0],[0], [0]])
sys = scipy.signal.StateSpace(A,B1,np.eye(r-1),0.*B1,dt=dt)
# sys = scipy.signal.StateSpace(A, B, np.array([[1,0,0,0]]), np.array([[0]]), dt=dt)

t_interval = slice(0, 14000)
u = savgol_filter(x[t_interval, -1], 51, 3)
t, y, _ = scipy.signal.dlsim(sys, x[t_interval,r-1], tspan[t_interval], x0=x[0,0:r-1])

plt.plot(Vt[t_interval, 0], 'k')
# plt.plot(y[:,0], 'r')
plt.show()

# save data to file
# sio.savemat('./DATA/ecg.mat', {'y': y, 't': t})
# sio.savemat('./DATA/ecg2.mat', {'Vt': Vt})