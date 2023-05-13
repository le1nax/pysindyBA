# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load ECG data and set model name
from gen_ecg import xdat, Vt as V, U, y, dt, data_end, x, r
ModelName = 'ECG'

# Define figure path
figpath = './'

# Zeitreihe
fig, ax = plt.subplots()

tspan = np.arange(0, 60, .004)
ax.plot(tspan, xdat, 'k', linewidth=2)
# ax.set_xticks([0, 10, 20, 30, 40, 50])
# ax.set_yticks([-20, -10, 0, 10, 20])
fig.set_size_inches(2*250/100, 2*175/100)
ax.axis('off')
ax.set_xlim([30, 50])
# plt.show()
# fig.savefig(figpath + ModelName + '_p2.eps', format='eps', dpi=1000, bbox_inches='tight')
# Eingebetteter Attraktor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
L = slice(20000, 40001)
ax.plot(V[L, 0], V[L, 1], V[L, 2], color=[.1, .1, .1], linewidth=1.5)
ax.set_box_aspect([1,1,1])
ax.view_init(-92, 47)
fig.set_size_inches(3*250/100, 3*175/100)
# fig.savefig(figpath + ModelName + '_p3.eps', format='eps', dpi=1000, bbox_inches='tight')



# Forcierungen
xmin = 2000
xmax = 7000
sliced1 = np.arange(xmin, xmax, 1)
sliced5 = np.arange(xmin, xmax, 5)
L = slice(xmin, xmax)
L2 = slice(xmin, xmax, 5)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
ax1.plot(sliced1, x[L, 0], color=[.4, .4, .4], linewidth=2.5)
ax1.plot(sliced5, y[L2, 0], '.', color=[0, 0, .5], linewidth=10, markersize=5)
ax1.set_xlim([xmin, xmax])
ax1.axis('off')
ax2.plot(sliced1, x[L, -1], color=[.5, 0, 0], linewidth=1.5)
ax2.set_xlim([xmin, xmax])
ax2.axis('off')
fig.set_size_inches(2*250/100, 2*175/100)
# fig.savefig(figpath + ModelName + '_p4.eps', format='eps', dpi=1000, bbox_inches='tight')



#Rekonstruierter Attraktor
L = slice(10000, 20000)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(y[L, 0], y[L, 1], y[L, 2], color=[0, 0, 0.5], linewidth=1.5)
ax.set_xlim3d(auto=True)
ax.set_ylim3d(auto=True)
ax.set_zlim3d(auto=True)
ax.axis('off')
ax.view_init(-92, 47)
fig.set_size_inches(3*5, 3*3.5)
# plt.savefig(figpath + ModelName + '_p5.eps', format='eps', dpi=1200, bbox_inches='tight')



#Statistik der Forcierung
fig, ax = plt.subplots()
Vtest = np.std(V[:, r]) * np.random.randn(200000)
h, hc = np.histogram(V[:, r] - np.mean(V[:, r]), bins=np.arange(-0.02, 0.021, 0.005))
hnormal, hnormalc = np.histogram(Vtest - np.mean(Vtest), bins=np.arange(-0.02, 0.021, 0.005))
ax.semilogy(hnormalc[:-1], hnormal/np.sum(hnormal), '--', color=[0.2, 0.2, 0.2], linewidth=4)
ax.semilogy(hc[:-1], h/np.sum(h), color=[0.5, 0, 0], linewidth=4)
ax.set_ylim([0.0001, 1])
ax.set_xlim([-0.02, 0.02])
ax.axis('off')
fig.set_size_inches(2*5, 2*3.5)
# plt.savefig(figpath + ModelName + '_p6.eps', format='eps', dpi=1200, bbox_inches='tight')


#U-Moden
fig, ax = plt.subplots()
CC = np.array([[2, 15, 32],
               [2, 35, 92],
               [22, 62, 149],
               [41, 85, 180],
               [83, 124, 213],
               [112, 148, 223],
               [114, 155, 215]])
ax.plot(U[:, :r], color=[0.5, 0.5, 0.5], linewidth=1.5)
for k in range(6, -1, -1):
    ax.plot(U[:, k], linewidth=1.5+2*k/10, color=CC[k, :]/255)
ax.axis('off')
fig.set_size_inches(2*5, 2*3.5)
# plt.savefig(figpath + ModelName + '_p7.eps', format='eps', dpi=1200, bbox_inches='tight')



# Attraktor mit gekenntzeichneten Forcierungen
L = np.arange(1, len(V)+1)
inds = V[L-1,r-1]**2 > 1.e-5
L = L[inds]
startvals = []
endvals = []
start = 165
numhits = 342
for k in range(numhits):
    startvals.append(start)
    endmax = start + 50
    interval = np.arange(start, endmax)
    hits = np.where(inds[interval-1])[0]
    endval = start + hits[-1]
    endvals.append(endval)
    newhit = np.where(inds[endval+1-1:])[0]
    start = endval + newhit[0] + 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for k in range(155, 303): # Change 303 to numhits if needed
    ax.plot(V[startvals[k-155]:endvals[k-155]+1,0], V[startvals[k-155]:endvals[k-155]+1,1], V[startvals[k-155]:endvals[k-155]+1,2], 'r', linewidth=1.5)
for k in range(155, 302): # Change 302 to numhits-1 if needed
    ax.plot(V[endvals[k-155]+1:startvals[k-155+1],0], V[endvals[k-155]+1:startvals[k-155+1],1], V[endvals[k-155]+1:startvals[k-155+1],2], color=[.25, .25, .25], linewidth=1.5)
ax.set_box_aspect([np.ptp(V[:,0]), np.ptp(V[:,1]), np.ptp(V[:,2])])
ax.axis('off')
ax.view_init(-92, 47)
plt.gcf().set_size_inches(3*250/100, 3*175/100)
plt.show()
