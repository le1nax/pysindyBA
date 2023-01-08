import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np#
import scipy.io
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Generate training data

mat = scipy.io.loadmat('signals.mat')
s2a = mat['sig_2a_equi']
s2b = mat['sig_2b_equi']
s3a = mat['sig_3a_equi']
s3b = mat['sig_3b_equi']
dt = 10
t_train = np.arange(0,10000010, dt)
t_train_span = (t_train[0], t_train[-1])
x_train = np.vstack((np.transpose(s2a), np.transpose(s2b))).T
print(x_train)
print(x_train.ndim)
print(x_train.shape)

# dt = 0.01
# t_train = np.arange(0, 25, dt)    
# print(t_train)
# t_train_span = (t_train[0], t_train[-1])
# print(t_train_span)
# x0_train = [2, 0]
# x_train = solve_ivp(linear_damped_SHO, t_train_span, 
#                     x0_train, t_eval=t_train, **integrator_keywords).y.T

# Fit the model

poly_order = 3
threshold = 0.00005

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    #feature_library=ps.PolynomialLibrary(degree=poly_order),
    feature_library=ps.PolynomialLibrary(degree= 1),
)
model.fit(x_train, t=dt)
model.print()

# # Simulate and plot the results

x_sim = model.simulate([1,1], t_train)
plot_kws = dict(linewidth=2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(t_train, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
plt.show()