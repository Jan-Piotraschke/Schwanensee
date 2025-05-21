import deepxde as dde

# Use TensorFlow backend
dde.backend.set_default_backend("tensorflow")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.interpolate import interp1d

from autograd import elementwise_grad
import autograd.numpy as anp

cmap = plt.get_cmap("tab10")

# FitzHugh-Nagumo model parameters
params = {"I": 0.5, "a": 0.8, "b": 0.7, "tau": 12.5}
ini = (0, 0)


def dv_dt(x, p=params):
    v, w = x.T
    return v - v**3 / 3 - w + p["I"]


def dw_dt(x, p=params):
    v, w = x.T
    return (v + p["a"] - p["b"] * w) / p["tau"]


def diff_eq(t, x, p=params):
    return [diff_eqs[i](x) for i in range(len(diff_eqs))]


def gen_truedata(func, t, ini, params):
    sol = integrate.solve_ivp(func, (min(t), max(t)), ini, t_eval=t, args=(params,))
    return sol.y.T


# Define PDE system using TensorFlow operations
def ode_system(x, y):
    t = x
    Z0 = y

    Z0dot = tf.squeeze(
        tf.stack(
            [dde.grad.jacobian(y, x, i=i, j=0) for i in range(len(diff_eqs))], axis=1
        )
    )

    # elementwise dot product
    constraint = tf.keras.backend.batch_dot(Z0, F_xlc) - 1

    # elementwise vector*matrix
    JT_dot_Z0 = tf.linalg.matvec(J_T, Z0)
    diff_eq = Z0dot + JT_dot_Z0

    return [constraint, diff_eq]


diff_eqs = [dv_dt, dw_dt]
eq_names = ["v", "w"]

# Simulate data to get the limit cycle
T = 1000
dt = 0.001
t_sim = np.linspace(0, T, int(T / dt))

x = gen_truedata(diff_eq, t_sim, ini, params)

# Detect peaks to get a single period
threshold = (max(x[:, 0]) - min(x[:, 0])) / 4 + min(x[:, 0])
crossed = False
pks = []
v_max_cur = -100

for i, v_ in enumerate(x[:, 0]):
    if v_ > threshold:
        if not crossed:
            crossed = True
            v_max_cur = v_
            i_peak = i
        else:
            if v_ > v_max_cur:
                i_peak = i
                v_max_cur = v_
    else:
        if crossed:
            pks.append(i_peak)
            crossed = False

P0 = (pks[-1] - pks[-2]) * dt
f0 = 1 / P0

# Limit cycle slice
x_lc = x[pks[-2] : pks[-1]]
t = t_sim[pks[-2] : pks[-1]] - t_sim[pks[-2]]

# Time derivative along limit cycle
F_xlc_ = np.array(diff_eq(t, x_lc)).T

# Compute Jacobian of limit cycle
J_ = anp.stack(
    [elementwise_grad(diff_eqs[i])(x_lc) for i in range(len(diff_eqs))], axis=1
)

# Geometry
geom = dde.geometry.TimeDomain(t[0], t[-1])
n_train = 300
n_bounds = 2

# Interpolation to match DeepXDE sampling points
data = dde.data.PDE(
    geom,
    ode_system,
    [],
    n_train,
    n_bounds,
    num_test=0,
)
data.test_x = data.train_x


f_interp = interp1d(t, F_xlc_.T)
F_xlc = tf.convert_to_tensor(f_interp(data.train_x.squeeze()).T, dtype=tf.float32)

# Jacobian interpolation
J_T = np.zeros((len(data.train_x), len(diff_eqs), len(diff_eqs)))
for ii, jj in enumerate(J_.T):
    for i, j in enumerate(jj):
        f = interp1d(t, j)
        J_T[:, ii, i] = f(data.train_x.squeeze())
J_T = tf.convert_to_tensor(J_T, dtype=tf.float32)


data.f = ode_system

# Define and train model
layer_size = [1] + [64] * 5 + [len(diff_eqs)]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.0001)
losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Evaluate model
Z0 = model.predict(t.reshape(-1, 1))

# Plot phase response curves
plt.figure(tight_layout=True)
for i, (z0, var) in enumerate(zip(Z0.T, eq_names)):
    plt.subplot(len(diff_eqs), 1, i + 1)
    plt.plot(t / P0, z0)
    plt.xlabel("phi")
    plt.ylabel(f"dphi/d{var}")

# Compare dZ0/dt and -J.T * Z0
plt.plot(t[1:] / P0, np.diff(Z0, axis=0) / dt, linewidth=5, c="y", label="dZ0/dt")
plt.plot(
    t / P0,
    -(J_.swapaxes(1, 2) @ Z0[..., None])[..., 0],
    linestyle="--",
    c="k",
    label="-J.T*Z0",
)
plt.xlabel("phi")
plt.legend()

# Dot product check: Z0 · F_xlc_
F_xlc_ = tf.convert_to_tensor(F_xlc_, dtype=tf.float32)
Z0 = tf.convert_to_tensor(Z0, dtype=tf.float32)
dot_product_results = tf.reduce_sum(Z0 * F_xlc_, axis=1)
plt.plot(t / P0, dot_product_results.numpy())
plt.xlabel("phi")
plt.ylabel("Z0 * W0")

model.net.export("floquet_mode_savedmodel")
