import deepxde as dde
import numpy as np

# Backend
from deepxde.backend import tf


# Define the pde function
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return (
        dy_t
        - dy_xx
        + tf.exp(-x[:, 1:])
        * (tf.sin(np.pi * x[:, 0:1]) - np.pi**2 * tf.sin(np.pi * x[:, 0:1]))
    )


# Initial condition function
def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


# Define geometry and time domain
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define boundary and initial conditions
bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

# Setup PDE data
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    train_distribution="pseudo",
    solution=func,
    num_test=10000,
)

# Define neural network
layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Create model
model = dde.Model(data, net)

# Setup callbacks and compile model
resampler = dde.callbacks.PDEPointResampler(period=100)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=2000, callbacks=[resampler])

# Plot and save results
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

model.net.save("heat.keras")
