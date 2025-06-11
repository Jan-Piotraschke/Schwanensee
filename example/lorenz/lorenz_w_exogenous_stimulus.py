"""Backend supported: tensorflow.compat.v1, tensorflow, paddle

https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/lorenz.inverse.forced.html
"""

import deepxde as dde
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

from generated.synthetic_data_generator import SyntheticDataGenerator
from generated.physio_sensai_model import DeepXDESystem


# ===============================================
# SECTION 1: DATA GENERATION & INPUT PREPARATION
#
# Simulated Unobservable Data:
# This section covers creating the synthetic data
# from the Lorenz system using the true parameters
# and preparing the input data.
#
# Observable Data:
# None
# ===============================================
# time points
maxtime = 3
time = np.linspace(0, maxtime, 200)
ex_input = 10 * np.sin(2 * np.pi * time)  # exogenous input

sdg = SyntheticDataGenerator()
x0, constants = sdg.initConsts()

# solve ODE
x = odeint(sdg.ODE, x0, time)
time = time.reshape(-1, 1)


# ==========================================
# SECTION 2: PHYSICS MODEL DEFINITION
#
# This section defines the physical model
# with unknown parameters that we're trying to identify,
# including the system equations,
# boundary conditions (BC),
# and initial conditions (IC).
# ==========================================
# interpolate time / lift vectors (for using exogenous variable without fixed time stamps)
def ex_func2(t):
    spline = sp.interpolate.Rbf(
        time, ex_input, function="thin_plate", smooth=0, episilon=0
    )
    return spline(t[:, 0:])


dxs = DeepXDESystem(maxtime)

# Initial conditions
ic1 = dde.icbc.IC(dxs.geom, lambda X: x0[0], dxs.boundary, component=0)
ic2 = dde.icbc.IC(dxs.geom, lambda X: x0[1], dxs.boundary, component=1)
ic3 = dde.icbc.IC(dxs.geom, lambda X: x0[2], dxs.boundary, component=2)

# Get the training data
observation_handle = dxs.get_observations(time, x)


# =============================================
# SECTION 3: NEURAL NETWORK DESING & TRAINING
#
# This section sets up the neural network architecture,
# compiles the model,
# and trains it to learn the parameters.
# =============================================
# define data object
data = dde.data.PDE(
    dxs.geom,
    dxs.ODE_system,
    [ic1, ic2, ic3, *observation_handle],
    num_domain=400,
    num_boundary=2,
    anchors=time,
    auxiliary_var_function=ex_func2,
)

# define FNN architecture and compile
net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=dxs.constants)

# callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue(dxs.constants, period=100, filename=fnamevar)

model.train(iterations=35000, callbacks=[variable])


# ==========================================
# SECTION 4: RESULTS ANALYSIS & MODEL EXPORT
#
# This section analyzes the results,
# visualizes the parameter convergence,
# compares predicted vs actual trajectories,
# and reports the trained model.
# ==========================================
yhat = model.predict(time)
plt.figure()
plt.plot(time, x, "-", time, yhat, "--")
plt.xlabel("Time")
plt.legend(["x", "y", "z", "xh", "yh", "zh"])
plt.title("Training data")
plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))
export_path = os.path.join(script_dir, "model", "generalized_patient")
os.makedirs(os.path.dirname(export_path), exist_ok=True)
model.net.export(export_path)
