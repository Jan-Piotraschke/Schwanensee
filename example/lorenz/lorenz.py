import deepxde as dde
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

from generated.synthetic_data_generator import SyntheticDataGenerator

script_dir = os.path.dirname(os.path.abspath(__file__))
export_path = os.path.join(script_dir, "model", "lorenz_pinn")
os.makedirs(os.path.dirname(export_path), exist_ok=True)

# ===============================================
# SECTION 1: DATA GENERATION & INPUT PREPARATION
#
# Simulated Physio Unobservable Data:
# This section covers creating the synthetic data
# from the Lorenz system using the true parameters
# and preparing the input data.
#
# Observable Physio Data:
# None
# ===============================================
# time points
maxtime = 3
time_points = np.linspace(0, maxtime, 200)
ex_input = 10 * np.sin(2 * np.pi * time_points)  # exogenous input

# Initialize with synthetic data
sdg = SyntheticDataGenerator()
x0, constants = sdg.initConsts()

x_data = odeint(sdg.ODE, x0, time_points)
time = time_points.reshape(-1, 1)

# Generate training data pairs with different initial conditions
def generate_training_data(num_samples=100):
    # Generate varied initial conditions around x0
    x0_samples = np.array([x0]) + np.random.normal(0, 0.5, (num_samples, 3))

    # Time points for each trajectory
    time_samples = []
    trajectory_data = []

    # Generate trajectories for each initial condition
    for i in range(num_samples):
        # Solve ODE with this initial condition
        trajectory = odeint(sdg.ODE, x0_samples[i], time_points)

        # Store data points from this trajectory
        for j in range(len(time_points)):
            # Input: [t, x0, y0, z0]
            time_samples.append([time_points[j], x0_samples[i][0], x0_samples[i][1], x0_samples[i][2]])
            # Output: [x(t), y(t), z(t)]
            trajectory_data.append(trajectory[j])

    return np.array(time_samples), np.array(trajectory_data)

# Generate training and test data
X_train, y_train = generate_training_data(50)  # 50 different initial conditions
X_test, y_test = generate_training_data(10)   # 10 different initial conditions for testing

# ==========================================
# SECTION 2: PHYSICS MODEL DEFINITION
#
# This section defines the physical model
# with unknown parameters that we're trying to identify,
# including the system equations,
# boundary conditions (BC),
# and initial conditions (IC).
# ==========================================

class LorenzSystem:
    def __init__(self, t_min=0, t_max=maxtime):
        # Define domain: [t, x0, y0, z0]
        # Use approximate bounds for the initial conditions
        # These bounds define the realistic space
        # E.g. a concentration can never be negative, so min. value can't be smaller then 0
        x_min, x_max = 0, 5
        y_min, y_max = 0, 5
        z_min, z_max = 0, 5

        # 4D hypercube domain for the case of IC as NN input
        self.geom = dde.geometry.Hypercube(
            [t_min, x_min, y_min, z_min],
            [t_max, x_max, y_max, z_max]
        )

        # Define constants (trainable parameters)
        self.constants = [dde.Variable(1.0), dde.Variable(1.0), dde.Variable(1.0)]

    def ODE_system(self, x, y):
        """
        Physics-informed part: enforce the Lorenz system ODEs

        x: input features [t, x0, y0, z0]
        y: predicted output [x(t), y(t), z(t)]

        Returns the residuals of the Lorenz system equations
        """
        # Current predicted values
        x_val = y[:, 0:1]
        y_val = y[:, 1:2]
        z_val = y[:, 2:3]

        # Get the derivatives with respect to time
        dx_dt = dde.grad.jacobian(y, x, i=0, j=0)
        dy_dt = dde.grad.jacobian(y, x, i=1, j=0)
        dz_dt = dde.grad.jacobian(y, x, i=2, j=0)

        return [
            dx_dt - (self.constants[2]*(y_val-x_val)),
            dy_dt - (x_val*(self.constants[1]-z_val)-y_val),
            dz_dt - (x_val*y_val-self.constants[0]*z_val)
        ]

    def get_initial_conditions(self):
        """Create initial condition constraints for the neural network"""
        # Generate points specifically at t=0 for initial conditions
        # We'll manually create points with t=0 and various initial conditions
        num_ic_points = 10
        ic_points = np.zeros((num_ic_points, 4))
        ic_points[:, 0] = 0  # t=0
        # Random initial conditions in the domain
        ic_points[:, 1:] = np.random.uniform(0, 5, (num_ic_points, 3))

        # The values at these points should equal the initial conditions
        # Extract the initial conditions from the input points
        ic1_values = ic_points[:, 1:2]  # x0 values
        ic2_values = ic_points[:, 2:3]  # y0 values
        ic3_values = ic_points[:, 3:4]  # z0 values

        # Create IC constraints with actual values
        ic1 = dde.icbc.PointSetBC(ic_points, ic1_values, component=0)
        ic2 = dde.icbc.PointSetBC(ic_points, ic2_values, component=1)
        ic3 = dde.icbc.PointSetBC(ic_points, ic3_values, component=2)

        return [ic1, ic2, ic3]

    def get_observations(self, X, y):
        """Creates observation points for training"""
        return [
            dde.icbc.PointSetBC(X, y[:, 0:1], component=0),
            dde.icbc.PointSetBC(X, y[:, 1:2], component=1),
            dde.icbc.PointSetBC(X, y[:, 2:3], component=2)
        ]

# Create system
lorenz_system = LorenzSystem()

# =============================================
# SECTION 3: NEURAL NETWORK DESIGN & TRAINING
# =============================================

# Get the training observations and initial conditions
observation_points = lorenz_system.get_observations(X_train, y_train)
initial_conditions = lorenz_system.get_initial_conditions()

# Define the data object - combining physics constraints with data
data = dde.data.PDE(
    lorenz_system.geom,
    lorenz_system.ODE_system,
    initial_conditions + observation_points,  # Both ICs and data points
    num_domain=500,  # Collocation points for enforcing PDEs
    num_boundary=100,  # Points on the boundary
    anchors=X_train,  # Include training points in collocation
)

# Define neural network architecture
# Input: [t, x0, y0, z0], Output: [x(t), y(t), z(t)]
net = dde.nn.FNN([4] + [60] * 3 + [3], "tanh", "Glorot uniform")

# Build model and compile
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=lorenz_system.constants)

# Callbacks for storing results
fnamevar = "lorenz_variables.dat"
variable = dde.callbacks.VariableValue(lorenz_system.constants, period=100, filename=fnamevar)
checkpointer = dde.callbacks.ModelCheckpoint(
    "./checkpoints/lorenz_pinn", verbose=1, save_better_only=True, period=1000
)

# Train the model
# model.train(iterations=10000, callbacks=[variable, checkpointer])
model.train(
    iterations=0,
    model_restore_path="./checkpoints/lorenz_pinn-10000.ckpt",
    callbacks=[variable, checkpointer],
)

# ==========================================
# SECTION 4: RESULTS ANALYSIS & MODEL EXPORT
# ==========================================
# Generate predictions for visualization
# Let's use the original initial condition to generate a trajectory
t_vis = np.linspace(0, maxtime, 200)
inputs = np.column_stack((
    t_vis,
    np.full(t_vis.shape, x0[0]),
    np.full(t_vis.shape, x0[1]),
    np.full(t_vis.shape, x0[2])
))

# Get predictions
predicted_trajectory = model.predict(inputs)

# Get ground truth for comparison
true_trajectory = odeint(sdg.ODE, x0, t_vis)

plt.figure()
plt.plot(t_vis, true_trajectory, "-", time, predicted_trajectory, "--")
plt.xlabel("Time")
plt.legend(["x", "y", "z", "xh", "yh", "zh"])
plt.title("Training data")
plt.show()

# Save the model
model.save(export_path)
