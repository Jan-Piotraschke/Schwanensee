import deepxde as dde
import numpy as np
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
maxtime = 2
time_points = np.linspace(0, maxtime, 150)
ex_input = 10 * np.sin(2 * np.pi * time_points)  # exogenous input

# Initialize with synthetic data
sdg = SyntheticDataGenerator()
x0, constants = sdg.initConsts()

physio_data = odeint(sdg.ODE, x0, time_points)
time = time_points.reshape(-1, 1)


# Generate training data pairs with different initial conditions
def generate_training_data(
    num_samples=100,
    positive_only=False,
    x_range=(-10, 20),
    y_range=(-10, 20),
    z_range=(-10, 20),
):
    # Prepare containers for data
    time_samples = []
    trajectory_data = []

    # Generate varied initial conditions
    x0_samples = []

    for _ in range(num_samples):
        # Sample each component from specified ranges
        x_init = np.random.uniform(x_range[0], x_range[1])
        y_init = np.random.uniform(y_range[0], y_range[1])
        z_init = np.random.uniform(z_range[0], z_range[1])

        # Apply positive-only constraint if requested
        if positive_only:
            x_init = abs(x_init)
            y_init = abs(y_init)
            z_init = abs(z_init)

        x0_samples.append([x_init, y_init, z_init])

    x0_samples = np.array(x0_samples)

    # Generate trajectories for each initial condition
    for i in range(num_samples):
        # Solve ODE with this initial condition
        try:
            trajectory = odeint(sdg.ODE, x0_samples[i], time_points)

            # Store data points from this trajectory
            for j in range(len(time_points)):
                # Input: [t, x0, y0, z0]
                time_samples.append(
                    [
                        time_points[j],
                        x0_samples[i][0],
                        x0_samples[i][1],
                        x0_samples[i][2],
                    ]
                )
                # Output: [x(t), y(t), z(t)]
                trajectory_data.append(trajectory[j])

        except Exception as e:
            print(
                f"Warning: Trajectory calculation failed for initial condition {x0_samples[i]}. Error: {str(e)}"
            )
            continue

    return np.array(time_samples), np.array(trajectory_data)


# Generate training and test data
X_train, y_train = generate_training_data(80)  # X different initial conditions
X_test, y_test = generate_training_data(10)  # 50 different initial conditions


# ==========================================
# SECTION 2: PHYSICS MODEL DEFINITION
#
# This section defines the physical model
# with unknown parameters that we're trying to identify,
# including the system equations,
# boundary conditions (BC),
# and initial conditions (IC).
# ==========================================


class DeepXDESystem:
    def __init__(self, t_min=0, t_max=maxtime):
        # Define domain: [t, x0, y0, z0]
        # Use approximate bounds for the initial conditions
        # These bounds define the realistic space
        # E.g. a concentration can never be negative, so min. value can't be smaller then 0
        # Here some realistic boundaries for the Lorenz system
        x_min, x_max = -20, 20
        y_min, y_max = -30, 30
        z_min, z_max = 0, 50

        # 4D hypercube domain for the case of IC as NN input
        self.geom = dde.geometry.Hypercube(
            [t_min, x_min, y_min, z_min], [t_max, x_max, y_max, z_max]
        )

        # Define constants (trainable parameters)
        # [Tipp:] Start with the median Physio parameters
        self.constants = [dde.Variable(2.5), dde.Variable(25.0), dde.Variable(8.0)]
        self.boundary = lambda _, on_boundary: on_boundary

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

        # Return the PDE residuals
        # Each residual represents how well the physics equation are satisfied
        # Returning a '0' means perfectly satisfied but could also result in an over-fit
        return [
            dx_dt - (self.constants[2] * (y_val - x_val)),
            dy_dt - (x_val * (self.constants[1] - z_val) - y_val),
            dz_dt - (x_val * y_val - self.constants[0] * z_val),
        ]

    def get_observations(self, X, y):
        """Creates observation points for training based on 'real' data

        Logic: "At point 'X' the output should be 'y'"
        """
        return [
            dde.icbc.PointSetBC(X, y[:, 0:1], component=0),
            dde.icbc.PointSetBC(X, y[:, 1:2], component=1),
            dde.icbc.PointSetBC(X, y[:, 2:3], component=2),
        ]


# Create system
dxs = DeepXDESystem()
observation_handle = dxs.get_observations(X_train, y_train)


# =============================================
# SECTION 3: NEURAL NETWORK DESING & TRAINING
#
# This section sets up the neural network architecture,
# compiles the model,
# and trains it to learn the parameters.
# =============================================
# Define data object
data = dde.data.PDE(
    dxs.geom,
    dxs.ODE_system,
    observation_handle,
    num_domain=1000,
    num_boundary=0,
    anchors=X_train,
)

# Define neural network architecture
# Input: [t, x0, y0, z0], Output: [x(t), y(t), z(t)]
layer_sizes = [4] + [64] * 5 + [3]
activation = "tanh"
kernel_initializer = "Glorot uniform"
dropout_rate = 0.01
net = dde.nn.FNN(
    layer_sizes=layer_sizes,
    activation=activation,
    kernel_initializer=kernel_initializer,
    dropout_rate=dropout_rate,
)

# Build model and compile
model = dde.Model(data, net)
# We have 3 weights for data fitting and 3 weights for alligning with the 3 physic equations
# We need to prevent the physics residuals from dominating the data fitting, as our Physio Model can't be perfect
loss_weights = [1, 1, 1] + [1, 1, 1]  # data weights + physics residual weights
model.compile(
    "adam",
    lr=0.001,
    external_trainable_variables=dxs.constants,
    loss_weights=loss_weights,
)

# Callbacks for storing results
# ! You can re-analyse the impact of each physio parameter using your physic model analytical
fnamevar = "simulated_physio_parameters.dat"
variable = dde.callbacks.VariableValue(dxs.constants, period=100, filename=fnamevar)
checkpointer = dde.callbacks.ModelCheckpoint(
    "./checkpoints/lorenz_pinn", verbose=1, save_better_only=True, period=1000
)
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-2, patience=1000)

ITERATIONS = 6000

# Train the model
model.train(iterations=ITERATIONS, callbacks=[variable, checkpointer, early_stopping])

# Dummy input to build the model
_ = model.predict(X_train[:1])  # triggers internal build of the TF model
model.train(
    iterations=0,
    model_restore_path=f"./checkpoints/lorenz_pinn-{ITERATIONS}.weights.h5",
    callbacks=[variable, checkpointer],
)

# model.compile("L-BFGS-B", lr=1.0e-5, loss_weights=loss_weights)
# model.compile("L-BFGS", external_trainable_variables=dxs.constants)
# model.train(iterations=10000, callbacks=[checkpointer, early_stopping])


# ==========================================
# SECTION 4: RESULTS ANALYSIS & MODEL EXPORT
#
# This section analyzes the results,
# visualizes the parameter convergence,
# compares predicted vs actual trajectories,
# and reports the trained model.
# ==========================================
# Generate predictions for visualization
# Let's use the original initial condition to generate a trajectory
input = np.column_stack(
    (
        time,
        np.full(time.shape, x0[0]),
        np.full(time.shape, x0[1]),
        np.full(time.shape, x0[2]),
    )
)

time_series_prediction = model.predict(input)
plt.figure()

colors = ["black", "red", "green"]

plt.plot(time, physio_data[:, 0], "-", color=colors[0])
plt.plot(time, physio_data[:, 1], "-", color=colors[2])
plt.plot(time, physio_data[:, 2], "-", color=colors[1])

plt.plot(time, time_series_prediction[:, 0], "--", color=colors[0])
plt.plot(time, time_series_prediction[:, 1], "--", color=colors[2])
plt.plot(time, time_series_prediction[:, 2], "--", color=colors[1])

plt.xlabel("Time")
plt.legend(["x", "y", "z", "xh", "yh", "zh"])
plt.title("Training data")
plt.show()

model.save(export_path)
