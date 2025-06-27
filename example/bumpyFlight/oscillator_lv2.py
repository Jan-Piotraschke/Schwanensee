import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.integrate

# Create directory for model saving
script_dir = os.path.dirname(os.path.abspath(__file__))
export_path = os.path.join(script_dir, "model", "damped_oscillator_simulator")
os.makedirs(os.path.dirname(export_path), exist_ok=True)

# ===============================================
# SECTION 1: DATA GENERATION & INPUT PREPARATION
# ===============================================


# Define a system with sharp rise to elevated level followed by damped oscillations
def elevated_damped_oscillator_ode(t, state):
    """
    Modified oscillator with:
    1. Sharp initial rise due to exponential growth term
    2. Damped oscillations around equilibrium
    3. Final stable limit cycle
    """
    x, y = state

    # System parameters
    altitude_requested = 4.0  # Elevation offset
    altitude_transition_time = 1.5  # Time constant for transition
    climb_response = 2.0  # How quickly the system reacts to climb commands
    damping_ratio = 0.5  # Damping factor
    bumpy_hz = 4.0  # Angular frequency
    bumpy_r0 = 0.3  # Target radius for limit cycle

    # Calculate position relative to elevated center point
    x_rel = x
    y_rel = y - altitude_requested * (1 - np.exp(-t / altitude_transition_time))

    # Calculate radius from elevated center
    r = np.sqrt(x_rel**2 + y_rel**2)

    # Time-dependent damping - initially negative (growth), then positive (damping)
    effective_damping = damping_ratio * (1 - np.exp(-t / altitude_transition_time))

    # Modified dynamics with initial growth and transition to limit cycle
    dx_dt = (
        climb_response * np.exp(-t / altitude_transition_time)
        - effective_damping * (r - bumpy_r0)
    ) * x_rel - bumpy_hz * y_rel

    # Additional term for the vertical rise
    elevation_rate = (
        altitude_requested
        * np.exp(-t / altitude_transition_time)
        / altitude_transition_time
    )
    dy_dt = (
        (
            climb_response * np.exp(-t / altitude_transition_time)
            - effective_damping * (r - bumpy_r0)
        )
        * y_rel
        + bumpy_hz * x_rel
        + elevation_rate
    )

    return [dx_dt, dy_dt]


# Generate training data with numerical integration
def generate_data(num_samples=5000, t_max=20.0):
    # Generate various initial conditions (near origin) and time points
    x0_values = np.random.uniform(-0.3, 0.3, num_samples)
    y0_values = np.random.uniform(-0.3, 0.3, num_samples)
    t_values = np.random.uniform(0, t_max, num_samples)

    # For each sample, solve ODE from initial condition up to the requested time
    X = np.zeros((num_samples, 3))  # [t, x0, y0]
    Y = np.zeros((num_samples, 2))  # [x(t), y(t)]

    print("Generating numerical solutions...")
    for i in range(num_samples):
        if i % 500 == 0:
            print(f"  Progress: {i}/{num_samples}")

        x0 = x0_values[i]
        y0 = y0_values[i]
        t_end = t_values[i]

        # Solve ODE from initial condition to t_end
        t_span = [0, t_end]
        t_eval = [t_end]  # We only need solution at t_end
        sol = scipy.integrate.solve_ivp(
            elevated_damped_oscillator_ode,
            t_span,
            [x0, y0],
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
        )

        # Store input [t, x0, y0] and output [x(t), y(t)]
        X[i] = [t_end, x0, y0]
        Y[i] = [sol.y[0][-1], sol.y[1][-1]]

    return X, Y


# Generate training and test data
print("Generating training data...")
X_train, y_train = generate_data(3000)
print("Generating test data...")
X_test, y_test = generate_data(500)
print("Data generation complete!")

# ==========================================
# SECTION 2: PHYSICS MODEL DEFINITION
# ==========================================


class ElevatedDampedOscillatorSystem:
    def __init__(self, t_min=0, t_max=20.0, xy_min=-1.0, xy_max=5.0):
        # Define domain: time and initial values (x0, y0)
        self.geom = dde.geometry.Cuboid(
            [t_min, xy_min, xy_min], [t_max, xy_max, xy_max]
        )

        # System parameters
        self.climb_response = 2.0  # Initial growth rate
        self.damping_ratio = 0.5  # Damping factor
        self.bumpy_hz = 4.0  # Angular frequency
        self.bumpy_r0 = 0.3  # Target radius for limit cycle
        self.altitude_transition_time = 1.5  # Time constant for transition
        self.altitude_requested = 4.0  # Elevation offset

    def ODE_system(self, x, y):
        """
        Enforces the modified oscillator equations

        x: input features [t, x0, y0]
        y: predicted output [x, y]
        """
        t = x[:, 0:1]
        x_pred = y[:, 0:1]  # x coordinate
        y_pred = y[:, 1:2]  # y coordinate

        # Calculate position relative to elevated center point
        x_rel = x_pred
        y_rel = y_pred - self.altitude_requested * (
            1 - dde.backend.exp(-t / self.altitude_transition_time)
        )

        # Calculate radius from elevated center
        r = dde.backend.tf.sqrt(x_rel**2 + y_rel**2)

        # Time-dependent damping - initially negative (growth), then positive (damping)
        effective_damping = self.damping_ratio * (
            1 - dde.backend.exp(-t / self.altitude_transition_time)
        )

        # Compute derivatives with respect to time
        dx_dt = dde.grad.jacobian(y, x, i=0, j=0)  # dx/dt
        dy_dt = dde.grad.jacobian(y, x, i=1, j=0)  # dy/dt

        # Additional term for the vertical rise
        elevation_rate = (
            self.altitude_requested
            * dde.backend.exp(-t / self.altitude_transition_time)
            / self.altitude_transition_time
        )

        # Modified oscillator equations
        eq1 = dx_dt - (
            (
                self.climb_response
                * dde.backend.exp(-t / self.altitude_transition_time)
                - effective_damping * (r - self.bumpy_r0)
            )
            * x_rel
            - self.bumpy_hz * y_rel
        )

        eq2 = dy_dt - (
            (
                self.climb_response
                * dde.backend.exp(-t / self.altitude_transition_time)
                - effective_damping * (r - self.bumpy_r0)
            )
            * y_rel
            + self.bumpy_hz * x_rel
            + elevation_rate
        )

        return [eq1, eq2]

    def get_observations(self, X, y):
        """Creates observation points for training"""
        return [dde.icbc.PointSetBC(X, y, component=i) for i in range(2)]


# Create system
system = ElevatedDampedOscillatorSystem()


# =============================================
# SECTION 3: NEURAL NETWORK DESING & TRAINING
#
# This section sets up the neural network architecture,
# compiles the model,
# and trains it to learn the parameters.
# =============================================

# Get the training observations
observation_points = system.get_observations(X_train, y_train)

# Define the data object - combining physics constraints with data
data = dde.data.PDE(
    system.geom,
    system.ODE_system,
    observation_points,  # Training data points
    num_domain=4000,  # Number of collocation points for ODE
    num_boundary=400,  # Number of points on the boundary
    anchors=X_train,  # Include training points in collocation
)

# Define neural network architecture
# Input: [t, x0, y0], Output: predicted [x, y]
layer_sizes = [3] + [84] * 5 + [2]
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

# Callbacks for storing results
checkpointer = dde.callbacks.ModelCheckpoint(
    "./checkpoints/elevated_damped_oscillator",
    verbose=1,
    save_better_only=False,
    period=1000,
)
early_stopping = dde.callbacks.EarlyStopping(min_delta=0.5, patience=1000)

ITERATIONS = 6000

# We need to prevent the physics residuals from dominating the data fitting, as our Physio Model can't be perfect
# That's why we start with data-focused training to bring the PINN into the correct possition
loss_weights = [10, 10] + [0, 0]  # data weights + physics residual weights
model.compile(
    "adam",
    lr=0.001,
    loss_weights=loss_weights,
)

# Train the model
model.train(iterations=ITERATIONS, callbacks=[checkpointer, early_stopping])

# We increase the weights of the physics to create the oscillation
loss_weights = [1, 1] + [50, 50]  # data weights + physics residual weights
model.compile(
    "adam",
    lr=0.001,
    loss_weights=loss_weights,
)

# Train the model
model.train(iterations=ITERATIONS, callbacks=[checkpointer, early_stopping])

loss_weights = [1, 1] + [1, 1]  # data weights + physics residual weights
model.compile(
    "adam",
    lr=0.001,
    loss_weights=loss_weights,
)

# Train the model
model.train(iterations=1000, callbacks=[checkpointer, early_stopping])

# Dummy input to build the model
# _ = model.predict(X_train[:1])  # triggers internal build of the TF model
# model.train(
#     iterations=0,
#     model_restore_path=f"./checkpoints/elevated_damped_oscillator-{ITERATIONS}.weights.h5",
#     callbacks=[checkpointer],
# )


# ==========================================
# SECTION 4: RESULTS ANALYSIS & MODEL EXPORT
#
# This section analyzes the results,
# visualizes the parameter convergence,
# compares predicted vs actual trajectories,
# and reports the trained model.
# ==========================================

# Test the model on test data
test_pred = model.predict(X_test)
test_mse = np.mean((test_pred - y_test) ** 2)
print(f"Test MSE: {test_mse:.6f}")


# Generate predictions for visualization - complete trajectory from origin
def visualize_trajectory(x0, y0, t_max=15.0, num_points=500):
    # Generate time points
    t_values = np.linspace(0, t_max, num_points)

    # Create inputs for prediction: each row is [t, x0, y0]
    inputs = np.zeros((num_points, 3))
    inputs[:, 0] = t_values
    inputs[:, 1] = x0
    inputs[:, 2] = y0

    # Predict using trained model
    predicted_values = model.predict(inputs)
    x_pred = predicted_values[:, 0]
    y_pred = predicted_values[:, 1]

    # Generate ground truth via numerical integration
    t_span = [0, t_max]
    sol = scipy.integrate.solve_ivp(
        elevated_damped_oscillator_ode,
        t_span,
        [x0, y0],
        method="RK45",
        t_eval=t_values,
        rtol=1e-6,
    )
    x_true = sol.y[0]
    y_true = sol.y[1]

    # Calculate distance from the elevated center
    y_center = system.altitude_requested * (
        1 - np.exp(-t_values / system.altitude_transition_time)
    )
    r_pred = np.sqrt(x_pred**2 + (y_pred - y_center) ** 2)
    r_true = np.sqrt(x_true**2 + (y_true - y_center) ** 2)

    return t_values, x_pred, y_pred, x_true, y_true, r_pred, r_true, y_center


# Visualize the oscillator behavior
x0, y0 = 0.1, 0.1  # Start near origin
t_values, x_pred, y_pred, x_true, y_true, r_pred, r_true, y_center = (
    visualize_trajectory(x0, y0)
)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# Phase space plot showing the complete trajectory
ax1 = fig.add_subplot(221)
ax1.plot(x_pred, y_pred, "r--", label="NN Prediction")
ax1.plot(x_true, y_true, "b-", alpha=0.7, label="True Solution")
ax1.set_title("Phase Space - Elevated Damped Oscillations")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True)
ax1.legend()

# Time series for y
ax2 = fig.add_subplot(222)
ax2.plot(t_values, y_pred, "g--", label="y (NN)")
ax2.plot(t_values, y_true, "g-", alpha=0.7, label="y (True)")
ax2.set_title("y-Coordinate vs Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("y")
ax2.grid(True)
ax2.legend()

# Evaluate model error over time
x_error = np.abs(x_pred - x_true)
y_error = np.abs(y_pred - y_true)
total_error = np.sqrt(x_error**2 + y_error**2)

# Error plot with log scale (replaces previous x time series subplot)
ax3 = fig.add_subplot(223)
ax3.semilogy(t_values, x_error, "r-", label="x Error")
ax3.semilogy(t_values, y_error, "g-", label="y Error")
ax3.semilogy(t_values, total_error, "b-", label="Total Error")
ax3.set_title("Neural Network Prediction Error vs Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Absolute Error (log scale)")
ax3.grid(True)
ax3.legend()

# Amplitude plot (shows the damped oscillations around elevated center)
ax4 = fig.add_subplot(224)
ax4.plot(t_values, r_pred, "r--", label="Oscillation Amplitude (NN)")
ax4.plot(t_values, r_true, "b-", alpha=0.7, label="Oscillation Amplitude (True)")
ax4.set_title("Oscillation Amplitude vs Time\n(relative to elevated center)")
ax4.set_xlabel("Time")
ax4.set_ylabel("Amplitude")
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save(export_path)
model.net.save(f"{export_path}.keras")
model.net.export(export_path)
