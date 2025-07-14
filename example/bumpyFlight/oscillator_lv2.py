import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.integrate
from psai.utils.training import find_latest_checkpoint

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
    4. Auto climb response when below threshold
    5. Planned descent at t~10s
    """
    x, y = state

    # System parameters
    altitude_requested = 5.0  # Elevation offset
    altitude_transition_time = 0.5  # Time constant for transition
    bumpy_hz = 4.0  # Angular frequency
    bumpy_r0 = 0.5  # Target radius for limit cycle
    climb_response = 2.0  # How quickly the system reacts to climb commands
    damping_ratio = 1  # Damping factor

    # Very simple descent function - linear ramp down and up
    descent = 0.0
    if 9 <= t <= 10:
        # Ramp down from 0 to 3 over 1 second
        descent = 3.0 * (t - 9)
    elif 10 < t <= 11:
        # Ramp up from 3 to 0 over 1 second
        descent = 3.0 * (11 - t)

    # Target altitude with descent
    target_altitude = (
        altitude_requested * (1 - np.exp(-t / altitude_transition_time)) - descent
    )

    # Calculate position relative to elevated center point
    x_rel = x
    y_rel = y - target_altitude

    # Calculate radius from elevated center
    r = np.sqrt(x_rel**2 + y_rel**2)

    # Time-dependent damping - initially negative (growth), then positive (damping)
    effective_damping = damping_ratio * (1 - np.exp(-t / altitude_transition_time))

    # Simple auto-climb response when below threshold (4.3)
    auto_climb = 0.0
    if y < 4.3:
        auto_climb = 0.5 * (4.3 - y)  # Simple proportional response

    # Modified dynamics with initial growth and transition to limit cycle
    dx_dt = (
        climb_response * np.exp(-t) - effective_damping * (r - bumpy_r0)
    ) * x_rel - bumpy_hz * y_rel

    dy_dt = (
        climb_response * np.exp(-t) - effective_damping * (r - bumpy_r0) + auto_climb
    ) * y_rel + bumpy_hz * x_rel

    return [dx_dt, dy_dt]


# Generate training data with numerical integration
def generate_data(num_samples=5000, t_max=20.0):
    # Generate various initial conditions (near origin) and time points
    x0_values = np.random.uniform(-0.3, 0.3, num_samples)
    y0_values = np.random.uniform(-0.3, 0.3, num_samples)
    t_values = np.random.uniform(0, t_max, num_samples)

    # For each sample, solve ODE from initial condition up to the requested time
    X = []  # [t, x0, y0]
    Y = []  # [x(t), y(t)]

    print("Generating numerical solutions...")
    count = 0
    for i in range(num_samples):
        if count % 500 == 0 and count > 0:
            print(f"  Progress: {count}/{num_samples}")

        x0 = x0_values[i]
        y0 = y0_values[i]
        t_end = t_values[i]

        # Solve ODE from initial condition to t_end
        try:
            t_span = [0, t_end]
            t_eval = [t_end]  # We only need solution at t_end
            sol = scipy.integrate.solve_ivp(
                elevated_damped_oscillator_ode,
                t_span,
                [x0, y0],
                method="RK45",
                t_eval=t_eval,
                rtol=1e-4,
                atol=1e-6,
                max_step=0.1,
            )

            if sol.success and sol.y.shape[1] > 0:
                # Store input [t, x0, y0] and output [x(t), y(t)]
                X.append([t_end, x0, y0])
                Y.append([sol.y[0][-1], sol.y[1][-1]])
                count += 1

        except Exception as e:
            # Skip this sample if integration fails
            pass

        if count >= num_samples:
            break

    print(f"Successfully generated {len(X)} samples")
    return np.array(X), np.array(Y)


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
    def __init__(self, t_min=0, t_max=20.0, xy_min=-1.0, xy_max=6.0):
        # Define domain: time and initial values (x0, y0)
        self.geom = dde.geometry.Cuboid(
            [t_min, xy_min, xy_min], [t_max, xy_max, xy_max]
        )

        # System parameters
        self.altitude_requested = 5.0  # Elevation offset
        self.altitude_transition_time = 0.5  # Time constant for transition
        self.bumpy_hz = 4.0  # Angular frequency
        self.bumpy_r0 = 0.5  # Target radius for limit cycle
        self.climb_response = 2.0  # Initial growth rate
        self.damping_ratio = 1  # Damping factor

    def ODE_system(self, x, y):
        """
        Enforces the modified oscillator equations

        x: input features [t, x0, y0]
        y: predicted output [x, y]
        """
        t = x[:, 0:1]
        x_pred = y[:, 0:1]  # x coordinate
        y_pred = y[:, 1:2]  # y coordinate

        # Simple auto-climb response when below threshold
        threshold = 4.3
        auto_climb = 0.5 * dde.backend.tf.maximum(0.0, threshold - y_pred)

        # Simple linear descent at t=10s
        descent = dde.backend.tf.zeros_like(t)

        # Ramp down from t=9 to t=10
        mask_down = dde.backend.tf.logical_and(t >= 9.0, t <= 10.0)
        descent = dde.backend.tf.where(mask_down, 3.0 * (t - 9.0), descent)

        # Ramp up from t=10 to t=11
        mask_up = dde.backend.tf.logical_and(t > 10.0, t <= 11.0)
        descent = dde.backend.tf.where(mask_up, 3.0 * (11.0 - t), descent)

        # Target altitude with descent
        target_altitude = (
            self.altitude_requested
            * (1 - dde.backend.exp(-t / self.altitude_transition_time))
            - descent
        )

        # Calculate position relative to elevated center point
        x_rel = x_pred
        y_rel = y_pred - target_altitude

        # Calculate radius from elevated center
        r = dde.backend.tf.sqrt(x_rel**2 + y_rel**2)

        # Time-dependent damping - initially negative (growth), then positive (damping)
        effective_damping = self.damping_ratio * (
            1 - dde.backend.exp(-t / self.altitude_transition_time)
        )

        # Compute derivatives with respect to time
        dx_dt = dde.grad.jacobian(y, x, i=0, j=0)  # dx/dt
        dy_dt = dde.grad.jacobian(y, x, i=1, j=0)  # dy/dt

        # Modified oscillator equations
        eq1 = dx_dt - (
            (
                self.climb_response * dde.backend.exp(-t)
                - effective_damping * (r - self.bumpy_r0)
            )
            * x_rel
            - self.bumpy_hz * y_rel
        )

        eq2 = dy_dt - (
            (
                self.climb_response * dde.backend.exp(-t)
                - effective_damping * (r - self.bumpy_r0)
                + auto_climb
            )
            * y_rel
            + self.bumpy_hz * x_rel
        )

        return [eq1, eq2]

    def get_observations(self, X, y):
        """Creates observation points for training based on 'real' data

        Logic: "At point 'X' the output should be 'y'"
        """
        return [
            dde.icbc.PointSetBC(X, y[:, 0:1], component=0),
            dde.icbc.PointSetBC(X, y[:, 1:2], component=1),
        ]


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
# We choosed a count of at least 5 hidden layers, as we have to represent a limit cycle
# NOTE ---
# a 128 * 7 NN didn't improve the y fitting, only the x fitting, as the phase diagram got better
# ---
layer_sizes = [3] + [84] * 5 + [2]
activation = "tanh"
kernel_initializer = "He normal"
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

ITERATIONS = 2000

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
model.train(iterations=3000, callbacks=[checkpointer, early_stopping])

# Try to find the latest checkpoint
latest_checkpoint = find_latest_checkpoint()

if latest_checkpoint:
    print(f"Loading model from {latest_checkpoint}")

    # Dummy input to build the model (triggers internal build of the TF model)
    _ = model.predict(X_train[:1])

    # Load the latest checkpoint
    model.train(
        iterations=0,
        model_restore_path=latest_checkpoint,
        callbacks=[checkpointer],
    )
    print("Model successfully loaded from checkpoint.")
else:
    print("No checkpoint found. Using the current model state.")


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
        rtol=1e-4,
        atol=1e-6,
    )
    x_true = sol.y[0]
    y_true = sol.y[1]

    # Calculate descent factor using simple linear ramp
    descent = np.zeros_like(t_values)
    # Ramp down
    mask_down = (t_values >= 9.0) & (t_values <= 10.0)
    descent[mask_down] = 3.0 * (t_values[mask_down] - 9.0)
    # Ramp up
    mask_up = (t_values > 10.0) & (t_values <= 11.0)
    descent[mask_up] = 3.0 * (11.0 - t_values[mask_up])

    # Calculate target altitude with planned descent
    y_center = (
        system.altitude_requested
        * (1 - np.exp(-t_values / system.altitude_transition_time))
        - descent
    )

    r_pred = np.sqrt(x_pred**2 + (y_pred - y_center) ** 2)
    r_true = np.sqrt(x_true**2 + (y_true - y_center) ** 2)

    # Find the index closest to t=10s
    t_10s_idx = np.argmin(np.abs(t_values - 10.0))

    return t_values, x_pred, y_pred, x_true, y_true, r_pred, r_true, y_center, t_10s_idx


# Visualize the oscillator behavior
x0, y0 = 0.1, 0.1  # Start near origin
t_values, x_pred, y_pred, x_true, y_true, r_pred, r_true, y_center, t_10s_idx = (
    visualize_trajectory(x0, y0)
)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# Phase space plot showing the complete trajectory
ax1 = fig.add_subplot(221)

# Define the stable oscillation radius (known parameter)
stable_radius = system.bumpy_r0 * 1.3  # Add a small margin to the known radius

# Initialize phase masks - we'll apply them in order of priority
phase_masks = {
    "stable": np.zeros_like(
        t_values, dtype=bool
    ),  # Stable oscillation (highest priority)
    "descent": np.zeros_like(t_values, dtype=bool),  # Descent (second priority)
    "rise": np.zeros_like(t_values, dtype=bool),  # Rise/Recovery (lowest priority)
}

# Step 1: Identify the descent phase first (focused around t=10s)
# Find the window where the planned descent occurs
descent_start_idx = np.argmin(np.abs(t_values - 9.0))
descent_end_idx = np.argmin(np.abs(t_values - 11.0))

# Extend window to find actual start and end of descent phase based on the behavior
extended_start = max(0, descent_start_idx - 10)
extended_end = min(len(t_values), descent_end_idx + 10)
extended_window = slice(extended_start, extended_end)

# Find when y drops significantly during descent
# Calculate the moving average of y to smooth out oscillations
window_size = 5
y_smooth = np.convolve(
    y_true[extended_window], np.ones(window_size) / window_size, mode="valid"
)
y_smooth_times = t_values[extended_window][window_size - 1 :]

# Find the lowest point in the smoothed data
min_idx_in_smooth = np.argmin(y_smooth)
actual_min_time = y_smooth_times[min_idx_in_smooth]
min_idx_in_full = np.argmin(np.abs(t_values - actual_min_time))

# Trace backward to find where descent begins (when r exceeds stable_radius)
descent_begin = min_idx_in_full
for i in range(min_idx_in_full, extended_start, -1):
    if (
        r_true[i] < stable_radius and y_true[i] > 4.0
    ):  # Inside stable zone and above threshold
        descent_begin = i + 1
        break

# Trace forward to find where descent ends (when system returns to stable oscillation)
descent_end = min_idx_in_full
for i in range(min_idx_in_full, extended_end):
    if (
        r_true[i] < stable_radius and y_true[i] > 4.0
    ):  # Back in stable zone and above threshold
        descent_end = i
        break

# Mark the descent phase
phase_masks["descent"][descent_begin : descent_end + 1] = True

# Step 2: Identify the stable oscillation phase
# This happens when the system is within stable_radius of the target altitude
# and after the initial rise phase (we'll use t > 2.0 to skip initial rise)
initial_rise_end = np.argmin(np.abs(t_values - 2.0))

for i in range(initial_rise_end, len(t_values)):
    # Skip if already marked as descent
    if phase_masks["descent"][i]:
        continue

    # If within stable radius and past initial rise
    if r_true[i] <= stable_radius:
        phase_masks["stable"][i] = True

# Step 3: Everything else is rise/recovery
phase_masks["rise"] = ~(phase_masks["stable"] | phase_masks["descent"])

# Create convex hulls for each phase to visualize the areas
from scipy.spatial import ConvexHull


# Define solid colors for phase areas
area_colors = {
    "stable": "#A453D4",  # Purple (highest priority)
    "descent": "#5E94CE",  # Dark grey (middle priority)
    "rise": "#AEAEAE",  # Light grey (lowest priority)
}

# Sort phases by priority (lowest to highest)
phases_by_priority = ["rise", "descent", "stable"]

# First clear any existing elements on the plot
ax1.clear()

# Draw the phase areas in order of priority (lowest first, highest last)
for phase in phases_by_priority:
    mask = phase_masks[phase]
    if np.sum(mask) > 3:  # Need at least 3 points
        points_x = x_true[mask]
        points_y = y_true[mask]

        # Create convex hull
        points = np.vstack((points_x, points_y)).T
        hull = ConvexHull(points)
        hull_vertices = hull.vertices
        hull_x = points[hull_vertices, 0]
        hull_y = points[hull_vertices, 1]

        # Plot the convex hull with solid color (alpha=1.0)
        ax1.fill(
            hull_x,
            hull_y,
            color=area_colors[phase],
            alpha=1.0,
            label=f"{phase.capitalize()} Area",
        )

# After drawing all areas, add the trajectories and markers on top
# Split the trajectory into before and after t=10s for NN prediction
ax1.plot(
    x_pred[: t_10s_idx + 1],
    y_pred[: t_10s_idx + 1],
    color="darkred",
    linestyle="--",
    linewidth=2,
    label="NN Prediction (before 10s)",
)
ax1.plot(
    x_pred[t_10s_idx:],
    y_pred[t_10s_idx:],
    color="lightcoral",
    linestyle="--",
    linewidth=2,
    label="NN Prediction (after 10s)",
)

# Plot true solution
ax1.plot(x_true, y_true, "b-", linewidth=2, label="True Solution")

# Add square marker at the 10s position
ax1.scatter(
    x_pred[t_10s_idx],
    y_pred[t_10s_idx],
    s=100,
    c="red",
    marker="s",
    edgecolors="black",
    linewidths=1.5,
    label="State at 10s",
)

# Set titles and labels
ax1.set_title("Phase Space - Elevated Damped Oscillations")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(False)
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
