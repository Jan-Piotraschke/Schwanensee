import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.integrate
from schwan.utils.plot import schwanensee
from schwan.utils.training import find_latest_checkpoint
from schwan.utils.default_nn import default_nn

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


# Generate training data where inputs are (t, x(t), y(t)) and outputs are (x(t), y(t), dx(t), dy(t))
def generate_data_state_and_rhs(
    num_samples=10000,
    t_max=20.0,
    x0_range=(-2, 4),
    y0_range=(0, 2),
    seed=1234,
):
    """
    For each sample:
      - sample (x0, y0) initial condition and a target time t_end
      - integrate the true ODE from 0 to t_end to get state [x(t_end), y(t_end)]
      - compute RHS at (t_end, x(t_end), y(t_end))
    Returns:
      X_inputs: shape (N,3) with rows [t, x_t, y_t]
      Y_targets: shape (N,4) with rows [x_t, y_t, dx_dt_t, dy_dt_t]
    """
    rng = np.random.default_rng(seed)
    X_list = []
    Y_list = []

    count = 0
    attempts = 0

    # We'll keep sampling until we have num_samples successful points (or bail out)
    while count < num_samples and attempts < num_samples * 3:
        attempts += 1
        x0 = rng.uniform(x0_range[0], x0_range[1])
        y0 = rng.uniform(y0_range[0], y0_range[1])
        t_end = rng.uniform(0.0, t_max)

        try:
            sol = scipy.integrate.solve_ivp(
                elevated_damped_oscillator_ode,
                (0.0, t_end),
                [x0, y0],
                method="RK45",
                t_eval=[t_end],
                rtol=1e-6,
                atol=1e-8,
                max_step=0.1,
            )
            if sol.success and sol.y.shape[1] > 0:
                x_t = float(sol.y[0, -1])
                y_t = float(sol.y[1, -1])
                dx_dt_t, dy_dt_t = elevated_damped_oscillator_ode(t_end, [x_t, y_t])

                X_list.append([t_end, x_t, y_t])
                Y_list.append([x_t, y_t, dx_dt_t, dy_dt_t])
                count += 1
        except Exception:
            # skip failed integration
            pass

    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    print(
        f"Generated {X_arr.shape[0]} samples (requested {num_samples}) after {attempts} attempts"
    )
    return X_arr, Y_arr


# Generate training and test data
print("Generating training data...")
X_train, y_train = generate_data_state_and_rhs(1000)
print("Generating test data...")
X_test, y_test = generate_data_state_and_rhs(500)
print("Data generation complete!")

# ==========================================
# SECTION 2: PHYSICS MODEL DEFINITION
# ==========================================


class ElevatedDampedOscillatorSystem:
    def __init__(self, t_min=0, t_max=20.0, xy_min=-1.0, xy_max=6.0):
        # Define domain: time and state space (we treat x and y as independent coordinates here)
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

    def ODE_system(self, X, Y):
        """
        X: input features [t, x_in, y_in] (we'll call them t, x_hat, y_hat)
        Y: predicted outputs from the NN with 4 components:
           Y[:,0] -> x_pred (state)
           Y[:,1] -> y_pred (state)
           Y[:,2] -> xdot_pred
           Y[:,3] -> ydot_pred
        Residuals enforce:
           1) d/dt x_pred == xdot_pred  (autodiff consistency)
           2) d/dt y_pred == ydot_pred
           3) xdot_pred == physics_rhs_x(t, x_pred, y_pred)
           4) ydot_pred == physics_rhs_y(t, x_pred, y_pred)
        """
        t = X[:, 0:1]
        x_pred = Y[:, 0:1]
        y_pred = Y[:, 1:2]
        xdot_pred = Y[:, 2:3]
        ydot_pred = Y[:, 3:4]

        # Autodiff time derivatives of the predicted state
        dx_dt_autodiff = dde.grad.jacobian(Y, X, i=0, j=0)  # d/dt of Y[:,0]
        dy_dt_autodiff = dde.grad.jacobian(Y, X, i=1, j=0)  # d/dt of Y[:,1]

        # Compute physics RHS (same formula as analytic ODE) evaluated at (t, x_pred, y_pred)
        # Build descent, target_altitude, auto_climb using symbolic/TF operations
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

        rhs_x = (
            self.climb_response * dde.backend.exp(-t)
            - effective_damping * (r - self.bumpy_r0)
        ) * x_rel - self.bumpy_hz * y_rel

        rhs_y = (
            self.climb_response * dde.backend.exp(-t)
            - effective_damping * (r - self.bumpy_r0)
            + auto_climb
        ) * y_rel + self.bumpy_hz * x_rel

        # Residuals
        eq1 = dx_dt_autodiff - xdot_pred
        eq2 = dy_dt_autodiff - ydot_pred
        eq3 = xdot_pred - rhs_x
        eq4 = ydot_pred - rhs_y

        return [eq1, eq2, eq3, eq4]

    def get_observations(self, X_input, Y_target):
        """
        Creates observation constraints for training.
        Here we supervise all 4 outputs at the sampled points:
          component 0 -> x (target Y_target[:,0])
          component 1 -> y (target Y_target[:,1])
          component 2 -> xdot (target Y_target[:,2])
          component 3 -> ydot (target Y_target[:,3])
        """
        return [
            dde.icbc.PointSetBC(X_input, Y_target[:, 0:1], component=0),
            dde.icbc.PointSetBC(X_input, Y_target[:, 1:2], component=1),
            dde.icbc.PointSetBC(X_input, Y_target[:, 2:3], component=2),
            dde.icbc.PointSetBC(X_input, Y_target[:, 3:4], component=3),
        ]


# Create system
system = ElevatedDampedOscillatorSystem()

# =============================================
# SECTION 3: NEURAL NETWORK DESIGN & TRAINING
#
# This section sets up the neural network architecture,
# compiles the model,
# and trains it to learn the parameters.
# =============================================
IS_CONTINOUS_LEARNING = False
ITERATIONS = 2000

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
# Input: [t, x, y], Output: predicted [x, y, dx/dt, dy/dt]
# We choosed a count of at least 5 hidden layers, as we have to represent a limit cycle
# NOTE ---
# a 128 * 7 NN didn't improve the y fitting, only the x fitting, as the phase diagram got better
# ---
layer_sizes = [3] + [84] * 5 + [4]
model, callbacks = default_nn.create(
    layer_sizes=layer_sizes, data=data, project_name="elevated_damped_oscillator_rhs"
)

# Try to find the latest checkpoint and restore if available
latest_checkpoint = find_latest_checkpoint.find_latest_checkpoint()

if latest_checkpoint:
    print(f"Loading model from {latest_checkpoint}")

    # Dummy input to build the model (triggers internal build of the TF model)
    model.compile("adam", lr=0.1)
    _ = model.predict(X_train[:1])

    # Load the latest checkpoint weights
    model.train(
        iterations=0,
        model_restore_path=latest_checkpoint,
        callbacks=callbacks,
    )
    print("Model successfully loaded from checkpoint.")
else:
    print("No checkpoint found. Using the current model state.")

# Training schedule similar to your original, but adapted for 4 outputs and 4 physics residuals
if IS_CONTINOUS_LEARNING or not latest_checkpoint:
    # Step 1: emphasize data to get the mapping right
    loss_weights = [10, 10, 10, 10] + [
        0,
        0,
        0,
        0,
    ]  # 4 data components + 4 physics residuals
    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    model.train(iterations=ITERATIONS, callbacks=callbacks)

    # Step 2: emphasize physics residuals to enforce dynamics
    loss_weights = [1, 1, 1, 1] + [
        50,
        50,
        50,
        50,
    ]  # reduce data weights, increase physics
    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    model.train(iterations=ITERATIONS, callbacks=callbacks)

    # Step 3: balance
    loss_weights = [1, 1, 1, 1] + [1, 1, 1, 1]
    model.compile("adam", lr=0.001, loss_weights=loss_weights)
    model.train(iterations=3000, callbacks=callbacks)
else:
    print("No further learning. Using the current model state.")


# ==========================================
# SECTION 4: RESULTS ANALYSIS & MODEL EXPORT
#
# This section analyzes the results,
# visualizes the parameter convergence,
# compares predicted vs actual trajectories,
# and reports the trained model.
# ==========================================

# Test the model on test data: compare predicted state to true state (first two components)
test_pred = model.predict(X_test)
test_mse_state = np.mean((test_pred[:, 0:2] - y_test[:, 0:2]) ** 2)
print(f"Test MSE (state x,y): {test_mse_state:.6f}")


# Utility: learned RHS function for integration: returns [xdot, ydot] for given (t, [x,y])
def learned_rhs_func_from_model(trained_model):
    def f(t, state):
        x_val, y_val = float(state[0]), float(state[1])
        inp = np.array([[t, x_val, y_val]], dtype=np.float32)
        pred = trained_model.predict(inp)
        # predicted xdot, ydot are components 2 and 3
        return [float(pred[0, 2]), float(pred[0, 3])]

    return f


# Generate predictions for visualization - integrate using learned RHS
def visualize_trajectory_with_learned_rhs(
    trained_model, x0, y0, t_max=15.0, num_points=500
):
    # Integrate using the learned RHS extracted from the NN
    f_learned = learned_rhs_func_from_model(trained_model)
    t_eval = np.linspace(0, t_max, num_points)
    sol = scipy.integrate.solve_ivp(
        f_learned, (0.0, t_max), [x0, y0], t_eval=t_eval, rtol=1e-6, atol=1e-8
    )
    x_learned = sol.y[0]
    y_learned = sol.y[1]

    # Ground truth trajectory
    sol_true = scipy.integrate.solve_ivp(
        elevated_damped_oscillator_ode,
        (0.0, t_max),
        [x0, y0],
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    x_true = sol_true.y[0]
    y_true = sol_true.y[1]

    return t_eval, x_learned, y_learned, x_true, y_true


# Visualize the oscillator behavior using your Schwanensee visualizer
x0, y0 = 1.0, 0.1  # Start near origin

t_values, x_learned, y_learned, x_true, y_true = visualize_trajectory_with_learned_rhs(
    model, x0, y0
)

# Compute radius about the planned center for plotting
descent = np.zeros_like(t_values)
mask_down = (t_values >= 9.0) & (t_values <= 10.0)
descent[mask_down] = 3.0 * (t_values[mask_down] - 9.0)
mask_up = (t_values > 10.0) & (t_values <= 11.0)
descent[mask_up] = 3.0 * (11.0 - t_values[mask_up])
y_center = (
    system.altitude_requested
    * (1 - np.exp(-t_values / system.altitude_transition_time))
    - descent
)

r_pred = np.sqrt(x_learned**2 + (y_learned - y_center) ** 2)
r_true = np.sqrt(x_true**2 + (y_true - y_center) ** 2)

# Create the phase space visualization with vector fields
fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(221)
phase_visualizer = schwanensee.SchwanenseeVisualizer(ax=ax1)

# LIC visualization with phase coloring using clustering
phase_visualizer.visualize(
    t_values,
    x_learned,
    y_learned,
    x_true,
    y_true,
    vector_field_type="lic",
    pinn_model=model,
    vector_field_t=0,
    lic_resolution=200,
    lic_cmap="gray",
    lic_alpha=0.7,
    lic_color_by_phase=True,
    lic_phase_method="cluster",
)

# LIC visualization with phase coloring using rule-based approach
ax2 = fig.add_subplot(222)
phase_visualizer2 = schwanensee.SchwanenseeVisualizer(ax=ax2)
phase_visualizer2.visualize(
    t_values,
    x_learned,
    y_learned,
    x_true,
    y_true,
    vector_field_type="arrows",
    pinn_model=model,
    vector_field_t=0,
    lic_resolution=200,
    lic_cmap="gray",
    lic_alpha=0.7,
    lic_color_by_phase=True,
    lic_phase_method="rule",
    arrow_grid_size=20
)

# Streamlines visualization
ax4 = fig.add_subplot(223)
phase_visualizer4 = schwanensee.SchwanenseeVisualizer(ax=ax4)
phase_visualizer4.visualize(
    t_values,
    x_learned,
    y_learned,
    x_true,
    y_true,
    vector_field_type="streamlines",
    pinn_model=model,
    vector_field_t=0,
    stream_density=1.5,
    stream_linewidth=1.2,
    stream_color="darkblue",
    stream_arrowsize=1.5,
    x_range=(-3, 4),
    y_range=(-1, 7),
)

ax1.set_title("Phase Space")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(False)
ax1.legend()

# Time series for y
ax2 = fig.add_subplot(224)
ax2.plot(t_values, y_learned, "g--", label="y (learned, integrated)")
ax2.plot(t_values, y_true, "g-", alpha=0.7, label="y (True)")
ax2.set_title("y-Coordinate vs Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("y")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save(export_path)
model.net.save(f"{export_path}.keras")
model.net.export(export_path)
