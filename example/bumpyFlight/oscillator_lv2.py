import deepxde as dde
import tf2onnx
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.integrate
from schwan.utils.plot import schwanensee
from schwan.utils.training import find_latest_checkpoint
from schwan.utils.default_nn import default_nn
from logger import get_logger

logger = get_logger("damped_oscillator")

# =============================================
# CONFIGURATION
# =============================================
IS_CONTINOUS_LEARNING = False
ITERATIONS = 2000

# Create directory for model saving
script_dir = os.path.dirname(os.path.abspath(__file__))
export_path = os.path.join(script_dir, "model", "damped_oscillator_simulator")
os.makedirs(os.path.dirname(export_path), exist_ok=True)

# ===============================================
# SECTION 1: DATA GENERATION & INPUT PREPARATION
#
# Simulated Physio Unobservable Data:
# This section covers creating the synthetic data
# from the physical model system
# and preparing the input data.
#
# Observable Physio Data:
# None
# ===============================================


# ODE system with support for multiple descent episodes
def elevated_damped_oscillator_ode(t, state, descent_schedule=None):
    """
    Oscillator with possible multiple stochastic descent events.
    descent_schedule: list of descent start times (each lasts 2s: 1s ramp down, 1s ramp up).
    """
    x, y = state

    # System parameters
    altitude_requested = 5.0
    altitude_transition_time = 0.5
    bumpy_hz = 4.0
    bumpy_r0 = 0.5
    climb_response = 2.0
    damping_ratio = 1

    # Descent contribution (sum of all active events)
    descent = 0.0
    if descent_schedule is not None:
        for t_start in descent_schedule:
            if t_start <= t <= t_start + 1.0:
                descent += 3.0 * (t - t_start)
            elif t_start + 1.0 < t <= t_start + 2.0:
                descent += 3.0 * (t_start + 2.0 - t)

    target_altitude = (
        altitude_requested * (1 - np.exp(-t / altitude_transition_time)) - descent
    )

    # Relative position
    x_rel = x
    y_rel = y - target_altitude
    r = np.sqrt(x_rel**2 + y_rel**2)

    # Damping
    effective_damping = damping_ratio * (1 - np.exp(-t / altitude_transition_time))

    # Auto climb
    auto_climb = 0.0
    if y < 4.3:
        auto_climb = 0.5 * (4.3 - y)

    dx_dt = (
        climb_response * np.exp(-t) - effective_damping * (r - bumpy_r0)
    ) * x_rel - bumpy_hz * y_rel
    dy_dt = (
        climb_response * np.exp(-t) - effective_damping * (r - bumpy_r0) + auto_climb
    ) * y_rel + bumpy_hz * x_rel

    return [dx_dt, dy_dt]


# Random descent schedule generator
def generate_descent_schedule(t_max, rng, prob_event=0.5, cutoff_time=15.0):
    """
    Generate a list of descent start times for one trajectory.
    - Only allowed after t > 5
    - Each event starts when y ~ near equilibrium, but here approximated by random draw
    - Multiple events possible
    - prob_event controls probability of triggering at each slot
    - cutoff_time: last possible start time for a descent
    """
    descent_times = []
    # Candidate windows every 2s after t=5 until min(t_max-2, cutoff_time)
    for candidate_t in np.arange(5, min(t_max - 2, cutoff_time), 2.0):
        if rng.random() < prob_event:
            descent_times.append(candidate_t)
    return descent_times


# Generate training data with stochastic descent episodes
def generate_data_state_and_rhs(
    num_samples=10000,
    t_max=20.0,
    x0_range=(-2, 4),
    y0_range=(0, 2),
    seed=1234,
):
    rng = np.random.default_rng(seed)
    X_list = []
    Y_list = []

    count = 0
    attempts = 0

    while count < num_samples and attempts < num_samples * 3:
        attempts += 1
        x0 = rng.uniform(x0_range[0], x0_range[1])
        y0 = rng.uniform(y0_range[0], y0_range[1])
        t_end = rng.uniform(0.0, t_max)

        # Generate descent schedule
        descent_schedule = generate_descent_schedule(t_max, rng, cutoff_time=15.0)

        try:
            sol = scipy.integrate.solve_ivp(
                lambda t, s: elevated_damped_oscillator_ode(t, s, descent_schedule),
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
                dx_dt_t, dy_dt_t = elevated_damped_oscillator_ode(
                    t_end, [x_t, y_t], descent_schedule
                )

                X_list.append([t_end, x_t, y_t])
                Y_list.append([x_t, y_t, dx_dt_t, dy_dt_t])
                count += 1
        except Exception:
            pass

    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    logger.info(
        f"Generated {X_arr.shape[0]} samples (requested {num_samples}) after {attempts} attempts"
    )
    return X_arr, Y_arr


# ==========================================
# SECTION 2: PHYSICS MODEL DEFINITION
#
# This section defines the physical model
# with unknown parameters that we're trying to identify,
# including the system equations,
# boundary conditions (BC),
# and initial conditions (IC).
# ==========================================


class ElevatedDampedOscillatorSystem:
    def __init__(self, t_min=0, t_max=20.0, xy_min=-1.0, xy_max=6.0):
        self.geom = dde.geometry.Cuboid(
            [t_min, xy_min, xy_min], [t_max, xy_max, xy_max]
        )
        self.altitude_requested = 5.0
        self.altitude_transition_time = 0.5
        self.bumpy_hz = 4.0
        self.bumpy_r0 = 0.5
        self.climb_response = 2.0
        self.damping_ratio = 1

    def ODE_system(self, X, Y):
        t = X[:, 0:1]
        x_pred = Y[:, 0:1]
        y_pred = Y[:, 1:2]
        xdot_pred = Y[:, 2:3]
        ydot_pred = Y[:, 3:4]

        dx_dt_autodiff = dde.grad.jacobian(Y, X, i=0, j=0)
        dy_dt_autodiff = dde.grad.jacobian(Y, X, i=1, j=0)

        threshold = 4.3
        auto_climb = 0.5 * dde.backend.tf.maximum(0.0, threshold - y_pred)

        # NOTE: In PINN we keep deterministic descent (9–11s) for consistency
        descent = dde.backend.tf.zeros_like(t)
        mask_down = dde.backend.tf.logical_and(t >= 9.0, t <= 10.0)
        descent = dde.backend.tf.where(mask_down, 3.0 * (t - 9.0), descent)
        mask_up = dde.backend.tf.logical_and(t > 10.0, t <= 11.0)
        descent = dde.backend.tf.where(mask_up, 3.0 * (11.0 - t), descent)

        target_altitude = (
            self.altitude_requested
            * (1 - dde.backend.exp(-t / self.altitude_transition_time))
            - descent
        )

        x_rel = x_pred
        y_rel = y_pred - target_altitude
        r = dde.backend.tf.sqrt(x_rel**2 + y_rel**2)

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

        eq1 = dx_dt_autodiff - xdot_pred
        eq2 = dy_dt_autodiff - ydot_pred
        eq3 = xdot_pred - rhs_x
        eq4 = ydot_pred - rhs_y

        return [eq1, eq2, eq3, eq4]

    def get_observations(self, X_input, Y_target):
        return [
            dde.icbc.PointSetBC(X_input, Y_target[:, 0:1], component=0),
            dde.icbc.PointSetBC(X_input, Y_target[:, 1:2], component=1),
            dde.icbc.PointSetBC(X_input, Y_target[:, 2:3], component=2),
            dde.icbc.PointSetBC(X_input, Y_target[:, 3:4], component=3),
        ]


# =============================================
# SECTION 3: NEURAL NETWORK DESIGN & TRAINING & MODEL EXPORT
#
# This section sets up the neural network architecture,
# compiles the model,
# and trains it to learn the parameters.
# =============================================

# Check for checkpoint first
latest_checkpoint = find_latest_checkpoint.find_latest_checkpoint()

# Create system
system = ElevatedDampedOscillatorSystem()

if latest_checkpoint and not IS_CONTINOUS_LEARNING:
    # =====================
    # LOAD EXISTING MODEL
    # =====================
    logger.info(f"Loading model from {latest_checkpoint} (no new data generated)")

    # Create dummy data geometry for model build
    dummy_data = dde.data.PDE(
        system.geom,
        system.ODE_system,
        [],
        num_domain=1,
        num_boundary=1,
    )

    # Input: [t, x, y], Output: predicted [x, y, dx/dt, dy/dt]
    # We choosed a count of at least 5 hidden layers, as we have to represent a limit cycle
    # NOTE ---
    # a 128 * 7 NN didn't improve the y fitting, only the x fitting, as the phase diagram got better
    # ---
    layer_sizes = [3] + [84] * 5 + [4]
    model, callbacks = default_nn.create(
        layer_sizes=layer_sizes,
        data=dummy_data,
        project_name="elevated_damped_oscillator_rhs",
    )

    model.compile("adam", lr=0.1)
    _ = model.predict(np.zeros((1, 3), dtype=np.float32))
    model.train(iterations=0, model_restore_path=latest_checkpoint, callbacks=callbacks)
    logger.info("Model successfully loaded from checkpoint.")

else:
    # =====================
    # DATA GENERATION
    # =====================
    logger.info("Generating training data...")
    X_train, y_train = generate_data_state_and_rhs(1000)
    logger.info("Generating test data...")
    X_test, y_test = generate_data_state_and_rhs(500)
    logger.info("Data generation complete!")

    # Build observations and data
    observation_points = system.get_observations(X_train, y_train)
    data = dde.data.PDE(
        system.geom,
        system.ODE_system,
        observation_points,
        num_domain=4000,
        num_boundary=400,
        anchors=X_train,
    )

    # Define neural network
    layer_sizes = [3] + [84] * 5 + [4]
    model, callbacks = default_nn.create(
        layer_sizes=layer_sizes,
        data=data,
        project_name="elevated_damped_oscillator_rhs",
    )

    if latest_checkpoint:
        logger.info(f"Loading model from {latest_checkpoint}")
        model.compile("adam", lr=0.1)
        _ = model.predict(X_train[:1])
        model.train(
            iterations=0, model_restore_path=latest_checkpoint, callbacks=callbacks
        )
        logger.info("Model successfully loaded from checkpoint.")

    # =====================
    # TRAINING
    # =====================
    if IS_CONTINOUS_LEARNING or not latest_checkpoint:
        # Step 1: emphasize data
        loss_weights = [10, 10, 10, 10] + [0, 0, 0, 0]
        model.compile("adam", lr=0.001, loss_weights=loss_weights)
        model.train(iterations=ITERATIONS, callbacks=callbacks)

        # Step 2: emphasize physics
        loss_weights = [1, 1, 1, 1] + [50, 50, 50, 50]
        model.compile("adam", lr=0.001, loss_weights=loss_weights)
        model.train(iterations=ITERATIONS, callbacks=callbacks)

        # Step 3: balance
        loss_weights = [1, 1, 1, 1] + [1, 1, 1, 1]
        model.compile("adam", lr=0.001, loss_weights=loss_weights)
        model.train(iterations=3000, callbacks=callbacks)
    else:
        logger.info("No further learning. Using the current model state.")

# Save the model
keras_path = f"{export_path}.keras"
model.net.save(keras_path)
model.net.export(export_path)

# NOTE ---
# variable batch size (-> thats why 'None'), but each sample has 3 features (t, x, y)
# ---
input_signature = [tf.TensorSpec([None, 3], tf.float32, name="inputs")]

# ! only temp as there is an open issue with tensorflow compability
# labels under which we can find the output
model.net.output_names = ["output"]
onnx_model, _ = tf2onnx.convert.from_keras(
    model.net, input_signature=input_signature, opset=15, output_path=None
)

# Save to file
onnx_path = "./model.onnx"
onnx.save(onnx_model, onnx_path)
logger.info(f"✅ ONNX model saved to {onnx_path}")

# ==========================================
# SECTION 4: RESULTS ANALYSIS
# Interfering ONNX model optional from C++
#
# This section analyzes the results,
# visualizes the parameter convergence,
# compares predicted vs actual trajectories,
# and reports the trained model.
# ==========================================

# Test the model on test data: compare predicted state to true state (first two components)
if "X_test" in globals() and "y_test" in globals():
    # Test the model on test data
    test_pred = model.predict(X_test)
    test_mse_state = np.mean((test_pred[:, 0:2] - y_test[:, 0:2]) ** 2)
    logger.info(f"Test MSE (state x,y): {test_mse_state:.6f}")
else:
    logger.info("No test data available (skipped test evaluation).")


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


def learned_rhs_func_from_onnx(onnx_model_path):
    sess = ort.InferenceSession(
        onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Read input/output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    def f(t, state):
        x_val, y_val = float(state[0]), float(state[1])
        inp = np.array([[t, x_val, y_val]], dtype=np.float32)
        pred = sess.run([output_name], {input_name: inp})[0]
        return [float(pred[0, 2]), float(pred[0, 3])]

    return f


def visualize_trajectory_with_onnx_rhs(
    onnx_model_path, x0, y0, t_max=15.0, num_points=500
):
    f_learned = learned_rhs_func_from_onnx(onnx_model_path)
    t_eval = np.linspace(0, t_max, num_points)
    sol = scipy.integrate.solve_ivp(
        f_learned, (0.0, t_max), [x0, y0], t_eval=t_eval, rtol=1e-6, atol=1e-8
    )
    x_learned = sol.y[0]
    y_learned = sol.y[1]
    return t_eval, x_learned, y_learned


# Visualize the oscillator behavior using your Schwanensee visualizer
x0, y0 = 1.0, 0.1  # Start near origin

t_values, x_learned, y_learned, x_true, y_true = visualize_trajectory_with_learned_rhs(
    model, x0, y0
)
# ONNX trajectory
t_values_onnx, x_learned_onnx, y_learned_onnx = visualize_trajectory_with_onnx_rhs(
    "./model.onnx", x0, y0
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

# Create a figure for PDF output
fig = plt.figure(figsize=(18, 12))

# Phase Space with LIC visualization
ax1 = fig.add_subplot(221)
phase_visualizer = schwanensee.SchwanenseeVisualizer(ax=ax1, onnx_model="model.onnx")
phase_visualizer.visualize(
    t_values,
    x_learned,
    y_learned,
    x_true,
    y_true,
    vector_field_type="density",
    lic_cmap="Greys_r",
    show_trajectories=False,
)
ax1.set_title("Phase Space with LIC Visualization")
ax1.grid(False)
ax1.legend()
params = phase_visualizer.highlight_circular_flow(
    x_range=(-3, 4),
    y_range=(-1, 7),
    vector_length=0.4,
    dist_tol=0.5,  # Search radius
    max_angle_deviation=15,  # Maximum deviation in degrees
)

# Phase Space with Arrows visualization
ax2 = fig.add_subplot(222)
phase_visualizer2 = schwanensee.SchwanenseeVisualizer(ax=ax2, onnx_model="model.onnx")
phase_visualizer2.visualize(
    t_values,
    x_learned,
    y_learned,
    x_true,
    y_true,
    vector_field_type="arrows",
    show_trajectories=True,
)
ax2.set_title("Phase Space with Vector Field Arrows")
ax2.grid(False)
ax2.legend()

# Time series for y
ax3 = fig.add_subplot(224)
ax3.plot(t_values, y_learned, "g--", label="y (learned TF)")
ax3.plot(t_values, y_true, "g-", alpha=0.7, label="y (True)")
ax3.plot(t_values_onnx, y_learned_onnx, "r-.", label="y (learned ONNX)")
ax3.set_title("y-Coordinate vs Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("y")
ax3.grid(True)
ax3.legend()

plt.tight_layout()

# For PDF output
plt.savefig("schwanensee_visualization.pdf", format="pdf", bbox_inches="tight")
plt.show()
