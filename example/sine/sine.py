import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for model saving
script_dir = os.path.dirname(os.path.abspath(__file__))
export_path = os.path.join(script_dir, "model", "sine_wave_simulator")
os.makedirs(os.path.dirname(export_path), exist_ok=True)

# ===============================================
# SECTION 1: DATA GENERATION & INPUT PREPARATION
# ===============================================

# Generate training data
def generate_data(num_samples=5000):
    x0 = np.random.uniform(-np.pi, np.pi, num_samples)
    t = np.random.uniform(0, 4*np.pi, num_samples)

    # Create input matrix [t, x0]
    X = np.column_stack((t, x0))

    # Calculate target: sin(x0 + t)
    y = np.sin(x0 + t)

    return X, y.reshape(-1, 1)

# Generate training and test data
X_train, y_train = generate_data(5000)
X_test, y_test = generate_data(1000)

# ==========================================
# SECTION 2: PHYSICS MODEL DEFINITION
# ==========================================

class SineWaveSystem:
    def __init__(self, t_min=0, t_max=2*np.pi):
        # Define domain: time and initial value
        self.geom = dde.geometry.Rectangle([t_min, -np.pi], [t_max, np.pi])

        # For ODE system (optional if we want to enforce the differential equation)
        self.constants = []

    def ODE_system(self, x, y):
        """
        Enforces the differential equation: d²y/dt² + y = 0 (harmonic oscillator)
        which has sine waves as solutions

        x: input features [t, x0]
        y: predicted output
        """
        t = x[:, 0:1]
        dy_dt = dde.grad.jacobian(y, x, i=0, j=0)
        d2y_dt2 = dde.grad.hessian(y, x, i=0, j=0, component=0)

        return d2y_dt2 + y  # This should be zero for sine waves

    def get_observations(self, X, y):
        """Creates observation points for training"""
        return [dde.icbc.PointSetBC(X, y, component=0)]

# Create system
system = SineWaveSystem()

# =============================================
# SECTION 3: NEURAL NETWORK DESIGN & TRAINING
# =============================================

# Get the training observations
observation_points = system.get_observations(X_train, y_train)

# Define the data object - combining physics constraints with data
data = dde.data.PDE(
    system.geom,
    system.ODE_system,
    observation_points,  # Training data points
    num_domain=400,      # Number of collocation points for ODE
    num_boundary=100,    # Number of points on the boundary
    anchors=X_train,     # Include training points in collocation
)

# Define neural network architecture
# Input: [t, x0], Output: predicted sine value
net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot uniform")

# Build model and compile
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

# Callbacks for storing results
checkpointer = dde.callbacks.ModelCheckpoint(
    "./checkpoints/sine_wave", verbose=1, save_better_only=True, period=1000
)

# Train the model
model.train(
    iterations=0,
    model_restore_path="./checkpoints/sine_wave-5000.ckpt",
    callbacks=[checkpointer],
)
# model.train(iterations=5000 , callbacks=[checkpointer])

# ==========================================
# SECTION 4: RESULTS ANALYSIS & MODEL EXPORT
# ==========================================

# Test the model on test data
test_pred = model.predict(X_test)
test_mse = np.mean((test_pred - y_test) ** 2)
print(f"Test MSE: {test_mse:.6f}")

# Generate predictions for visualization
x0_value = 0.3
t_values = np.linspace(0, 6*np.pi, 200)
inputs = np.column_stack((t_values, np.full(t_values.shape, x0_value)))

predicted_values = model.predict(inputs)
true_values = np.sin(x0_value + t_values).reshape(-1, 1)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, true_values, 'b-', label=f'True sin({x0_value} + t)')
plt.plot(t_values, predicted_values, 'r--', label='NN Prediction')
plt.title('DeepXDE Neural Network Sine Wave Simulation')
plt.xlabel('Time (t)')
plt.ylabel('sin(x0 + t)')
plt.legend()
plt.grid(True)
plt.show()

# Save the model
model.save(export_path)
