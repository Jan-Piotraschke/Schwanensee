import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class SchwanenseeVisualizer:
    """
    A class for visualizing phase space trajectories with different phase regions.
    """

    def __init__(self, ax=None):
        """
        Initialize the visualizer with an optional matplotlib axis.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            The matplotlib axis to plot on. If None, a new one will be created.
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        else:
            self.ax = ax

        # Define solid colors for phase areas
        self.area_colors = {
            "stable": "#A453D4",  # Purple (highest priority)
            "descent": "#5E94CE",  # Dark grey (middle priority)
            "rise": "#AEAEAE",  # Light grey (lowest priority)
        }

        # Sort phases by priority (lowest to highest)
        self.phases_by_priority = ["rise", "descent", "stable"]

    def identify_phases(self, t_values, x_true, y_true, r_true, stable_radius=0.65):
        """
        Identify different phases in the trajectory.

        Parameters:
        -----------
        t_values : array-like
            Time values of the trajectory.
        x_true : array-like
            x-coordinates of the true trajectory.
        y_true : array-like
            y-coordinates of the true trajectory.
        r_true : array-like
            Radius values (distance from center) of the true trajectory.
        stable_radius : float, optional
            Radius threshold for stable oscillation.

        Returns:
        --------
        phase_masks : dict
            Dictionary containing boolean masks for each phase.
        t_10s_idx : int
            Index closest to t=10s.
        """
        # Initialize phase masks
        phase_masks = {
            "stable": np.zeros_like(
                t_values, dtype=bool
            ),  # Stable oscillation (highest priority)
            "descent": np.zeros_like(t_values, dtype=bool),  # Descent (second priority)
            "rise": np.zeros_like(
                t_values, dtype=bool
            ),  # Rise/Recovery (lowest priority)
        }

        # Find the index closest to t=10s
        t_10s_idx = np.argmin(np.abs(t_values - 10.0))

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

        return phase_masks, t_10s_idx

    def plot_trajectories(self, x_pred, y_pred, x_true, y_true, t_10s_idx):
        """
        Plot the predicted and true trajectories.

        Parameters:
        -----------
        x_pred : array-like
            x-coordinates of the predicted trajectory.
        y_pred : array-like
            y-coordinates of the predicted trajectory.
        x_true : array-like
            x-coordinates of the true trajectory.
        y_true : array-like
            y-coordinates of the true trajectory.
        t_10s_idx : int
            Index closest to t=10s.
        """
        # Split the trajectory into before and after t=10s for NN prediction
        self.ax.plot(
            x_pred[: t_10s_idx + 1],
            y_pred[: t_10s_idx + 1],
            color="darkred",
            linestyle="--",
            linewidth=2,
            label="NN Prediction (before 10s)",
        )
        self.ax.plot(
            x_pred[t_10s_idx:],
            y_pred[t_10s_idx:],
            color="lightcoral",
            linestyle="--",
            linewidth=2,
            label="NN Prediction (after 10s)",
        )

        # Plot true solution
        self.ax.plot(x_true, y_true, "b-", linewidth=2, label="True Solution")

        # Add square marker at the 10s position
        self.ax.scatter(
            x_pred[t_10s_idx],
            y_pred[t_10s_idx],
            s=100,
            c="red",
            marker="s",
            edgecolors="black",
            linewidths=1.5,
            label="State at 10s",
        )

    def plot_lic(
        self,
        model_function,
        t=0.0,
        x_range=(-3, 3),
        y_range=(0, 6),
        resolution=200,
        kernel_length=31,
        cmap="gray",
        alpha=0.8,
    ):
        """
        Add a grayscale Line Integral Convolution (LIC) vector field visualization.
        """
        try:
            import numpy as np
            from scipy import ndimage
        except ImportError:
            print("Required modules not available for LIC visualization")
            return

        # Ensure kernel length is odd
        if kernel_length % 2 == 0:
            kernel_length += 1

        # Create a grid for vector field computation
        x_grid = np.linspace(x_range[0], x_range[1], resolution)
        y_grid = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Calculate spacing for derivatives
        dx = (x_range[1] - x_range[0]) / (resolution - 1)
        dy = (y_range[1] - y_range[0]) / (resolution - 1)

        # Initialize arrays for the vector components
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Calculate vector components at each grid point
        # PINN: Use the trained model to predict dx/dt, dy/dt
        inputs = np.zeros((resolution * resolution, 3))
        inputs[:, 0] = t  # Fixed time
        inputs[:, 1] = X.flatten()  # Initial x
        inputs[:, 2] = Y.flatten()  # Initial y

        # Get the predicted state and compute gradient using finite differences
        delta_t = 0.01
        inputs_dt = inputs.copy()
        inputs_dt[:, 0] = t + delta_t

        # Predict at t and t+dt
        pred_t = model_function.predict(inputs)
        pred_t_dt = model_function.predict(inputs_dt)

        # Compute approximate derivatives
        dx_dt = (pred_t_dt[:, 0] - pred_t[:, 0]) / delta_t
        dy_dt = (pred_t_dt[:, 1] - pred_t[:, 1]) / delta_t

        # Reshape to grid
        U = dx_dt.reshape(resolution, resolution)
        V = dy_dt.reshape(resolution, resolution)

        # Normalize the vector field for LIC
        magnitude = np.sqrt(U**2 + V**2)
        eps = 1e-10  # Small value to avoid division by zero
        U_norm = U / (magnitude + eps)
        V_norm = V / (magnitude + eps)

        # Create white noise texture for LIC input
        texture = np.random.rand(resolution, resolution)

        # Apply LIC - we'll use a more accurate implementation than before
        lic_result = np.zeros_like(texture)
        half_kernel = kernel_length // 2

        # For each pixel, trace streamline and average the texture values
        for i in range(resolution):
            for j in range(resolution):
                # Starting position (in pixel coordinates)
                x, y = j, i

                # Initialize accumulators
                acc = texture[i, j]
                weight_sum = 1.0

                # Forward integration
                x_pos, y_pos = x, y
                for k in range(1, half_kernel + 1):
                    # Interpolate velocity at current position
                    ix, iy = int(x_pos), int(y_pos)

                    # Check if we're still in the grid
                    if ix < 0 or ix >= resolution - 1 or iy < 0 or iy >= resolution - 1:
                        break

                    # Bilinear interpolation weights
                    wx = x_pos - ix
                    wy = y_pos - iy

                    # Interpolate vector components
                    u_interp = (
                        (1 - wx) * (1 - wy) * U_norm[iy, ix]
                        + wx * (1 - wy) * U_norm[iy, ix + 1]
                        + (1 - wx) * wy * U_norm[iy + 1, ix]
                        + wx * wy * U_norm[iy + 1, ix + 1]
                    )
                    v_interp = (
                        (1 - wx) * (1 - wy) * V_norm[iy, ix]
                        + wx * (1 - wy) * V_norm[iy, ix + 1]
                        + (1 - wx) * wy * V_norm[iy + 1, ix]
                        + wx * wy * V_norm[iy + 1, ix + 1]
                    )

                    # Update position (simple Euler step)
                    x_pos += u_interp
                    y_pos += v_interp

                    # Check if still in bounds
                    if 0 <= x_pos < resolution - 1 and 0 <= y_pos < resolution - 1:
                        # Interpolate texture value at new position
                        ix, iy = int(x_pos), int(y_pos)
                        wx = x_pos - ix
                        wy = y_pos - iy

                        tex_val = (
                            (1 - wx) * (1 - wy) * texture[iy, ix]
                            + wx * (1 - wy) * texture[iy, ix + 1]
                            + (1 - wx) * wy * texture[iy + 1, ix]
                            + wx * wy * texture[iy + 1, ix + 1]
                        )

                        # Weight by distance from center (higher weight near center)
                        weight = 1.0 - k / half_kernel
                        acc += weight * tex_val
                        weight_sum += weight
                    else:
                        break

                # Backward integration
                x_pos, y_pos = x, y
                for k in range(1, half_kernel + 1):
                    # Interpolate velocity at current position
                    ix, iy = int(x_pos), int(y_pos)

                    # Check if we're still in the grid
                    if ix < 0 or ix >= resolution - 1 or iy < 0 or iy >= resolution - 1:
                        break

                    # Bilinear interpolation weights
                    wx = x_pos - ix
                    wy = y_pos - iy

                    # Interpolate vector components
                    u_interp = (
                        (1 - wx) * (1 - wy) * U_norm[iy, ix]
                        + wx * (1 - wy) * U_norm[iy, ix + 1]
                        + (1 - wx) * wy * U_norm[iy + 1, ix]
                        + wx * wy * U_norm[iy + 1, ix + 1]
                    )
                    v_interp = (
                        (1 - wx) * (1 - wy) * V_norm[iy, ix]
                        + wx * (1 - wy) * V_norm[iy, ix + 1]
                        + (1 - wx) * wy * V_norm[iy + 1, ix]
                        + wx * wy * V_norm[iy + 1, ix + 1]
                    )

                    # Update position (simple Euler step, but backward)
                    x_pos -= u_interp
                    y_pos -= v_interp

                    # Check if still in bounds
                    if 0 <= x_pos < resolution - 1 and 0 <= y_pos < resolution - 1:
                        # Interpolate texture value at new position
                        ix, iy = int(x_pos), int(y_pos)
                        wx = x_pos - ix
                        wy = y_pos - iy

                        tex_val = (
                            (1 - wx) * (1 - wy) * texture[iy, ix]
                            + wx * (1 - wy) * texture[iy, ix + 1]
                            + (1 - wx) * wy * texture[iy + 1, ix]
                            + wx * wy * texture[iy + 1, ix + 1]
                        )

                        # Weight by distance from center (higher weight near center)
                        weight = 1.0 - k / half_kernel
                        acc += weight * tex_val
                        weight_sum += weight
                    else:
                        break

                # Store average
                lic_result[i, j] = acc / weight_sum

        # Apply some contrast enhancement to make flow structure more visible
        lic_result = (lic_result - np.min(lic_result)) / (
            np.max(lic_result) - np.min(lic_result)
        )

        # Plot the LIC image
        img = self.ax.imshow(
            lic_result,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin="lower",
            cmap=cmap,
            alpha=alpha,
            interpolation="bicubic",  # Smooth interpolation for better visualization
        )

        return img

    def plot_phase_areas(self, x_true, y_true, phase_masks):
        """
        Plot the phase areas using convex hulls.

        Parameters:
        -----------
        x_true : array-like
            x-coordinates of the true trajectory.
        y_true : array-like
            y-coordinates of the true trajectory.
        phase_masks : dict
            Dictionary containing boolean masks for each phase.
        """
        # Clear any existing elements on the plot
        self.ax.clear()

        # Draw the phase areas in order of priority (lowest first, highest last)
        for phase in self.phases_by_priority:
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
                self.ax.fill(
                    hull_x,
                    hull_y,
                    color=self.area_colors[phase],
                    alpha=0.5,
                    label=f"{phase.capitalize()} Area",
                )

    def visualize(
        self,
        t_values,
        x_pred,
        y_pred,
        x_true,
        y_true,
        r_true,
        stable_radius=0.65,
        show_lic=False,
        pinn_model=None,
        vector_field_t=5.0,
        lic_resolution=200,
        lic_cmap="gray",
    ):
        # Identify phases
        phase_masks, t_10s_idx = self.identify_phases(
            t_values, x_true, y_true, r_true, stable_radius
        )

        # Plot phase areas
        self.plot_phase_areas(x_true, y_true, phase_masks)

        # Add LIC visualization if requested
        if show_lic:
            if pinn_model is not None:
                self.plot_lic(
                    pinn_model,
                    t=vector_field_t,
                    x_range=(-3, 4),
                    y_range=(-1, 7),
                    resolution=lic_resolution,
                    cmap=lic_cmap,
                    alpha=0.8,
                )

        # Plot trajectories
        self.plot_trajectories(x_pred, y_pred, x_true, y_true, t_10s_idx)

        return self.ax
