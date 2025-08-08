import numpy as np
import matplotlib.pyplot as plt
from .phase import PhaseIdentifier
from scipy.interpolate import RegularGridInterpolator
import scipy


class SchwanenseeVisualizer:
    def __init__(self, ax=None):
        """
        Initialize the visualizer with an optional matplotlib axis.
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

        self.phase_identifier = PhaseIdentifier()

    def plot_trajectories(self, x_pred, y_pred, x_true, y_true, t_10s_idx):
        """
        Plot the predicted and true trajectories.
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

    def plot_vector_field(
        self,
        pinn_model,
        x_range=(-2, 4),
        y_range=(0, 2),
        grid_density=20,
        x0=1.0,
        y0=0.1,
        t_max=15.0,
        num_points=500,
        arrow_scale=0.05,
        arrow_width=0.003,
        color="darkblue",
        alpha=0.6,
    ):
        """
        Plot vector field from the PINN model.

        Physical insight:
        - indication of state change speed and its acceleration / deceleration
        - spotting critical points as they appear with very short arrows
        """

        # Generate evenly spaced starting points
        num_points_x = 10
        num_points_y = 10
        x_vals = np.linspace(x_range[0], x_range[1], num_points_x)
        y_vals = np.linspace(y_range[0], y_range[1], num_points_y)
        start_points = [(xx, yy) for xx in x_vals for yy in y_vals]  # 100 points

        # Allowed plotting region
        x_min, x_max = -3, 4
        y_min, y_max = -1, 7

        for start_x, start_y in start_points:
            # Learned RHS from PINN
            def learned_rhs_func(t, state):
                x_val, y_val = float(state[0]), float(state[1])
                inp = np.array([[t, x_val, y_val]], dtype=np.float32)
                pred = pinn_model.predict(inp)
                return [float(pred[0, 2]), float(pred[0, 3])]

            # Integrate PINN trajectory
            t_eval = np.linspace(0, t_max, num_points)
            sol = scipy.integrate.solve_ivp(
                learned_rhs_func,
                (0.0, t_max),
                [start_x, start_y],
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8,
            )
            x_traj = sol.y[0]
            y_traj = sol.y[1]

            # NOTE: Keep only points inside allowed region
            mask = (
                (x_traj >= x_min)
                & (x_traj <= x_max)
                & (y_traj >= y_min)
                & (y_traj <= y_max)
            )
            if not np.any(mask):
                continue  # skip if nothing inside

            x_traj = x_traj[mask]
            y_traj = y_traj[mask]
            t_eval_masked = t_eval[mask]

            # Derivatives along masked path
            inputs = np.column_stack([t_eval_masked, x_traj, y_traj])
            preds = pinn_model.predict(inputs)
            dx_dt = preds[:, 2]
            dy_dt = preds[:, 3]

            # Normalize for display
            mag = np.sqrt(dx_dt**2 + dy_dt**2)
            mag[mag == 0] = 1.0
            dx_dt /= mag
            dy_dt /= mag

            # Arrows along masked path
            self.ax.quiver(
                x_traj,
                y_traj,
                dx_dt,
                dy_dt,
                angles="xy",
                scale_units="xy",
                scale=1 / arrow_scale,
                width=arrow_width,
                color=color,
                alpha=alpha,
            )

    def plot_streamlines(
        self,
        pinn_model,
        t=0.0,
        x_range=(-3, 3),
        y_range=(0, 6),
        density=1.0,
        linewidth=1.0,
        color="darkblue",
        arrowsize=1.2,
        arrowstyle="->",
        min_length=0.1,
        start_points=None,
        grid_size=40,
    ):
        """
        Add streamlines to visualize the flow in the vector field using direct derivative outputs.
        Fixed to ensure correct alignment with trajectories.

        Physical insight:
        - visualizing boundaries between different flow regimes easy visible
        """
        # Create a higher resolution grid for the vector field
        # NOTE: The ordering is crucial - numpy's meshgrid uses (x,y) but matplotlib's
        # streamplot expects arrays where first dimension is y, second is x
        y_grid = np.linspace(y_range[0], y_range[1], grid_size)
        x_grid = np.linspace(x_range[0], x_range[1], grid_size)
        Y, X = np.meshgrid(y_grid, x_grid)  # Note: Y, X order for meshgrid

        # Flatten in correct order for prediction
        points = np.zeros((grid_size * grid_size, 3))
        points[:, 0] = t  # Fixed time
        points[:, 1] = X.flatten()  # x coordinates
        points[:, 2] = Y.flatten()  # y coordinates

        # Get the predicted derivatives directly from the model
        predictions = pinn_model.predict(points)

        # Extract derivatives (components 2 and 3)
        dx_dt = predictions[:, 2].reshape(grid_size, grid_size)
        dy_dt = predictions[:, 3].reshape(grid_size, grid_size)

        # Create the streamplot
        # NOTE: streamplot takes x_grid, y_grid as 1D arrays, and U,V as 2D arrays
        # where U[i,j], V[i,j] is the vector at position (x_grid[j], y_grid[i])
        streamlines = self.ax.streamplot(
            x_grid,
            y_grid,
            dx_dt.T,
            dy_dt.T,
            density=density,
            linewidth=linewidth,
            color=color,
            arrowsize=arrowsize,
            arrowstyle=arrowstyle,
            integration_direction="forward",
            minlength=min_length,
            start_points=start_points,
        )

        return streamlines

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
        color_by_phase=False,
        phase_method="rule",  # "rule" or "cluster"
    ):
        """
        Add a Line Integral Convolution (LIC) vector field visualization using direct derivative outputs.

        Parameters:
            color_by_phase: If True, colors the LIC by phase regions
            phase_method: Method to determine phases ("rule" or "cluster")

        Physical insight:
        - reveal of the dense flow pattern
        - stencil source for the drawing of the phases into the Phase Space
        """
        # Ensure kernel length is odd
        if kernel_length % 2 == 0:
            kernel_length += 1

        # Create a grid for vector field computation
        x_grid = np.linspace(x_range[0], x_range[1], resolution)
        y_grid = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Initialize arrays for the vector components
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Calculate vector components at each grid point using model's direct derivative outputs
        inputs = np.zeros((resolution * resolution, 3))
        inputs[:, 0] = t  # Fixed time
        inputs[:, 1] = X.flatten()  # Current x
        inputs[:, 2] = Y.flatten()  # Current y

        # Get the predicted derivatives directly
        predictions = model_function.predict(inputs)

        # Extract derivatives (components 2 and 3)
        dx_dt = predictions[:, 2]
        dy_dt = predictions[:, 3]

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

        # Add phase coloring if requested
        if color_by_phase:
            # Generate phase classification
            if phase_method.lower() == "cluster":
                phases, X_phase, Y_phase, field = (
                    self.phase_identifier.cluster_based_classification(
                        model_function, t, x_range, y_range, grid_size=50, n_clusters=3
                    )
                )
            else:  # Default to rule-based
                phases, X_phase, Y_phase, field = (
                    self.phase_identifier.rule_based_classification(
                        model_function, t, x_range, y_range, grid_size=50
                    )
                )

            phase_colors = self.phase_identifier.get_phase_colors(phases)
            phase_colors_high_res = self.phase_identifier.interpolate_phase_colors(
                phase_colors, x_range, y_range, resolution
            )

            # Map phase values to colors
            for i in range(phases.shape[0]):
                for j in range(phases.shape[1]):
                    if phases[i, j] == 1:  # stable
                        phase_colors[i, j] = [
                            0.643,
                            0.325,
                            0.831,
                            0.5,
                        ]  # Purple with alpha
                    elif phases[i, j] == 2:  # descent
                        phase_colors[i, j] = [
                            0.369,
                            0.580,
                            0.808,
                            0.5,
                        ]  # Blue with alpha
                    elif phases[i, j] == 3:  # rise
                        phase_colors[i, j] = [
                            0.682,
                            0.682,
                            0.682,
                            0.5,
                        ]  # Gray with alpha
                    else:
                        phase_colors[i, j] = [0, 0, 0, 0]  # Transparent

            # Get coordinates for interpolation
            x_phase = np.linspace(x_range[0], x_range[1], phases.shape[1])
            y_phase = np.linspace(y_range[0], y_range[1], phases.shape[0])
            x_lic = np.linspace(x_range[0], x_range[1], resolution)
            y_lic = np.linspace(y_range[0], y_range[1], resolution)

            # Create grid points for evaluation
            X_lic, Y_lic = np.meshgrid(x_lic, y_lic)
            points_lic = np.vstack((X_lic.flatten(), Y_lic.flatten())).T

            # Interpolate each color channel
            for c in range(4):
                # Create interpolator for this color channel
                interpolator = RegularGridInterpolator(
                    (x_phase, y_phase),
                    phase_colors[
                        :, :, c
                    ].T,  # Note: transpose needed for correct orientation
                    bounds_error=False,
                    fill_value=0,
                )

                # Apply interpolation and reshape to grid
                phase_colors_high_res[:, :, c] = interpolator(points_lic).reshape(
                    resolution, resolution
                )

            # Overlay the phase colors on the LIC plot
            self.ax.imshow(
                phase_colors_high_res,
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                origin="lower",
                interpolation="bicubic",
            )

        return img

    def visualize(
        self,
        t_values,
        x_pred,
        y_pred,
        x_true,
        y_true,
        vector_field_type="none",  # Options: "arrows", "lic", "streamlines"
        pinn_model=None,
        vector_field_t=0.0,
        # Arrow vector field options
        arrow_grid_size=12,
        arrow_skip=None,
        arrow_scale=8,
        arrow_width=0.003,
        arrow_color="darkblue",
        arrow_alpha=0.7,
        # LIC options
        lic_resolution=200,
        lic_cmap="gray",
        lic_alpha=0.8,
        lic_color_by_phase=False,
        lic_phase_method="cluster",  # "rule" or "cluster"
        # Streamline options
        stream_density=1.0,
        stream_linewidth=1.0,
        stream_color="darkblue",
        stream_arrowsize=1.2,
        x_range=(-3, 4),
        y_range=(-1, 7),
    ):
        t_10s_idx = np.argmin(np.abs(t_values - 10.0))

        # Add vector field visualization if requested and if we have a PINN model
        if pinn_model is not None:
            if vector_field_type.lower() == "lic":
                # Use Line Integral Convolution
                self.plot_lic(
                    pinn_model,
                    t=vector_field_t,
                    x_range=x_range,
                    y_range=y_range,
                    resolution=lic_resolution,
                    cmap=lic_cmap,
                    alpha=lic_alpha,
                    color_by_phase=lic_color_by_phase,
                    phase_method=lic_phase_method,
                )
            elif vector_field_type.lower() == "arrows":
                # Use arrow-based vector field
                self.plot_vector_field(pinn_model, x0=2.0, y0=1.5, t_max=15)
            elif vector_field_type.lower() == "streamlines":
                # Use streamline visualization
                self.plot_streamlines(
                    pinn_model,
                    t=vector_field_t,
                    x_range=x_range,
                    y_range=y_range,
                    density=stream_density,
                    linewidth=stream_linewidth,
                    color=stream_color,
                    arrowsize=stream_arrowsize,
                )

        # Plot trajectories
        self.plot_trajectories(x_pred, y_pred, x_true, y_true, t_10s_idx)

        return self.ax
