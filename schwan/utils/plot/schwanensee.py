import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
from matplotlib.colors import Normalize
import cv2
import tempfile
import os
from .phase import PhaseIdentifier
from scipy.interpolate import RegularGridInterpolator
import scipy
from sklearn.cluster import DBSCAN
import matplotlib


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
        self.phases_by_priority = ["rise", "descent", "stable"]
        self.phase_identifier = PhaseIdentifier()

    def plot_trajectories(self, x_pred, y_pred, x_true, y_true, t_10s_idx):
        """
        Plot the predicted and true trajectories.
        """
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
        self.ax.plot(x_true, y_true, "b-", linewidth=2, label="True Solution")
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
        num_points_x = 10
        num_points_y = 10
        x_vals = np.linspace(0, 3, num_points_x)
        y_vals = np.linspace(0, 2, num_points_y)
        start_points = [(xx, yy) for xx in x_vals for yy in y_vals]

        x_min, x_max = x_range
        y_min, y_max = y_range

        for start_x, start_y in start_points:

            def learned_rhs_func(t, state):
                x_val, y_val = float(state[0]), float(state[1])
                inp = np.array([[t, x_val, y_val]], dtype=np.float32)
                pred = pinn_model.predict(inp)
                return [float(pred[0, 2]), float(pred[0, 3])]

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
            mask = (
                (x_traj >= x_min)
                & (x_traj <= x_max)
                & (y_traj >= y_min)
                & (y_traj <= y_max)
            )
            if not np.any(mask):
                continue

            x_traj = x_traj[mask]
            y_traj = y_traj[mask]
            t_eval_masked = t_eval[mask]

            inputs = np.column_stack([t_eval_masked, x_traj, y_traj])
            preds = pinn_model.predict(inputs)
            dx_dt = preds[:, 2]
            dy_dt = preds[:, 3]

            mag = np.sqrt(dx_dt**2 + dy_dt**2)
            mag[mag == 0] = 1.0
            dx_dt /= mag
            dy_dt /= mag

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

    def plot_lic(
        self,
        pinn_model,
        t=0.0,
        x_range=(-3, 4),
        y_range=(-1, 7),
        grid_density=200,
        t_max=15.0,
        num_points=500,
        kernel_length=20,
        noise_amp=400,
        cmap="Greys_r",
        resolution=200,
        color_by_phase=True,
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
        # Step 1: Generate vector field grid for LIC
        x_vals = np.linspace(x_range[0], x_range[1], grid_density)
        y_vals = np.linspace(y_range[0], y_range[1], grid_density)
        X, Y = np.meshgrid(x_vals, y_vals)
        inputs = np.column_stack([np.full(X.size, 0.0), X.ravel(), Y.ravel()])

        # Get model predictions for the grid
        preds = pinn_model.predict(inputs)
        U = preds[:, 2].reshape(X.shape)
        V = preds[:, 3].reshape(X.shape)

        # Normalize for LIC
        mag = np.sqrt(U**2 + V**2)
        U_norm = U / (mag + 1e-8)
        V_norm = V / (mag + 1e-8)

        # Step 2: Create density map from trajectory paths
        density_map = np.zeros_like(X)

        # Generate evenly spaced starting points
        num_points_x = 10
        num_points_y = 10

        start_x_vals = np.linspace(0, 3, num_points_x)
        start_y_vals = np.linspace(0, 2, num_points_y)

        # Define allowed plotting region
        x_min, x_max = x_range
        y_min, y_max = y_range

        for start_x in start_x_vals:
            for start_y in start_y_vals:
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

                # Keep only points inside allowed region
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

                # Map trajectory to grid cells and increment density
                xi = np.searchsorted(x_vals, x_traj)
                yi = np.searchsorted(y_vals, y_traj)

                # Clip indices to prevent out-of-bounds errors
                xi = np.clip(xi, 0, grid_density - 1)
                yi = np.clip(yi, 0, grid_density - 1)

                # Add to density map - weighted by time to emphasize steady states
                for i in range(len(xi)):
                    density_map[yi[i], xi[i]] += 1 + i / len(
                        xi
                    )  # More weight to later points

        # Normalize density
        density_map = density_map / (density_map.max() + 1e-8)

        # Create mask
        levels = [0.0, 0.1, 0.3, 0.7, 1.0]
        thresholds = [0.01, 0.1, 0.3, 0.7]
        mask = np.zeros_like(density_map)

        for i in range(len(thresholds)):
            mask[density_map >= thresholds[i]] = levels[i + 1]

        # Generate white noise texture
        noise = np.random.rand(*X.shape) * noise_amp

        # LIC calculation
        def lic_texture(U, V, noise, length):
            tex = np.zeros_like(noise)
            for j in range(noise.shape[0]):
                for i in range(noise.shape[1]):
                    vals = []
                    for s in range(-length // 2, length // 2):
                        xi = int(i + s * U[j, i])
                        yj = int(j + s * V[j, i])
                        if 0 <= xi < noise.shape[1] and 0 <= yj < noise.shape[0]:
                            vals.append(noise[yj, xi])
                    if vals:
                        tex[j, i] = np.mean(vals)
            return tex

        lic_img = lic_texture(U_norm, V_norm, noise, kernel_length)

        # Apply mask
        lic_weighted = lic_img * mask

        # Plot
        self.ax.imshow(
            lic_weighted,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin="lower",
            cmap=cmap,
            alpha=0.7,
            norm=Normalize(vmin=0, vmax=lic_weighted.max()),
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Reduced LIC of PINN Vector Field")

        # Add phase coloring if requested
        if color_by_phase:
            # Generate phase classification
            if phase_method.lower() == "cluster":
                phases, X_phase, Y_phase, field = (
                    self.phase_identifier.cluster_based_classification(
                        pinn_model, t, x_range, y_range, grid_size=50, n_clusters=3
                    )
                )
            else:  # Default to rule-based
                phases, X_phase, Y_phase, field = (
                    self.phase_identifier.rule_based_classification(
                        pinn_model, t, x_range, y_range, grid_size=50
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

    def highlight_circular_flow_from_density(
        self,
        pinn_model,
        x_range=(-3, 4),
        y_range=(-1, 7),
        blur_radius=5,
        white_threshold=0.98,
        morph_kernel_size=3,
        min_area=30,
        color="orange",
    ):
        """
        Detect the bright-white elliptical loop from the arrow density map and plot it.

        Uses near-maximum intensity pixels to isolate the core loop region.
        """

        # 1. Create arrow density image (from plot_arrow_density_imaging)
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        ax.axis("off")
        tmp_vis = SchwanenseeVisualizer(ax=ax)
        tmp_vis.plot_vector_field(pinn_model, x_range=x_range, y_range=y_range)
        plt.axis("off")

        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmpfile.name
        plt.savefig(
            tmp_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=False
        )
        plt.close(fig)

        img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        os.unlink(tmp_path)

        # 2. Invert if needed
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        # 3. Blur & normalize
        blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
        norm_img = blurred.astype(np.float32) / 255.0

        # 4. White threshold mask (near-maximum intensity)
        max_val = norm_img.max()
        mask = (norm_img >= white_threshold * max_val).astype(np.uint8) * 255

        # 5. Morphological closing to fill gaps
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 6. Remove small specks by keeping largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No bright-white loop found.")
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < min_area:
            print("White region too small.")
            return None

        # 7. Fit ellipse to the white region
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (width, height), angle = ellipse

            img_h, img_w = norm_img.shape
            cx_data = x_range[0] + (cx / img_w) * (x_range[1] - x_range[0])
            cy_data = y_range[0] + ((img_h - cy) / img_h) * (y_range[1] - y_range[0])
            width_data = (width / img_w) * (x_range[1] - x_range[0])
            height_data = (height / img_h) * (y_range[1] - y_range[0])

            # 8. Draw ellipse
            ellipse_patch = matplotlib.patches.Ellipse(
                (cx_data, cy_data),
                width=width_data,
                height=height_data,
                angle=-angle,
                edgecolor=color,
                facecolor="none",
                linewidth=2,
                label="Stable Oscillation",
            )
            self.ax.add_patch(ellipse_patch)
            self.ax.legend()

            # 9. Return ellipse parameters for later analysis
            return {
                "center": (cx_data, cy_data),
                "width": width_data,
                "height": height_data,
                "angle": -angle,
            }
        else:
            print("Not enough points to fit ellipse.")
            return None

    def plot_arrow_density_imaging(
        self,
        pinn_model,
        x_range=(-3, 4),
        y_range=(-1, 7),
        blur_radius=1,
        cmap="hot",
        density_threshold=0.7,  # filter low-density regions
    ):
        """
        Create a heatmap of arrow density by rendering the vector field
        and processing it as an image.

        Physical insight:
        -
        """

        # Temporary figure for quiver
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        ax.set_facecolor("white")
        ax.axis("off")

        tmp_vis = SchwanenseeVisualizer(ax=ax)
        tmp_vis.plot_vector_field(pinn_model, x_range=x_range, y_range=y_range)
        plt.axis("off")

        # Save to temp file
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmpfile.name
        plt.savefig(
            tmp_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=False
        )
        plt.close(fig)

        # Read image
        img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)

        # Invert if background is white
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        # Flip vertically to match coordinate orientation
        img = cv2.flip(img, 0)

        # Blur to spread intensity
        blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)

        # Normalize to 0–1
        density_map = blurred.astype(np.float32) / 255.0

        # Filter out low-density regions
        density_map[density_map < density_threshold] = 0.0

        # Plot on current axis
        self.ax.imshow(
            density_map,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin="lower",
            cmap=cmap,
            alpha=0.8,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",  # prevents forced square image
        )

        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

    def visualize(
        self,
        t_values,
        x_pred,
        y_pred,
        x_true,
        y_true,
        vector_field_type="none",  # Options: "arrows", "lic"
        pinn_model=None,
        vector_field_t=0.0,
        # LIC options
        lic_resolution=200,
        lic_cmap="gray",
        lic_color_by_phase=False,
        lic_phase_method="cluster",  # "rule" or "cluster"
        x_range=(-3, 4),
        y_range=(-1, 7),
        show_trajectories=True,
    ):
        """
        Comprehensive visualization method combining vector field and trajectories.
        """
        t_10s_idx = np.argmin(np.abs(t_values - 10.0))

        # Add vector field visualization if requested and if we have a PINN model
        if pinn_model is not None:
            if vector_field_type.lower() == "density":
                # Use Line Integral Convolution
                self.plot_arrow_density_imaging(
                    pinn_model, x_range=x_range, y_range=y_range, cmap=lic_cmap
                )
            elif vector_field_type.lower() == "arrows":
                # Use arrow-based vector field
                self.plot_vector_field(pinn_model, x_range=x_range, y_range=y_range)
            elif vector_field_type.lower() == "lic":
                # Use arrow-based vector field
                self.plot_lic(
                    pinn_model,
                    x_range=x_range,
                    y_range=y_range,
                    color_by_phase=lic_color_by_phase,
                    phase_method=lic_phase_method,
                )

        # Plot trajectories if requested
        if show_trajectories:
            self.plot_trajectories(x_pred, y_pred, x_true, y_true, t_10s_idx)

        # Set labels and title
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)

        return self.ax
