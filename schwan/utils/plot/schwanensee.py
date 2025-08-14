import os
from dataclasses import dataclass
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import scipy.integrate

from .phase import PhaseIdentifier


@dataclass(frozen=True)
class _VFParams:
    model_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    num_points_x: int
    num_points_y: int
    t_max: float
    num_points: int


class SchwanenseeVisualizer:
    def __init__(self, ax=None):
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        else:
            self.ax = ax

        self.area_colors = {
            "stable": "#A453D4",
            "descent": "#5E94CE",
            "rise": "#AEAEAE",
        }
        self.phases_by_priority = ["rise", "descent", "stable"]
        self.phase_identifier = PhaseIdentifier()

        # Vector-field caches
        self._vf_cache = {}  # key: _VFParams -> dict(data)
        self._density_cache = {}  # key: (_VFParams, res_x, res_y, blur_radius, density_threshold) -> density_map

    def prepare_vector_field(
        self,
        pinn_model,
        x_range=(-2, 4),
        y_range=(0, 2),
        t_max=15.0,
        num_points=500,
        num_points_x=10,
        num_points_y=10,
    ):
        """
        Integrate the learned ODE from a seed grid ONCE and cache:
          - concatenated positions X, Y, times T
          - normalized velocities VX, VY
          - seed grid and params for reuse by other functions
        """
        key = _VFParams(
            id(pinn_model),
            x_range[0],
            x_range[1],
            y_range[0],
            y_range[1],
            num_points_x,
            num_points_y,
            t_max,
            num_points,
        )
        if key in self._vf_cache:
            return key  # already prepared

        x_min, x_max = x_range
        y_min, y_max = y_range

        # seed points
        x_vals = np.linspace(0, 3, num_points_x)
        y_vals = np.linspace(0, 2, num_points_y)
        start_points = [(xx, yy) for xx in x_vals for yy in y_vals]

        all_X = []
        all_Y = []
        all_T = []
        all_VX = []
        all_VY = []

        def learned_rhs_func(t, state):
            x_val, y_val = float(state[0]), float(state[1])
            inp = np.array([[t, x_val, y_val]], dtype=np.float32)
            pred = pinn_model.predict(inp)
            # pred columns assumed: [?, ?, dx, dy] as in your original code
            return [float(pred[0, 2]), float(pred[0, 3])]

        t_eval = np.linspace(0, t_max, num_points)

        for start_x, start_y in start_points:
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

            # mask to plotting window
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
            t_masked = t_eval[mask]

            # velocities at those points
            inputs = np.column_stack([t_masked, x_traj, y_traj]).astype(np.float32)
            preds = pinn_model.predict(inputs)
            dx_dt = preds[:, 2]
            dy_dt = preds[:, 3]
            mag = np.sqrt(dx_dt**2 + dy_dt**2)
            mag[mag == 0] = 1.0
            dx_dt /= mag
            dy_dt /= mag

            all_X.append(x_traj)
            all_Y.append(y_traj)
            all_T.append(t_masked)
            all_VX.append(dx_dt)
            all_VY.append(dy_dt)

        if len(all_X) == 0:
            # No trajectories intersect the window; cache an empty set to avoid recomputation
            data = dict(
                X=np.array([]),
                Y=np.array([]),
                T=np.array([]),
                VX=np.array([]),
                VY=np.array([]),
                x_range=x_range,
                y_range=y_range,
                t_max=t_max,
                num_points=num_points,
                seeds=(num_points_x, num_points_y),
            )
            self._vf_cache[key] = data
            return key

        X = np.concatenate(all_X)
        Y = np.concatenate(all_Y)
        T = np.concatenate(all_T)
        VX = np.concatenate(all_VX)
        VY = np.concatenate(all_VY)

        data = dict(
            X=X,
            Y=Y,
            T=T,
            VX=VX,
            VY=VY,
            x_range=x_range,
            y_range=y_range,
            t_max=t_max,
            num_points=num_points,
            seeds=(num_points_x, num_points_y),
        )
        self._vf_cache[key] = data
        return key

    def _ensure_vf(self, pinn_model, **kwargs):
        """Ensure vector-field data exist; returns the cache key."""
        return self.prepare_vector_field(pinn_model, **kwargs)

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
        t_max=15.0,
        num_points=500,
        arrow_scale=0.05,
        arrow_width=0.003,
        color="darkblue",
        alpha=0.6,
        num_points_x=10,
        num_points_y=10,
    ):
        """
        Plot vector field arrows from cached simulation data.

        Physical insight:
        - indication of state change speed and its acceleration / deceleration
        - spotting critical points as they appear with very short arrows
        """
        key = self._ensure_vf(
            pinn_model,
            x_range=x_range,
            y_range=y_range,
            t_max=t_max,
            num_points=num_points,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
        )
        data = self._vf_cache[key]
        X, Y, VX, VY = data["X"], data["Y"], data["VX"], data["VY"]
        if X.size == 0:
            return

        self.ax.quiver(
            X,
            Y,
            VX,
            VY,
            angles="xy",
            scale_units="xy",
            scale=1 / arrow_scale,
            width=arrow_width,
            color=color,
            alpha=alpha,
        )

    def _get_density_map(
        self,
        key: _VFParams,
        res_x=600,
        res_y=600,
        blur_radius=1,
        density_threshold=0.0,
    ):
        cache_key = (key, res_x, res_y, int(blur_radius), float(density_threshold))
        if cache_key in self._density_cache:
            return self._density_cache[cache_key]

        data = self._vf_cache[key]
        X, Y = data["X"], data["Y"]
        x_min, x_max = data["x_range"]
        y_min, y_max = data["y_range"]

        if X.size == 0:
            density = np.zeros((res_y, res_x), dtype=np.float32)
            self._density_cache[cache_key] = density
            return density

        # 2D histogram (point density of trajectory samples)
        H, _, _ = np.histogram2d(
            X,
            Y,
            bins=[res_x, res_y],
            range=[[x_min, x_max], [y_min, y_max]],
        )
        # histogram2d returns shape (res_x, res_y); we want [row, col] = [y, x]
        density = H.T.astype(np.float32)

        if blur_radius and blur_radius > 1:
            # GaussianBlur expects odd kernel size; derive a reasonable kernel
            k = max(3, int(blur_radius) | 1)  # make it odd
            density = cv2.GaussianBlur(density, (k, k), 0)

        # normalize 0..1
        max_val = density.max() if density.size else 0.0
        if max_val > 0:
            density = density / max_val
        if density_threshold > 0:
            density = np.where(density >= density_threshold, density, 0.0)

        self._density_cache[cache_key] = density
        return density

    def plot_arrow_density_imaging(
        self,
        pinn_model,
        x_range=(-3, 4),
        y_range=(-1, 7),
        blur_radius=1,
        cmap="hot",
        density_threshold=0.7,  # filter low-density regions
        t_max=15.0,
        num_points=500,
        num_points_x=10,
        num_points_y=10,
    ):
        """
        Create a heatmap of arrow density by rendering the vector field
        and processing it as an image.

        Physical insight:
        -
        """
        # Ensure we have cached vector field
        self._ensure_vf(
            pinn_model,
            x_range=x_range,
            y_range=y_range,
            t_max=t_max,
            num_points=num_points,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
        )

        # Render to temp figure (like old version)
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        ax.set_facecolor("white")
        ax.axis("off")
        tmp_vis = SchwanenseeVisualizer(ax=ax)
        tmp_vis.plot_vector_field(
            pinn_model,
            x_range=x_range,
            y_range=y_range,
            t_max=t_max,
            num_points=num_points,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
        )
        plt.axis("off")

        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmpfile.name
        plt.savefig(
            tmp_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=False
        )
        plt.close(fig)

        # Read back in grayscale
        img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        try:
            os.unlink(tmp_path)
        except:
            pass

        # Invert if arrows are dark on white
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        # Flip vertically to match plot coordinates
        img = cv2.flip(img, 0)

        # Blur and normalize
        blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
        density_map = blurred.astype(np.float32) / 255.0
        density_map[density_map < density_threshold] = 0.0

        # Overlay on current axes
        self.ax.imshow(
            density_map,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            origin="lower",
            cmap=cmap,
            alpha=0.8,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
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
        outflow_color="red",
        inflow_color="blue",
        draw_orthogonal_vectors=False,
        n_vectors=12,
        vector_length=0.3,
        t_max=15.0,
        num_points=500,
        num_points_x=10,
        num_points_y=10,
        n_segments=150,
        dist_tol=0.5,
        max_angle_deviation=5.0,  # Max deviation in degrees
        loop_scale_factor=1.1,
    ):
        """Detect the bright-white elliptical loop from the arrow density map, plot it,
        and color segments red for outflow, blue for inflow, or default color for stable regions.
        """
        # Convert max angle deviation to dot product threshold
        # cos(5°) ≈ 0.9962
        min_flow_dot = np.cos(np.radians(max_angle_deviation))

        # Ensure we have cached data
        key = self._ensure_vf(
            pinn_model,
            x_range=x_range,
            y_range=y_range,
            t_max=t_max,
            num_points=num_points,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
        )

        # Get vector field data directly
        data = self._vf_cache.get(key)
        if data is None:
            print("No vector field data in cache.")
            return None

        X, Y, VX, VY = data["X"], data["Y"], data["VX"], data["VY"]

        if len(X) == 0:
            print("Vector field data is empty.")
            return None

        print(f"Vector field contains {len(X)} points")
        print(
            f"Using min dot product of {min_flow_dot:.6f} (corresponds to {max_angle_deviation}° max deviation)"
        )
        print(f"Loop scale factor: {loop_scale_factor:.2f}")

        # Render temp vector field image for contour detection
        fig_tmp, ax_tmp = plt.subplots(figsize=(6, 6), facecolor="white")
        ax_tmp.axis("off")
        tmp_vis = SchwanenseeVisualizer(ax=ax_tmp)
        tmp_vis.plot_vector_field(
            pinn_model,
            x_range=x_range,
            y_range=y_range,
            t_max=t_max,
            num_points=num_points,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
        )
        plt.axis("off")
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmpfile.name
        plt.savefig(
            tmp_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=False
        )
        plt.close(fig_tmp)

        # Load grayscale image
        img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        try:
            os.unlink(tmp_path)
        except:
            pass
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        # Blur and threshold
        blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
        norm_img = blurred.astype(np.float32) / 255.0
        max_val = norm_img.max()
        mask = (norm_img >= white_threshold * max_val).astype(np.uint8) * 255
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No bright-white loop found.")
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < min_area:
            print("White region too small.")
            return None

        # Fit ellipse to the contour
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (width, height), angle = ellipse

            # Convert to data coordinates
            img_h, img_w = norm_img.shape
            cx_data = x_range[0] + (cx / img_w) * (x_range[1] - x_range[0])
            cy_data = y_range[0] + ((img_h - cy) / img_h) * (y_range[1] - y_range[0])
            width_data = (width / img_w) * (x_range[1] - x_range[0])
            height_data = (height / img_h) * (y_range[1] - y_range[0])

            # Apply the scaling factor to the ellipse dimensions
            width_data *= loop_scale_factor
            height_data *= loop_scale_factor

            a = width_data / 2
            b = height_data / 2
            rot_rad = np.deg2rad(-angle)

            print(
                f"Ellipse center: ({cx_data:.2f}, {cy_data:.2f}), size: {width_data:.2f}x{height_data:.2f} (after scaling)"
            )

            # Calculate orthogonal vectors first (these are the reference vectors)
            orthogonal_vectors = []
            theta_ortho = np.linspace(0, 2 * np.pi, n_vectors, endpoint=False)

            for t in theta_ortho:
                x_ell = a * np.cos(t)
                y_ell = b * np.sin(t)
                x_rot = cx_data + x_ell * np.cos(rot_rad) - y_ell * np.sin(rot_rad)
                y_rot = cy_data + x_ell * np.sin(rot_rad) + y_ell * np.cos(rot_rad)

                # Calculate outward normal vector
                nx_local = 2 * np.cos(t) / (a**2)
                ny_local = 2 * np.sin(t) / (b**2)

                # Normalize
                norm_len = np.sqrt(nx_local**2 + ny_local**2)
                nx_local /= norm_len
                ny_local /= norm_len

                # Rotate to align with ellipse orientation
                nx = nx_local * np.cos(rot_rad) - ny_local * np.sin(rot_rad)
                ny = nx_local * np.sin(rot_rad) + ny_local * np.cos(rot_rad)

                orthogonal_vectors.append((x_rot, y_rot, nx, ny))

            # Draw ellipse segments
            theta_vals = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
            segment_points = []

            # First calculate all segment points and their normals
            for i, t in enumerate(theta_vals):
                x_ell = a * np.cos(t)
                y_ell = b * np.sin(t)
                x_rot = cx_data + x_ell * np.cos(rot_rad) - y_ell * np.sin(rot_rad)
                y_rot = cy_data + x_ell * np.sin(rot_rad) + y_ell * np.cos(rot_rad)

                # Calculate normal vector at this segment point
                nx_local = 2 * np.cos(t) / (a**2)
                ny_local = 2 * np.sin(t) / (b**2)

                # Normalize
                norm_len = np.sqrt(nx_local**2 + ny_local**2)
                nx_local /= norm_len
                ny_local /= norm_len

                # Rotate to align with ellipse orientation
                nx = nx_local * np.cos(rot_rad) - ny_local * np.sin(rot_rad)
                ny = nx_local * np.sin(rot_rad) + ny_local * np.cos(rot_rad)

                segment_points.append((x_rot, y_rot, nx, ny))

            # Now check each vector field point against the orthogonal vectors
            outflow_segments = set()
            inflow_segments = set()

            for i in range(len(X)):
                x_vec, y_vec = X[i], Y[i]
                vx, vy = VX[i], VY[i]

                # Find the closest orthogonal vector
                closest_ortho_idx = -1
                min_dist_ortho = float("inf")

                for j, (x_ortho, y_ortho, nx_ortho, ny_ortho) in enumerate(
                    orthogonal_vectors
                ):
                    dist = np.sqrt((x_vec - x_ortho) ** 2 + (y_vec - y_ortho) ** 2)
                    if dist < dist_tol and dist < min_dist_ortho:
                        min_dist_ortho = dist
                        closest_ortho_idx = j

                # If we found a close orthogonal vector
                if closest_ortho_idx >= 0:
                    _, _, nx_ortho, ny_ortho = orthogonal_vectors[closest_ortho_idx]

                    # Check alignment with orthogonal vector (outflow)
                    dot_outflow = vx * nx_ortho + vy * ny_ortho

                    # Check alignment with opposite of orthogonal vector (inflow)
                    dot_inflow = vx * (-nx_ortho) + vy * (-ny_ortho)

                    # Find the closest segment to mark
                    closest_seg_idx = -1
                    min_dist_seg = float("inf")

                    for j, (x_seg, y_seg, _, _) in enumerate(segment_points):
                        dist = np.sqrt((x_vec - x_seg) ** 2 + (y_vec - y_seg) ** 2)
                        if dist < dist_tol and dist < min_dist_seg:
                            min_dist_seg = dist
                            closest_seg_idx = j

                    if closest_seg_idx >= 0:
                        if dot_outflow > min_flow_dot:
                            outflow_segments.add(closest_seg_idx)
                            if len(outflow_segments) <= 3:  # Debug first few
                                angle_deg = np.degrees(np.arccos(dot_outflow))
                                print(
                                    f"Outflow at segment {closest_seg_idx}, angle deviation: {angle_deg:.2f}°"
                                )
                        elif dot_inflow > min_flow_dot:
                            inflow_segments.add(closest_seg_idx)
                            if len(inflow_segments) <= 3:  # Debug first few
                                angle_deg = np.degrees(np.arccos(dot_inflow))
                                print(
                                    f"Inflow at segment {closest_seg_idx}, angle deviation: {angle_deg:.2f}°"
                                )

            # Now draw the segments
            outflow_count = 0
            inflow_count = 0

            for i, t in enumerate(theta_vals):
                # Get segment endpoints
                t_next = t + 2 * np.pi / n_segments
                x1, y1, _, _ = segment_points[i]

                # Calculate the next point
                x_ell2 = a * np.cos(t_next)
                y_ell2 = b * np.sin(t_next)
                x2 = cx_data + x_ell2 * np.cos(rot_rad) - y_ell2 * np.sin(rot_rad)
                y2 = cy_data + x_ell2 * np.sin(rot_rad) + y_ell2 * np.cos(rot_rad)

                # Determine segment color
                seg_color = color  # Default stable color
                if i in outflow_segments:
                    seg_color = outflow_color
                    outflow_count += 1
                elif i in inflow_segments:
                    seg_color = inflow_color
                    inflow_count += 1

                self.ax.plot([x1, x2], [y1, y2], color=seg_color, linewidth=4)

            print(
                f"Detected {outflow_count} outflow segments and {inflow_count} inflow segments out of {n_segments} total segments."
            )

            # Draw orthogonal vectors if requested
            if draw_orthogonal_vectors:
                for x_ortho, y_ortho, nx_ortho, ny_ortho in orthogonal_vectors:
                    self.ax.arrow(
                        x_ortho,
                        y_ortho,
                        nx_ortho * vector_length,
                        ny_ortho * vector_length,
                        head_width=0.05,
                        head_length=0.08,
                        fc=color,
                        ec=color,
                    )

            # Add legend
            outflow_line = mlines.Line2D(
                [], [], color=outflow_color, linewidth=2, label="Outflow Region"
            )
            inflow_line = mlines.Line2D(
                [], [], color=inflow_color, linewidth=2, label="Inflow Region"
            )
            normal_line = mlines.Line2D(
                [], [], color=color, linewidth=2, label="Stable Region"
            )
            self.ax.legend(handles=[outflow_line, inflow_line, normal_line])

            return {
                "center": (cx_data, cy_data),
                "width": width_data,
                "height": height_data,
                "angle": -angle,
            }
        else:
            print("Not enough points to fit ellipse.")
            return None

    def visualize(
        self,
        t_values,
        x_pred,
        y_pred,
        x_true,
        y_true,
        vector_field_type="none",
        pinn_model=None,
        lic_cmap="gray",
        x_range=(-3, 4),
        y_range=(-1, 7),
        show_trajectories=True,
        t_max=15.0,
        num_points=500,
        num_points_x=10,
        num_points_y=10,
    ):
        # ensure vector field data if needed
        if pinn_model is not None and vector_field_type.lower() in {
            "density",
            "arrows",
        }:
            self._ensure_vf(
                pinn_model,
                x_range=x_range,
                y_range=y_range,
                t_max=t_max,
                num_points=num_points,
                num_points_x=num_points_x,
                num_points_y=num_points_y,
            )

        if pinn_model is not None:
            if vector_field_type.lower() == "density":
                self.plot_arrow_density_imaging(
                    pinn_model,
                    x_range=x_range,
                    y_range=y_range,
                    cmap=lic_cmap,
                    t_max=t_max,
                    num_points=num_points,
                    num_points_x=num_points_x,
                    num_points_y=num_points_y,
                )
            elif vector_field_type.lower() == "arrows":
                self.plot_vector_field(
                    pinn_model,
                    x_range=x_range,
                    y_range=y_range,
                    t_max=t_max,
                    num_points=num_points,
                    num_points_x=num_points_x,
                    num_points_y=num_points_y,
                )

        if show_trajectories:
            t_10s_idx = np.argmin(np.abs(t_values - 10.0))
            self.plot_trajectories(x_pred, y_pred, x_true, y_true, t_10s_idx)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        return self.ax
