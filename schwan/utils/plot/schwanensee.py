import os
from dataclasses import dataclass
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scipy.integrate
import onnxruntime as ort

from .phase import PhaseIdentifier
from .flow_analysis import FlowAnalyzer


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
    def __init__(self, ax=None, onnx_model=None):
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

        self.flow_analyzer = FlowAnalyzer()

        # Vector-field caches
        self._vf_cache = {}
        self._density_cache = {}
        self.onnx_model = onnx_model

        # Initialize ONNX session
        if onnx_model is None:
            raise ValueError("Please provide a path to an ONNX model")
        self.onnx_session = ort.InferenceSession(
            onnx_model,
            providers=["CPUExecutionProvider"],
        )
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_output_names = [o.name for o in self.onnx_session.get_outputs()]

    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run ONNX model inference.
        Assumes input shape: (N, 3) with columns [t, x, y]
        and output with at least 4 columns where [:,2] = dx/dt, [:,3] = dy/dt.
        """
        outputs = self.onnx_session.run(
            self.onnx_output_names, {self.onnx_input_name: inputs}
        )
        # take first output if multiple
        pred = outputs[0]
        return pred

    def prepare_vector_field(
        self,
        x_range=(-2, 4),
        y_range=(0, 2),
        t_max=15.0,
        num_points=500,
        num_points_x=10,
        num_points_y=10,
    ):
        key = _VFParams(
            id(self.onnx_session),
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
            return key

        x_min, x_max = x_range
        y_min, y_max = y_range

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
            pred = self._predict(inp)
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

            inputs = np.column_stack([t_masked, x_traj, y_traj]).astype(np.float32)
            preds = self._predict(inputs)
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
            self._vf_cache[key] = dict(
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
            return key

        X, Y, T = np.concatenate(all_X), np.concatenate(all_Y), np.concatenate(all_T)
        VX, VY = np.concatenate(all_VX), np.concatenate(all_VY)

        self._vf_cache[key] = dict(
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
        return key

    def _ensure_vf(self, **kwargs):
        return self.prepare_vector_field(**kwargs)

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
        tmp_vis = SchwanenseeVisualizer(ax=ax, onnx_model=self.onnx_model)
        tmp_vis.plot_vector_field(
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

    def highlight_circular_flow(
        self,
        x_range=(-3, 4),
        y_range=(-1, 7),
        blur_radius=5,
        white_threshold=0.98,
        morph_kernel_size=3,
        min_area=30,
        loop_scale_factor=1.1,
        n_segments=150,
        max_angle_deviation=5.0,
        dist_tol=0.5,
        draw_normals=False,
        vector_length=0.3,
    ):
        """
        Detect circular flow patterns and highlight inflow/outflow regions.
        """
        # Ensure vector field data
        key = self._ensure_vf(
            x_range=x_range,
            y_range=y_range,
            t_max=15.0,
            num_points=500,
            num_points_x=10,
            num_points_y=10,
        )

        vector_field_data = self._vf_cache[key]

        # Step 1: Detect the oscillation
        ellipse = self.flow_analyzer.detect_oscillation(
            vector_field_data,
            x_range=x_range,
            y_range=y_range,
            blur_radius=blur_radius,
            white_threshold=white_threshold,
            morph_kernel_size=morph_kernel_size,
            min_area=min_area,
            loop_scale_factor=loop_scale_factor,
        )

        if ellipse is None:
            return None

        # Step 2: Calculate ellipse segments
        ellipse = self.flow_analyzer.calculate_ellipse_segments(
            ellipse, n_segments=n_segments
        )

        # Step 3: Analyze flow at each segment
        flow_segments = self.flow_analyzer.analyze_flow_segments(
            ellipse,
            vector_field_data,
            max_angle_deviation=max_angle_deviation,
            dist_tol=dist_tol,
        )

        # Step 4: Draw the flow analysis
        legend_handles = self.flow_analyzer.draw_flow_analysis(
            self.ax,
            ellipse,
            flow_segments,
            draw_normals=draw_normals,
            vector_length=vector_length,
        )

        self.ax.legend(handles=legend_handles)

        return {
            "center": ellipse.center,
            "width": ellipse.width,
            "height": ellipse.height,
            "angle": ellipse.angle,
            "flow_segments": flow_segments,
        }

    def visualize(
        self,
        t_values,
        x_pred,
        y_pred,
        x_true,
        y_true,
        vector_field_type="none",
        lic_cmap="gray",
        x_range=(-3, 4),
        y_range=(-1, 7),
        show_trajectories=True,
        t_max=15.0,
        num_points=500,
        num_points_x=10,
        num_points_y=10,
    ):
        # Ensure vector field data if needed
        if vector_field_type.lower() in {"density", "arrows"}:
            self._ensure_vf(
                x_range=x_range,
                y_range=y_range,
                t_max=t_max,
                num_points=num_points,
                num_points_x=num_points_x,
                num_points_y=num_points_y,
            )

        if vector_field_type.lower() == "density":
            self.plot_arrow_density_imaging(
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
