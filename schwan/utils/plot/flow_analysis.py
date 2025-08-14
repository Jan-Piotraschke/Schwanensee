import numpy as np
import cv2
import tempfile
import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class EllipseData:
    """Data class for storing fitted ellipse properties"""

    center: tuple
    width: float
    height: float
    angle: float
    segments: list = None  # Will store segment points and their normals


class FlowAnalyzer:
    """Class dedicated to analyzing vector field flow patterns around detected loops"""

    def __init__(self):
        self.flow_types = {"outflow": "red", "inflow": "blue", "stable": "orange"}

    def detect_oscillation(
        self,
        vector_field_data,
        x_range,
        y_range,
        blur_radius=5,
        white_threshold=0.98,
        morph_kernel_size=3,
        min_area=30,
        loop_scale_factor=1.1,
    ):
        """
        Detect circular flow pattern from vector field density image
        """
        # Extract vector field data
        X, Y = vector_field_data.get("X", []), vector_field_data.get("Y", [])

        if len(X) == 0:
            print("Vector field data is empty.")
            return None

        # Render temp vector field image for contour detection
        fig_tmp, ax_tmp = plt.subplots(figsize=(6, 6), facecolor="white")
        ax_tmp.axis("off")

        # Draw vector field
        ax_tmp.quiver(
            X,
            Y,
            vector_field_data.get("VX", []),
            vector_field_data.get("VY", []),
            angles="xy",
            scale_units="xy",
            scale=20,
            width=0.003,
            color="darkblue",
            alpha=0.6,
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
        if len(largest_contour) < 5:
            print("Not enough points to fit ellipse.")
            return None

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

        print(
            f"Ellipse center: ({cx_data:.2f}, {cy_data:.2f}), "
            f"size: {width_data:.2f}x{height_data:.2f} (after scaling)"
        )

        return EllipseData(
            center=(cx_data, cy_data),
            width=width_data,
            height=height_data,
            angle=-angle,
        )

    def calculate_ellipse_segments(self, ellipse, n_segments=150):
        """
        Calculate points and normals along the ellipse perimeter
        """
        cx_data, cy_data = ellipse.center
        width_data, height_data = ellipse.width, ellipse.height
        rot_rad = np.deg2rad(ellipse.angle)

        a = width_data / 2
        b = height_data / 2

        # Calculate segment points and normals
        segment_points = []
        theta_vals = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)

        for t in theta_vals:
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

        ellipse.segments = segment_points
        return ellipse

    def analyze_flow_segments(
        self, ellipse, vector_field_data, max_angle_deviation=5.0, dist_tol=0.5
    ):
        """
        Analyze vector field to determine flow direction at each ellipse segment
        """
        if not ellipse.segments:
            raise ValueError("Ellipse segments must be calculated first")

        X, Y = vector_field_data.get("X", []), vector_field_data.get("Y", [])
        VX, VY = vector_field_data.get("VX", []), vector_field_data.get("VY", [])

        if len(X) == 0:
            print("Vector field data is empty.")
            return {}

        # Convert max angle deviation to dot product threshold
        min_flow_dot = np.cos(np.radians(max_angle_deviation))
        print(
            f"Using min dot product of {min_flow_dot:.6f} "
            f"(corresponds to {max_angle_deviation}° max deviation)"
        )

        outflow_segments = set()
        inflow_segments = set()

        # Analyze each vector field point
        for i in range(len(X)):
            x_vec, y_vec = X[i], Y[i]
            vx, vy = VX[i], VY[i]

            # Find closest segment
            closest_seg_idx = -1
            min_dist_seg = float("inf")

            for j, (x_seg, y_seg, _, _) in enumerate(ellipse.segments):
                dist = np.sqrt((x_vec - x_seg) ** 2 + (y_vec - y_seg) ** 2)
                if dist < dist_tol and dist < min_dist_seg:
                    min_dist_seg = dist
                    closest_seg_idx = j

            if closest_seg_idx >= 0:
                _, _, nx_seg, ny_seg = ellipse.segments[closest_seg_idx]

                # Check alignment with normal vector (outflow)
                dot_outflow = vx * nx_seg + vy * ny_seg

                # Check alignment with opposite normal vector (inflow)
                dot_inflow = vx * (-nx_seg) + vy * (-ny_seg)

                if dot_outflow > min_flow_dot:
                    outflow_segments.add(closest_seg_idx)
                    if len(outflow_segments) <= 3:  # Debug first few
                        angle_deg = np.degrees(np.arccos(dot_outflow))
                        print(
                            f"Outflow at segment {closest_seg_idx}, "
                            f"angle deviation: {angle_deg:.2f}°"
                        )

                elif dot_inflow > min_flow_dot:
                    inflow_segments.add(closest_seg_idx)
                    if len(inflow_segments) <= 3:  # Debug first few
                        angle_deg = np.degrees(np.arccos(dot_inflow))

        print(
            f"Detected {len(outflow_segments)} outflow segments and "
            f"{len(inflow_segments)} inflow segments out of {len(ellipse.segments)} total segments."
        )

        return {
            "outflow": outflow_segments,
            "inflow": inflow_segments,
            "segments_total": len(ellipse.segments),
        }

    def draw_flow_analysis(
        self,
        ax,
        ellipse,
        flow_segments,
        colors=None,
        line_width=4,
        draw_normals=False,
        vector_length=0.3,
    ):
        """
        Draw the ellipse with segments colored according to flow direction
        """
        if not ellipse.segments:
            raise ValueError("Ellipse segments must be calculated first")

        if colors is None:
            colors = self.flow_types

        outflow_segments = flow_segments.get("outflow", set())
        inflow_segments = flow_segments.get("inflow", set())

        # Draw each segment
        n_segments = len(ellipse.segments)
        for i in range(n_segments):
            # Get segment endpoints
            x1, y1, nx1, ny1 = ellipse.segments[i]
            next_idx = (i + 1) % n_segments
            x2, y2, _, _ = ellipse.segments[next_idx]

            # Determine segment color
            if i in outflow_segments:
                seg_color = colors["outflow"]
            elif i in inflow_segments:
                seg_color = colors["inflow"]
            else:
                seg_color = colors["stable"]

            # Draw segment
            ax.plot([x1, x2], [y1, y2], color=seg_color, linewidth=line_width)

            # Draw normal vectors if requested
            if draw_normals and i % 10 == 0:  # Draw every 10th normal for clarity
                ax.arrow(
                    x1,
                    y1,
                    nx1 * vector_length,
                    ny1 * vector_length,
                    head_width=0.05,
                    head_length=0.08,
                    fc=seg_color,
                    ec=seg_color,
                )

        legend_handles = [
            mlines.Line2D(
                [], [], color=colors["outflow"], linewidth=2, label="Outflow Region"
            ),
            mlines.Line2D(
                [], [], color=colors["inflow"], linewidth=2, label="Inflow Region"
            ),
            mlines.Line2D(
                [], [], color=colors["stable"], linewidth=2, label="Stable Region"
            ),
        ]

        return legend_handles
