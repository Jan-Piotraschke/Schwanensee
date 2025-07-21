import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import RegularGridInterpolator


class PhaseIdentifier:
    """
    Class for identifying phases in vector fields of dynamical systems.

    Phase Identification Criteria in Vector Fields:

    This implementation identifies three distinct phases in dynamical systems based on
    vector field properties:

    1. Stable/Oscillation Phase:
       - Characterized by high curl (rotational behavior) with low divergence
       - Vectors change direction significantly but maintain similar magnitude
       - Typically forms closed or near-closed loops in the phase space
       - Mathematically: |curl| > threshold && |divergence| < threshold

    2. Descent Phase:
       - Significant downward component in the vector field
       - Vectors point downward with increasing magnitude in the direction of motion
       - Often characterized by negative vertical velocity component
       - Mathematically: Vy < -threshold * |V| (vertical component is significantly negative)

    3. Rise/Recovery Phase:
       - Significant upward component in the vector field
       - Vectors point upward, potentially with decreasing magnitude as they approach stable phase
       - Often transitions into oscillatory behavior as system stabilizes
       - Mathematically: Vy > threshold * |V| (vertical component is significantly positive)
    """

    def __init__(self):
        # Phase IDs
        self.UNDEFINED = 0
        self.STABLE = 1
        self.DESCENT = 2
        self.RISE = 3

        # Default colors for phases (RGBA)
        self.phase_colors = {
            self.UNDEFINED: [0, 0, 0, 0],  # Transparent
            self.STABLE: [0.643, 0.325, 0.831, 0.5],  # Purple
            self.DESCENT: [0.369, 0.580, 0.808, 0.5],  # Blue
            self.RISE: [0.682, 0.682, 0.682, 0.5],  # Gray
        }

    def compute_vector_field(self, model_function, t, x_range, y_range, grid_size):
        """
        Compute the vector field and derived properties.
        Returns a dictionary of field properties.
        """
        # Create a grid of points
        x_grid = np.linspace(x_range[0], x_range[1], grid_size)
        y_grid = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Calculate vector field at each grid point
        inputs = np.zeros((grid_size * grid_size, 3))
        inputs[:, 0] = t  # Fixed time
        inputs[:, 1] = X.flatten()  # x positions
        inputs[:, 2] = Y.flatten()  # y positions

        delta_t = 0.01
        inputs_dt = inputs.copy()
        inputs_dt[:, 0] = t + delta_t

        pred_t = model_function.predict(inputs)
        pred_t_dt = model_function.predict(inputs_dt)

        dx_dt = (pred_t_dt[:, 0] - pred_t[:, 0]) / delta_t
        dy_dt = (pred_t_dt[:, 1] - pred_t[:, 1]) / delta_t

        # Reshape to grid
        U = dx_dt.reshape(grid_size, grid_size)
        V = dy_dt.reshape(grid_size, grid_size)

        # Calculate vector properties
        magnitude = np.sqrt(U**2 + V**2)
        angle = np.arctan2(V, U)

        # Calculate derivatives and differential operators
        mag_grad_x, mag_grad_y = np.gradient(magnitude)
        angle_grad_x, angle_grad_y = np.gradient(angle)

        curl = np.gradient(V, x_grid, axis=1) - np.gradient(U, y_grid, axis=0)
        divergence = np.gradient(U, x_grid, axis=1) + np.gradient(V, y_grid, axis=0)

        # Return all computed fields
        return {
            "X": X,
            "Y": Y,
            "U": U,
            "V": V,
            "magnitude": magnitude,
            "angle": angle,
            "curl": curl,
            "divergence": divergence,
            "mag_grad_x": mag_grad_x,
            "mag_grad_y": mag_grad_y,
            "angle_grad_x": angle_grad_x,
            "angle_grad_y": angle_grad_y,
            "x_grid": x_grid,
            "y_grid": y_grid,
        }

    def rule_based_classification(
        self, model_function, t=0.0, x_range=(-3, 3), y_range=(0, 6), grid_size=50
    ):
        """
        Classify phases using rule-based approach.
        """
        # Compute vector field and properties
        field = self.compute_vector_field(
            model_function, t, x_range, y_range, grid_size
        )

        # Extract needed fields
        U, V = field["U"], field["V"]
        magnitude = field["magnitude"]
        curl = field["curl"]
        divergence = field["divergence"]
        X, Y = field["X"], field["Y"]

        # Initialize phase grid
        phases = np.zeros((grid_size, grid_size), dtype=int)

        # Phase classification rules
        threshold = 0.3
        for i in range(grid_size):
            for j in range(grid_size):
                # Classification rules based on domain knowledge
                if abs(curl[i, j]) > 0.5 and abs(divergence[i, j]) < 0.2:
                    # High curl, low divergence = oscillation/stable phase
                    phases[i, j] = self.STABLE
                elif V[i, j] < -threshold * magnitude[i, j]:
                    # Significant downward component = descent phase
                    phases[i, j] = self.DESCENT
                elif V[i, j] > threshold * magnitude[i, j]:
                    # Significant upward component = rise phase
                    phases[i, j] = self.RISE
                else:
                    # Default/transition
                    phases[i, j] = self.UNDEFINED

        return phases, X, Y, field

    def cluster_based_classification(
        self,
        model_function,
        t=0.0,
        x_range=(-3, 3),
        y_range=(0, 6),
        grid_size=50,
        n_clusters=3,
    ):
        """
        Classify phases using clustering approach.
        """
        # Compute vector field and properties
        field = self.compute_vector_field(
            model_function, t, x_range, y_range, grid_size
        )

        # Extract needed fields
        U, V = field["U"], field["V"]
        magnitude = field["magnitude"]
        curl = field["curl"]
        divergence = field["divergence"]
        X, Y = field["X"], field["Y"]

        # Prepare feature matrix for clustering
        features = np.column_stack(
            [
                magnitude.flatten(),
                np.cos(
                    field["angle"].flatten()
                ),  # Use sine/cosine to handle angle wrapping
                np.sin(field["angle"].flatten()),
                curl.flatten(),
                divergence.flatten(),
                (
                    V / np.maximum(magnitude, 1e-10)
                ).flatten(),  # Normalized vertical component
            ]
        )

        # Normalize features
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        features_norm = (features - feature_means) / (feature_stds + 1e-10)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_norm)

        # Reshape clusters back to grid
        phase_clusters = clusters.reshape(grid_size, grid_size)

        # Map clusters to phases based on centroids
        centroids = kmeans.cluster_centers_

        # Analyze centroids to determine mappings
        cluster_mapping = {}

        for i, centroid in enumerate(centroids):
            # Extract normalized centroid features
            normalized_curl = centroid[3] * feature_stds[3] + feature_means[3]
            normalized_divergence = centroid[4] * feature_stds[4] + feature_means[4]
            normalized_v_component = centroid[5] * feature_stds[5] + feature_means[5]

            # Classify based on centroid characteristics
            if abs(normalized_curl) > 0.3 and abs(normalized_divergence) < 0.2:
                cluster_mapping[i] = self.STABLE  # Oscillation
            elif normalized_v_component < -0.2:
                cluster_mapping[i] = self.DESCENT  # Descent
            elif normalized_v_component > 0.2:
                cluster_mapping[i] = self.RISE  # Rise
            else:
                cluster_mapping[i] = self.UNDEFINED  # Undefined

        # Map cluster IDs to phase IDs
        mapped_phases = np.zeros_like(phase_clusters)
        for cluster_id, phase_id in cluster_mapping.items():
            mapped_phases[phase_clusters == cluster_id] = phase_id

        return mapped_phases, X, Y, field

    def get_phase_colors(self, phases, alpha=0.5):
        """
        Get an RGBA array of colors based on phase IDs.
        """
        # Initialize color array
        phase_colors = np.zeros((*phases.shape, 4))

        # Set colors based on phase IDs
        for i in range(phases.shape[0]):
            for j in range(phases.shape[1]):
                phase_id = phases[i, j]
                if phase_id in self.phase_colors:
                    color = self.phase_colors[phase_id].copy()
                    color[3] = alpha  # Set custom alpha
                    phase_colors[i, j] = color
                else:
                    phase_colors[i, j] = [0, 0, 0, 0]  # Transparent for unknown phases

        return phase_colors

    def interpolate_phase_colors(
        self, phase_colors, x_range, y_range, target_resolution
    ):
        """
        Interpolate phase colors to a higher resolution.
        """
        # Source grid coordinates
        x_source = np.linspace(x_range[0], x_range[1], phase_colors.shape[1])
        y_source = np.linspace(y_range[0], y_range[1], phase_colors.shape[0])

        # Target grid coordinates
        x_target = np.linspace(x_range[0], x_range[1], target_resolution)
        y_target = np.linspace(y_range[0], y_range[1], target_resolution)
        X_target, Y_target = np.meshgrid(x_target, y_target)
        points_target = np.vstack((X_target.flatten(), Y_target.flatten())).T

        # Initialize high-resolution color array
        colors_high_res = np.zeros((target_resolution, target_resolution, 4))

        # Interpolate each color channel
        for c in range(4):
            interpolator = RegularGridInterpolator(
                (x_source, y_source),
                phase_colors[:, :, c].T,
                bounds_error=False,
                fill_value=0,
            )

            colors_high_res[:, :, c] = interpolator(points_target).reshape(
                target_resolution, target_resolution
            )

        return colors_high_res
