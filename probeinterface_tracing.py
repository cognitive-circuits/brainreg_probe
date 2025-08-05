''' Here we essentially want to fit probeinterface probes to the signal data from brainsaw imaging slices.'''

import numpy as np
import tifffile
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from skimage import measure, morphology, filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tifffile
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from skimage import measure, morphology, filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import probeinterface as pi
from pathlib import Path



class ElectrodeArrayRegistration:
    def __init__(self, signal_data, probe_contacts_2d, voxel_size=(10, 10, 10)):
        """
        Initialize the registration class.
        
        Parameters:
        -----------
        signal_data : numpy.ndarray
            3D array of signal intensities from downsampled.tiff (in 10μm voxels)
        probe_contacts_2d : numpy.ndarray
            Nx2 array of electrode coordinates from probeinterface (in 1μm units)
        voxel_size : tuple
            Voxel size in micrometers (z, y, x) - typically (10, 10, 10)
        """
        self.signal_data = signal_data
        self.probe_contacts_2d = np.array(probe_contacts_2d)  # In 1μm units
        self.voxel_size = np.array(voxel_size)  # In μm
        self.brain_mask = None
        self.surface_coords = None
        self.thresholded_signal = None
        
        print(f"Signal data shape: {signal_data.shape} (10μm voxels)")
        print(f"Probe contacts: {len(probe_contacts_2d)} electrodes")
        print(f"Probe range: X=[{probe_contacts_2d[:, 0].min():.1f}, {probe_contacts_2d[:, 0].max():.1f}]μm, "
              f"Y=[{probe_contacts_2d[:, 1].min():.1f}, {probe_contacts_2d[:, 1].max():.1f}]μm")
        
    def identify_brain_surface(self, method='largest_component', smoothing_sigma=1.0):
        """
        Identify brain surface coordinates from the signal data.
        
        Parameters:
        -----------
        method : str
            Method for surface identification ('largest_component' or 'threshold')
        smoothing_sigma : float
            Gaussian smoothing parameter
        """
        print("Identifying brain surface...")
        
        # Smooth the data to reduce noise
        smoothed = filters.gaussian(self.signal_data, sigma=smoothing_sigma)
        
        if method == 'largest_component':
            # Use Otsu thresholding to separate brain from background
            threshold = filters.threshold_otsu(smoothed)
            binary = smoothed > threshold
            
            # Keep only the largest connected component (brain)
            labeled = measure.label(binary)
            props = measure.regionprops(labeled)
            largest_label = max(props, key=lambda x: x.area).label
            self.brain_mask = labeled == largest_label
            
        elif method == 'threshold':
            # Simple thresholding approach
            threshold = np.percentile(smoothed[smoothed > 0], 75)
            self.brain_mask = smoothed > threshold
        
        # Clean up the mask
        self.brain_mask = morphology.binary_closing(self.brain_mask, 
                                                   morphology.ball(2))
        self.brain_mask = morphology.binary_opening(self.brain_mask, 
                                                   morphology.ball(1))
        
        # Extract surface coordinates
        surface_binary = self.brain_mask ^ morphology.binary_erosion(
            self.brain_mask, morphology.ball(1))
        self.surface_coords = np.array(np.where(surface_binary)).T
        
        print(f"Found {len(self.surface_coords)} surface voxels")
        
    def threshold_signal_data(self, gamma=1.5):
        """
        Apply gamma correction followed by Otsu thresholding to the signal data.
        
        Parameters:
        -----------
        gamma : float
            Gamma correction value. Values >1 emphasize bright regions.
        """
        print("Applying gamma thresholding...")

        # Normalize to [0, 1] for gamma correction
        signal_norm = self.signal_data.astype(np.float32)
        signal_norm -= signal_norm.min()
        signal_norm /= signal_norm.max()

        # Gamma correction
        corrected = signal_norm ** gamma

        # Otsu thresholding within the brain mask
        threshold = filters.threshold_otsu(corrected[self.brain_mask])

        # Binary thresholded signal
        self.thresholded_signal = (corrected > threshold) & self.brain_mask

        print(f"Gamma: {gamma}")
        print(f"Threshold after gamma correction: {threshold:.3f}")
        print(f"Number of active voxels: {np.sum(self.thresholded_signal)}")
        
    def project_2d_to_3d_coordinates(self, insertion_depth_um, probe_tip_voxel, probe_angles):
        """
        Project 2D probe coordinates to 3D space with given insertion parameters.
        
        Parameters:
        -----------
        insertion_depth_um : float
            Insertion depth in micrometers (fitted parameter)
        probe_tip_voxel : numpy.ndarray
            3D position of probe tip in voxel coordinates [z, y, x]
        probe_angles : numpy.ndarray
            Probe orientation angles [pitch, yaw, roll] in radians
        
        Returns:
        --------
        coords_3d_voxels : numpy.ndarray
            Nx3 array of 3D electrode positions in voxel coordinates
        valid_electrodes : numpy.ndarray
            Boolean mask of electrodes within insertion depth
        """
        
        # Find electrodes within insertion depth
        # Probe coordinate system: tip is at y=0, electrodes extend in +y direction
        probe_tip_y = self.probe_contacts_2d[:, 1].min()  # Tip position in probe coords
        distances_from_tip = self.probe_contacts_2d[:, 1] - probe_tip_y
        valid_electrodes = distances_from_tip <= insertion_depth_um
        
        # Get valid electrode positions in probe coordinates (1μm units)
        valid_contacts_2d = self.probe_contacts_2d[valid_electrodes]
        
        # Create 3D coordinates in probe reference frame (1μm units)
        # Probe coordinate system: x=lateral, y=along probe axis, z=0 (2D probe)
        coords_3d_um = np.zeros((len(valid_contacts_2d), 3))
        coords_3d_um[:, 0] = valid_contacts_2d[:, 0]  # lateral (x)
        coords_3d_um[:, 1] = valid_contacts_2d[:, 1] - probe_tip_y  # distance from tip (y)
        coords_3d_um[:, 2] = 0  # probe is 2D (z)
        
        # Apply probe orientation (rotation matrices)
        pitch, yaw, roll = probe_angles
        
        # Rotation around x-axis (pitch)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        
        # Rotation around y-axis (yaw)  
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        
        # Rotation around z-axis (roll)
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Apply rotation
        coords_3d_rotated_um = (R @ coords_3d_um.T).T
        
        # Convert probe tip position to micrometers
        probe_tip_um = probe_tip_voxel * self.voxel_size
        
        # Translate to brain coordinate system (micrometers)
        coords_3d_brain_um = coords_3d_rotated_um + probe_tip_um
        
        # Convert to voxel coordinates
        coords_3d_voxels = coords_3d_brain_um / self.voxel_size
        
        return coords_3d_voxels, valid_electrodes
        
    def compute_registration_cost(self, transform_params):
        """
        Compute cost function for registration optimization.
        
        Parameters:
        -----------
        transform_params : numpy.ndarray
            [tip_z, tip_y, tip_x, pitch, yaw, roll, insertion_depth_um]
            - tip position in voxel coordinates
            - angles in radians
            - insertion depth in micrometers
        
        Returns:
        --------
        cost : float
            Registration cost (lower is better)
        """
        
        if len(transform_params) != 7:
            raise ValueError(f"Expected 7 parameters, got {len(transform_params)}")
            
        tip_z, tip_y, tip_x, pitch, yaw, roll, insertion_depth_um = transform_params
        
        # Ensure insertion depth is positive and reasonable
        if insertion_depth_um <= 0 or insertion_depth_um > 10000:  # 0-10mm range
            return 1e6
        
        probe_tip_voxel = np.array([tip_z, tip_y, tip_x])
        probe_angles = np.array([pitch, yaw, roll])
        
        try:
            # Project to 3D coordinates
            coords_3d_voxels, valid_electrodes = self.project_2d_to_3d_coordinates(
                insertion_depth_um, probe_tip_voxel, probe_angles
            )
            
            if len(coords_3d_voxels) == 0:
                return 1e6
            
            # Round to nearest voxel indices
            voxel_coords = np.round(coords_3d_voxels).astype(int)
            
            # Check bounds
            bounds_mask = (
                (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < self.signal_data.shape[0]) &
                (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < self.signal_data.shape[1]) &
                (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < self.signal_data.shape[2])
            )
            
            if not np.any(bounds_mask):
                return 1e6  # High penalty for out-of-bounds
            
            valid_coords = voxel_coords[bounds_mask]
            
            # Cost function: negative sum of signal intensities at electrode positions
            signal_sum = np.sum(self.thresholded_signal[valid_coords[:, 0], 
                                                       valid_coords[:, 1], 
                                                       valid_coords[:, 2]])
            
            # Add penalty for electrodes outside brain
            brain_penalty = np.sum(~self.brain_mask[valid_coords[:, 0], 
                                                   valid_coords[:, 1], 
                                                   valid_coords[:, 2]]) * 10
            
            # Add penalty for very few electrodes contributing
            if len(valid_coords) < 10:
                electrode_penalty = (10 - len(valid_coords)) * 5
            else:
                electrode_penalty = 0
            
            # We want to maximize signal_sum, so return negative
            cost = -signal_sum + brain_penalty + electrode_penalty
            
            return cost
            
        except Exception as e:
            print(f"Error in cost computation: {e}")
            return 1e6
        
    def register_electrode_array(self, initial_tip_position=None, max_insertion_depth=6000):
        """
        Register the electrode array to the signal data.
        
        Parameters:
        -----------
        initial_tip_position : tuple or None
            Initial guess for tip position in voxel coordinates (z, y, x)
            If None, uses center of thresholded signal
        max_insertion_depth : float
            Maximum insertion depth in micrometers
        
        Returns:
        --------
        result : dict
            Registration results including transformation parameters
        """
        print("Starting electrode array registration...")
        
        # Initial guess for tip position
        if initial_tip_position is None:
            signal_coords = np.array(np.where(self.thresholded_signal)).T
            signal_center = np.mean(signal_coords, axis=0)
            initial_tip_position = signal_center
        
        # Initial parameters: [tip_z, tip_y, tip_x, pitch, yaw, roll, insertion_depth_um]
        initial_params = np.array([
            initial_tip_position[0],  # tip_z
            initial_tip_position[1],  # tip_y  
            initial_tip_position[2],  # tip_x
            0.0,  # pitch
            0.0,  # yaw
            0.0,  # roll
            3000.0  # insertion_depth_um (initial guess: 3mm)
        ])
        
        print(f"Initial tip position (voxels): {initial_tip_position}")
        print(f"Initial insertion depth: {initial_params[6]:.1f} μm")
        
        # Parameter bounds
        bounds = [
            (0, self.signal_data.shape[0]),  # tip_z
            (0, self.signal_data.shape[1]),  # tip_y
            (0, self.signal_data.shape[2]),  # tip_x
            (-np.pi/4, np.pi/4),  # pitch (-45 to +45 degrees)
            (-np.pi/4, np.pi/4),  # yaw
            (-np.pi/4, np.pi/4),  # roll
            (500, max_insertion_depth)  # insertion_depth_um (0.5mm to max)
        ]
        
        # Optimize
        result = minimize(
            self.compute_registration_cost,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': True}
        )
        
        print(f"Registration completed. Final cost: {result.fun:.3f}")
        
        if result.success:
            tip_z, tip_y, tip_x, pitch, yaw, roll, insertion_depth_um = result.x
            print(f"Fitted tip position: ({tip_z:.1f}, {tip_y:.1f}, {tip_x:.1f}) voxels")
            print(f"Fitted angles: pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°, roll={np.degrees(roll):.1f}°")
            print(f"Fitted insertion depth: {insertion_depth_um:.1f} μm")
            
            # Get final electrode positions
            probe_tip_voxel = np.array([tip_z, tip_y, tip_x])
            probe_angles = np.array([pitch, yaw, roll])
            final_coords_3d, valid_electrodes = self.project_2d_to_3d_coordinates(
                insertion_depth_um, probe_tip_voxel, probe_angles
            )
            
        else:
            final_coords_3d = None
            valid_electrodes = None
        
        # Store results
        registration_result = {
            'success': result.success,
            'transformation_params': result.x,
            'cost': result.fun,
            'final_coords_3d': final_coords_3d,
            'valid_electrodes': valid_electrodes,
            'n_electrodes_fitted': len(final_coords_3d) if final_coords_3d is not None else 0,
            'insertion_depth_um': result.x[6] if result.success else None
        }
        
        return registration_result
        
    def visualize_results(self, registration_result, slice_idx=None):
        """
        Visualize registration results.
        
        Parameters:
        -----------
        registration_result : dict
            Results from register_electrode_array
        slice_idx : int or None
            Slice index for 2D visualization, if None shows 3D
        """
        if not registration_result['success']:
            print("Registration failed, cannot visualize results")
            return
            
        final_coords_3d = registration_result['final_coords_3d']
        
        if slice_idx is not None:
            # 2D slice visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original signal
            axes[0].imshow(self.signal_data[slice_idx], cmap='viridis')
            axes[0].set_title('Original Signal')
            
            # Thresholded signal with brain mask
            overlay = np.zeros_like(self.signal_data[slice_idx])
            overlay[self.brain_mask[slice_idx]] = 0.3
            overlay[self.thresholded_signal[slice_idx]] = 1.0
            axes[1].imshow(overlay, cmap='hot')
            axes[1].set_title('Thresholded Signal + Brain Mask')
            
            # Result with electrodes
            axes[2].imshow(self.signal_data[slice_idx], cmap='viridis', alpha=0.7)
            
            # Plot electrodes in this slice
            electrode_z = final_coords_3d[:, 0]
            in_slice = np.abs(electrode_z - slice_idx) < 1
            if np.any(in_slice):
                slice_electrodes = final_coords_3d[in_slice]
                axes[2].scatter(slice_electrodes[:, 2], slice_electrodes[:, 1], 
                               c='red', s=50, marker='x')
            axes[2].set_title(f'Registration Result (slice {slice_idx})')
            
            plt.tight_layout()
            plt.show()
            
        else:
            # 3D visualization
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Sample signal points for visualization
            signal_points = np.array(np.where(self.thresholded_signal)).T
            if len(signal_points) > 1000:
                sample_idx = np.random.choice(len(signal_points), 1000, replace=False)
                sampled_signal = signal_points[sample_idx]
            else:
                sampled_signal = signal_points
            
            # Plot signal
            ax.scatter(sampled_signal[:, 2], sampled_signal[:, 1], sampled_signal[:, 0], 
                      c='blue', alpha=0.3, s=1, label='Signal')
            
            # Plot registered electrodes
            ax.scatter(final_coords_3d[:, 2], final_coords_3d[:, 1], final_coords_3d[:, 0], 
                      c='red', s=50, marker='o', label='Registered Electrodes')
            
            # Plot probe tip
            tip_pos = registration_result['transformation_params'][:3]
            ax.scatter(tip_pos[2], tip_pos[1], tip_pos[0], 
                      c='green', s=100, marker='*', label='Probe Tip')
            
            ax.set_xlabel('X (voxels)')
            ax.set_ylabel('Y (voxels)')
            ax.set_zlabel('Z (voxels)')
            ax.legend()
            ax.set_title(f'3D Registration Result\n'
                        f'Insertion depth: {registration_result["insertion_depth_um"]:.1f} μm\n'
                        f'Electrodes: {registration_result["n_electrodes_fitted"]}')
            
            plt.show()


def load_neuropixels_probe():
    """Load Neuropixels 2.0 probe using probeinterface."""
    print("Loading Neuropixels 2.0 probe...")
    probe_object = pi.get_probe(manufacturer='imec', probe_name='NP2014')
    contact_positions = probe_object.contact_positions  # In 1μm units
    
    print(f"Loaded probe: {probe_object.name}")
    print(f"Number of contacts: {len(contact_positions)}")
    print(f"Contact range: x=[{contact_positions[:, 0].min():.1f}, {contact_positions[:, 0].max():.1f}]μm, "
          f"y=[{contact_positions[:, 1].min():.1f}, {contact_positions[:, 1].max():.1f}]μm")
    
    return probe_object, contact_positions


def main():
    """Example usage of the ElectrodeArrayRegistration class."""
    
    # Load Neuropixels probe
    probe_object, contact_positions = load_neuropixels_probe()
    
    # Load signal data
    print("\nLoading signal data...")
    filepath = Path(r"D:\explore_exploit\data\raw_data\histology\TA_G_03_01__06_04__08\EX06\anat\allen_mouse_10um\2\downsampled.tiff")
    signal_data = tifffile.imread(filepath)  # Load your actual file
    
    # Initialize registration
    reg = ElectrodeArrayRegistration(signal_data, contact_positions, 
                                   voxel_size=(10, 10, 10))
    
    # Step 1: Identify brain surface
    reg.identify_brain_surface(method='largest_component')
    
    # Step 2: Threshold signal data
    reg.threshold_signal_data()
    
    # Step 3: Register electrode array (insertion depth will be fitted)
    result = reg.register_electrode_array(max_insertion_depth=6000)
    
    # Step 4: Visualize results
    if result['success']:
        print("\nRegistration successful!")
        
        # Show 2D slice
        middle_slice = signal_data.shape[0] // 2
        reg.visualize_results(result, slice_idx=middle_slice)
        
        # Show 3D visualization
        reg.visualize_results(result)
    else:
        print("Registration failed!")
        
    return reg, result



if __name__ == "__main__":
    reg, result = main()
    
    
    
    
## plotting ##

import plotly.graph_objects as go

def plot_signal_3d_interactive(signal_data, signal_threshold, max_points=5000):
    """
    Plot thresholded signal coordinates in 3D using Plotly.
    
    Parameters:
    -----------
    max_points : int
        Maximum number of points to plot for performance.
    """

    # Get coordinates of active voxels
    coords = np.array(np.where(signal_threshold)).T
    n_points = len(coords)

    if n_points == 0:
        print("No active voxels to plot.")
        return

    # Subsample if too many points
    if n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        coords = coords[indices]

    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]


    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=signal_data[signal_threshold],# Color by z-depth
            colorscale='Viridis',
            opacity=0.3
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X (voxels)',
            yaxis_title='Y (voxels)',
            zaxis_title='Z (voxels)',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"Thresholded Signal (showing {len(coords)} voxels)"
    )

    fig.show()
    

#plot_signal_3d_interactive(reg.signal_data, reg.thresholded_signal, max_points = 100000)


## ChatGPT version

