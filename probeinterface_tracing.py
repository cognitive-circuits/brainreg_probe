''' Here we essentially want to fit probeinterface probes to the signal data from brainsaw imaging slices.'''
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import KDTree
import plotly.graph_objects as go
from matplotlib import pyplot as plt

import probeinterface as pi
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from pathlib import Path
import tifffile 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def get_test_signal_data():
    filepath = Path(r"../data/raw_data/histology/EX03/anat/allen_mouse_10um/2/downsampled.tiff")
    signal_data = tifffile.imread(filepath)  # Load your actual file
    return signal_data

# 1. Standardize orientation
def standardize_orientation(data: np.ndarray, orientation: str = 'psl') -> np.ndarray:
    axis_map = {'p': 0, 'a': 0, 's': 1, 'i': 1, 'l': 2, 'r': 2}
    flips = {'a', 'i', 'r'}
    axes = [axis_map[c] for c in orientation]
    data = np.transpose(data, axes)
    for i, c in enumerate(orientation):
        if c in flips:
            data = np.flip(data, axis=i)
    return data

# 2. Threshold signal via gamma+Otsu
def threshold_signal_gamma(data: np.ndarray,
                           brain_mask: np.ndarray = None,
                           gamma: float = 1.5) -> np.ndarray:
    norm = data.astype(np.float32)
    norm -= norm.min(); norm /= norm.max()
    corr = norm ** gamma
    mask = brain_mask if brain_mask is not None else np.ones_like(corr, bool)
    thresh = threshold_otsu(corr[mask])
    return (corr > thresh) & mask

# 3. Build signal DataFrame
def make_signal_df(data: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    coords = np.argwhere(mask)
    values = data[mask]
    df = pd.DataFrame(coords, columns=['x','y','z'])
    df['value'] = values
    return df[['value','x','y','z']]

# 4. Get probe 2D contacts
def get_probe_contacts_df(manufacturer: str = 'imec', probe_name: str = 'NP2014') -> pd.DataFrame:
    probe = pi.get_probe(manufacturer=manufacturer, probe_name=probe_name)
    df = pd.DataFrame(probe.contact_positions, columns=['2d_x','2d_y'])
    df['value'] = 1
    return df

# 5. Cluster signal points
def cluster_signal(df, n_clusters:int, eps=100, min_samples=100):
    """
    Add DBSCAN cluster labels to a DataFrame with (x,y,z) coordinates.
    
    Parameters:
    - df: DataFrame with columns 'x', 'y', 'z'
    - eps: DBSCAN distance parameter
    - min_samples: minimum points for dense region
    - n_clusters_hint: if provided, will auto-adjust eps to try to get this many clusters
    
    Returns:
    - DataFrame with added 'cluster' column (-1 for noise, 0+ for clusters)
    """

    df_copy = df.copy()
    coords = df_copy[['x', 'y', 'z']].values
    
    # Auto-tune eps if n_clusters_hint provided
    if n_clusters is not None:
        best_eps = eps
        best_diff = float('inf')
        
        for test_eps in np.linspace(50, 200, 20):
            labels = DBSCAN(eps=test_eps, min_samples=min_samples).fit_predict(coords)
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            diff = abs(n_found - n_clusters)
            
            if diff < best_diff:
                best_diff = diff
                best_eps = test_eps
                
        eps = best_eps
    
    # Cluster
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    df_copy['cluster'] = labels
    
    return df_copy


# 6. Fit plane to signal via PCA
def fit_plane_to_signal(signal_df: pd.DataFrame) -> dict:
    """
    Fit a 2D plane to 3D points using PCA.
    Returns dict with centroid, normal, u_axis, v_axis.
    """
    points = signal_df[['x','y','z']].values
    centroid = np.median(points,axis=0)
    centered = points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered)
    # normal: least variance direction
    normal = pca.components_[-1]
    u_axis = pca.components_[0]
    v_axis = pca.components_[1]
    return {'centroid': centroid,
            'normal': normal,
            'u_axis': u_axis,
            'v_axis': v_axis}


# 7. Transform 2d probe contacts before projection
def transform_2d_probe(points_df: pd.DataFrame, params: np.ndarray, voxel_size_um = 10) -> pd.DataFrame:
    """
    Transform 2D probe contacts by rotating and translating.
    points_df: DataFrame with '2d_x', '2d_y' columns.
    params: [offset_u, offset_v, theta]
    """
    off_x, off_y, theta, probe_depth = params  # transformation parameters
    transformed_df = points_df.copy() #scale by the voxel size in micrometers
    transformed_df = transformed_df[transformed_df['2d_y'] <= probe_depth]/voxel_size_um  # filter by depth and then rescale
    pts2d = transformed_df[['2d_x', '2d_y']].values.astype(float)
    # Center the probe points before rotation
    pts2d[:, 0] -= pts2d[:, 0].max() / 2  # center x
    pts2d[:, 1] -= pts2d[:, 1].max() / 2  # center y
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    pts_rot = pts2d @ R2.T
    pts_rot[:, 0] += off_x  # x offset
    pts_rot[:, 1] += off_y  # y offset
    transformed_df[['2d_x', '2d_y']] = pts_rot
    return transformed_df


# 8. Project 2D contacts onto plane (in-plane mapping)
def project_2d_points_to_plane(points_df: pd.DataFrame,
                               plane_info: dict,) -> np.ndarray:
    """
    Map 2D contacts by rotating and translating in plane axes.
    points_df: DataFrame with '2d_x', '2d_y' columns.
    params: [offset_u, offset_v, theta]
    """
   # Extract 2D coordinates
    u_coords = points_df['2d_x'].values
    v_coords = points_df['2d_y'].values
    # Project onto 3D plane using the plane's coordinate system
    # 3D_point = centroid + u_coord * u_axis + v_coord * v_axis
    points_3d = (plane_info['centroid'][np.newaxis, :] + 
                 u_coords[:, np.newaxis] * plane_info['u_axis'][np.newaxis, :] + 
                 v_coords[:, np.newaxis] * plane_info['v_axis'][np.newaxis, :])
    return pd.DataFrame(points_3d, columns=['x', 'y', 'z'])

# 9. Optimize plane-aligned placement
def optimize_probe_plane(signal_df: pd.DataFrame,
                         probe_df: pd.DataFrame,
                         plane: dict,
                         initial_params: np.ndarray = None,
                         bounds: list = None) -> dict:
    sig_pts = signal_df[['x','y','z']].values
    tree = KDTree(sig_pts)
    if initial_params is None:
        initial_params = np.array([0.0, 0.0, 0.0, 3000.0]) # [offset_x, offset_y, theta, probe_depth]
    if bounds is None:
        extent = np.ptp(sig_pts, axis=0).max()
        bounds = [(-extent, extent), 
                  (-extent, extent), 
                  (-np.pi/2, np.pi/2),
                  (500,6000)]
    def cost(p):
        points_2d = transform_2d_probe(probe_df, p)
        coords = project_2d_points_to_plane(points_2d, plane)
        dists, _ = tree.query(coords)
        return np.sum(dists)
    res = minimize(cost, initial_params, bounds=bounds, options = {'maxiter': 1000, 'maxls': 1000})
    transformed_points = transform_2d_probe(probe_df, res.x)
    fitted = project_2d_points_to_plane(transformed_points, plane)
    return {'result': res, 'coords': fitted}

# 9. Interactive 3D Plot
def plot_3d(signal_df: pd.DataFrame,
            probe_coords: np.ndarray = None,
            max_points: int = 5000,
            fig: go.Figure = None):
    df = signal_df.copy()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=df.x, y=df.y, z=df.z,
                               mode='markers', marker=dict(size=2, opacity=0.6)))
    if probe_coords is not None:
        fig.add_trace(go.Scatter3d(x=probe_coords[:,0], 
                                   y=probe_coords[:,1], 
                                   z=probe_coords[:,2],
                                   mode='markers', marker=dict(size=4, color='red')))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.show()

# 10. Demo: Fit PCA plane and plot
if __name__ == '__main__':
    signal_data = get_test_signal_data()
    signal_data = standardize_orientation(signal_data, 'psl')
    threshold_signal = threshold_signal_gamma(signal_data)
    signal_df = make_signal_df(signal_data, threshold_signal)
    signal_df = cluster_signal(signal_df, n_clusters=2)
    signal_df1 = signal_df[signal_df['cluster'] == 1]
    probe_df = get_probe_contacts_df()
    fig = go.Figure()
    for cluster in [0,1]:
        signal_cluster_df = signal_df[signal_df['cluster'] == cluster]
        plane = fit_plane_to_signal(signal_cluster_df)
        # sample points on plane to visualise the fit?
        # sample_points = np.stack(np.meshgrid(np.arange(10),np.arange(10), indexing = 'xy')).reshape(-1,100).T - 5
        #points_df = pd.DataFrame(sample_points, columns=['2d_x', '2d_y'])
        #points_3d = project_2d_points_to_plane(points_df, plane)
        #plot_3d(signal_df, points_3d)
        results = optimize_probe_plane(signal_cluster_df,probe_df,plane)
        if results['result'].success:
            print(f"Fitted parameters: {results['result'].x}")
            plot_3d(signal_cluster_df,results['coords'].values, fig=fig)
