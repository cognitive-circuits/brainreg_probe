''' Here we essentially want to fit probeinterface probes to the signal data from brainsaw imaging slices.

Note: the code here expects data output from brainreg registered to 10um Allen Brain Atlas.
This data is in 'sample space' which is downsampled and reoriented.
We expect 3D image voxel data with shape (n_x, n_y, n_z).
This data is (re)oriented as 'asl', meaning that the origin is in the anterior, superior, left corner of the 3D data.
The first axis ('x') is anterior-posterior, the second ('y') is superior-inferior, and the third ('y') is left-right.

@charlesdgburns
'''
## Setup ## 

#data loading and handling
import json
import tifffile 
import numpy as np
import pandas as pd
from pathlib import Path
# plotting
import plotly.graph_objects as go
from matplotlib import pyplot as plt
# probe info and fitting functions
import probeinterface as pi
from skimage.filters import threshold_otsu
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

## Global variables ##
TEST_ATLAS_PATH =  Path(r"../data/raw_data/histology/EX03/anat/allen_mouse_10um")

EXAMPLE_IMPLANT_INFO = {'probe_depth':[None,]} 
with open('./probeinterface_tracing/allen_name2acronym.json', 'r') as f:
    ALLEN_NAMES2ACRONYM = json.load(f)

## Top level function ##
def get_probe_registration_df(brainreg_atlas_path: Path,
                       probeinterface_probe: object,
                       plot_fit: bool = True):
    return None

## Subfunctions ##

# 0. load necessary data
def get_data(brainreg_atlas_path, signal_channel = '2', control_channel = '3'):
    data = {} #we want to output a structured data dictionary
    # NOTE: Signal_data is what we fit the probe geometry to
    data.update({'signal_data' :tifffile.imread(brainreg_atlas_path/signal_channel/"downsampled.tiff")})
    #data.update({'boundaries_data':tifffile.imread(brainreg_atlas_path/control_channel/"boundaries.tiff")})  # Load your actual file
    data.update({'atlas_registration_data':tifffile.imread(brainreg_atlas_path/control_channel/"registered_atlas.tiff")}) #this is technically in sample space coordinates
    #data.update({'atlas_transformed_data':tifffile.imread(brainreg_atlas_path/control_channel/"downsampled_standard.tiff")})
    data.update({'volumes_df' :pd.read_csv(brainreg_atlas_path/control_channel/'volumes.csv')})
    # NOTE: loading data to transform 3D sample space coordinates to allan atlas coordinates
    for i in range(3):
        data.update({f'deformation_field_{i}': tifffile.imread(brainreg_atlas_path/signal_channel/f"deformation_field_{i}.tiff")})
    data.update({'affine_matrix':np.loadtxt(brainreg_atlas_path/signal_channel/"niftyreg/affine_matrix.txt")})
    return data

# 1. Threshold signal via gamma+Otsu
def threshold_signal_gamma(data: np.ndarray,
                           brain_mask: np.ndarray = None,
                           gamma: float = 1.5) -> np.ndarray:
    '''We threshold the signal to separate dye signal from background autoflorescence.
    Returns: 
    - thresholded_signal: np.ndarray() #(n_x, n_y, n_z) voxel data boolean mask where True indicates signal above threshold.
    Parameters:
    - data: np.ndarray of shape (n_x, n_y, n_z) voxel data with dye signal.
    - gamma: float, gamma exponent to apply to the data before thresholding.
    '''
    norm = data.astype(np.float32)
    norm -= norm.min(); norm /= norm.max()
    corr = norm ** gamma
    mask = brain_mask if brain_mask is not None else np.ones_like(corr, bool)
    thresh = threshold_otsu(corr[mask])
    return (corr > thresh) & mask


# 2. Build signal DataFrame
def make_signal_df(data: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    coords = np.argwhere(mask)
    values = data[mask]
    df = pd.DataFrame(coords, columns=['x','y','z'])
    df['value'] = values
    df['norm_value'] = (values-values.min())/values.max() #normalise to [0,1]
    return df


# 3. Get probe 2D contacts
def get_probe_contacts_df(manufacturer: str = 'imec', probe_name: str = 'NP2014') -> pd.DataFrame:
    probe = pi.get_probe(manufacturer=manufacturer, probe_name=probe_name)
    df = pd.DataFrame(probe.contact_positions, columns=['probe_coords.x','probe_coords.y'])
    df['value'] = 1
    return df


# 4. Cluster signal points
def cluster_signal(df, n_clusters:int, eps=50, min_samples=100):
    """
    Separate signal into data from probes and noise from autoflorescence.
    
    Parameters:
    - df: DataFrame with columns 'x', 'y', 'z'
    - eps: DBSCAN distance required for two points to be considered neighbours. 
        Default is 50x10um = 500um which covers at least two neurpixel 2.0 shanks.
    - min_samples: minimum number of neighbouring points required to be grouped into a cluster.
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
        #we want the smallest eps that gives us the right number of clusters
        for test_eps in np.linspace(10, 200, 20): 
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


# 5. Fit plane to signal via PCA
def fit_plane_to_signal(signal_df: pd.DataFrame,
                        insertion_axis:str= 'si',
                        contact_axis:str= 'lr',
                        longer_than_wider: bool = True) -> dict:
    """
    Fit a 2D plane to 3D points using PCA.
    Returns dict with centroid, normal, u_axis, v_axis, where u,v are aligned with probe depth,width respectively.
    Params:
    - signal_df: DataFrame with 'x', 'y', 'z' columns.
    - contact_axis: str() specifying axis along which the probe contacts are facing
    - insertion_axis: str() specifying axis along which the probe was inserted
    - longer_than_wider: bool, if True (as for neuropixel probes), assumes the probe is longer than it is wide.
    Returns: dict() with
    - centroid: list() [x,y,z] coordinates in sample space corresponding to the centre of the signal 
    - surface_coord: list() [x,y,z] coordinates 
    Note:
    axis strings should be one of 'lr' (left-right), 'pa' (posterior-anterior), 'si' (superior-inferior)
    or their inverses 'rl','ap', and'is'. 
    This is for correspondence with probeinterface axes, 
    where 2D origin is the lower left of the probe i.e. the deepest and left-most contact (when looking at contact faces).
    """
    points = signal_df[['x','y','z']].values
    centroid = np.median(points,axis=0)
    centered = points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered)
    #very cool trick here ngl.
    normal = pca.components_[-1] #least variance direction will be orthogonal to plane.
    u_axis = pca.components_[0] if longer_than_wider else pca.components_[1]
    v_axis = pca.components_[1] if longer_than_wider else pca.components_[0]
    
    # standardise orientation of axes of plane 
    axis2atlas_vector = {'ap':[1,0,0],'si':[0,1,0],'lr':[0,0,1], # axes in 3D space acccording to 'psl' orientation.
                'pa':[-1,0,0],'is':[0,-1,0],'rl':[0,0,-1]} #inverses
    def _cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # flip u_axis if in similar direction to probe depth axis (we want it to point from bottom of probe to top, so actually opposite the insertion axis)
    u_axis = -u_axis if 0 < _cosine_similarity(u_axis, axis2atlas_vector[insertion_axis]) else u_axis
    # flip the normal if it is not in the same direction as the contact facing axis.
    normal = -normal if 0 > _cosine_similarity(normal, axis2atlas_vector[contact_axis]) else normal
    # flip v_axis requires a bit more logic.
    # we know the axis along width is orthogonal to the other two axes, so we can determine the dimension:
    possible_v_axes = [axis_label for axis_label in axis2atlas_vector.keys() if axis_label[0] not in (insertion_axis+contact_axis) ] #we check the first character of each string since all characters are unique.
    # we pick the atlas axis to align to by looking at the signs of the other two axes. 
    # If only one of the axes does not align with the atlas axis, we need to align the v_axis with the inverse of the atlas axis.
    # we also know that if only one of the axes does not align with the atlas axis, the sum of the vector values will sum to 0.
    only_one_axis_misaligned = sum(axis2atlas_vector[insertion_axis]+axis2atlas_vector[contact_axis])==0
    atlas_axis_to_align_to = possible_v_axes[1] if only_one_axis_misaligned else possible_v_axes[0] #these are ordered in the dictionary as axis aligned and inverse.
    v_axis = -v_axis if _cosine_similarity(v_axis, axis2atlas_vector[atlas_axis_to_align_to]) < 0 else v_axis
    
    #lastly, we want a surface estimate, which is essentially the centroid shifted to the brain surface
    surface_coord = centroid.copy()
    u_coords = np.dot(centered, u_axis)  # Projection onto u-axis, aligned with length of probe, with brain surface the largest coordinate.
    top_1_percent_threshold = np.percentile(u_coords, 99)
    top_1_percent_coords = u_coords[u_coords >= top_1_percent_threshold]
    surface_coord += u_axis * np.median(top_1_percent_coords)
    
    return {'centroid': centroid,
            'surface_coord': surface_coord,
            'normal': normal,
            'u_axis': u_axis,
            'v_axis': v_axis}


# 6. Transform 2d probe contacts before projection
def transform_2d_probe(probe_df: pd.DataFrame, params_values: dict, voxel_size_um = 10) -> pd.DataFrame:
    """
    Transform 2D probe contacts by rotating and translating.
    Returns: 
    - pandas DataFrame with transformed 'probe_coords.x', 'probe_coords.y' columns.
    Params:
    - points_df: DataFrame with original 'probe_coords.x', 'probe_coords.y' columns.
    - params: [offset_u, offset_v, theta]
    
    Note: we want to imagine that we are projecting the probe by its depth starting from the top of the signal.
    """
    probe_depth, brain_shrinkage_pct, probe_width_scaling, theta, offset_x,  = params_values  # transformation parameters
    transformed_df = probe_df.copy() #scale by the voxel size in micrometers
    transformed_df = transformed_df[transformed_df['probe_coords.y'] <= probe_depth]  # filter by depth and then rescale
    pts2d = transformed_df[['probe_coords.x', 'probe_coords.y']].values.astype(float)
    # Center the probe points before rescaling
    pts2d[:, 0] -= pts2d[:, 0].max() / 2  # center x
    pts2d[:, 1] -= pts2d[:, 1].max() / 2  # center y
    # rescale the points
    scaling_matrix = np.array([[probe_width_scaling, 0], #here we shrink x axis on plane, aligned with probe width
                               [0, 1]]) * ((100 - brain_shrinkage_pct)/100)  # scale by shrinkage percentage
    pts_scaled = pts2d @ scaling_matrix.T  #apply scaling; transposed due to row-major order of numpy arrays
    # rotate the points
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    pts_rot = pts_scaled @ rotation_matrix.T #apply rotation; transposed due to row-major order of numpy arrays
    # then offset the points
    pts_rot[:, 0] += offset_x  # x offset
    pts_rot[:, 1] -= (probe_depth/2) #offset along probe depth, now in the 
    #Lastly, rescale to the voxel size
    pts_in_voxel_size = pts_rot/voxel_size_um
    transformed_df[['probe_coords.x', 'probe_coords.y']] = pts_in_voxel_size
    return transformed_df


# 7. Project 2D contacts onto plane (in-plane mapping)
def project_2d_points_to_plane(points_df: pd.DataFrame,
                               plane_info: dict,) -> np.ndarray:
    """
    Map 2D contacts by rotating and translating in plane axes.
    Returns:
    - points_3d: np.ndarray of shape (n_points, 3) with projected 3D coordinates.
    Parameters: 
    - points_df: DataFrame with 'probe_coords.x', 'probe_coords.y' columns.
    - plane_info: dict with keys 'surface_coord', 'u_axis', 'v_axis' for a 2D plane in 3D space.
    NOTE: the origin of the projection is 'surface_coord'.
          i.e., the points_df should be transformed accordingly.
    """
   # Extract 2D coordinates
    u_coords = points_df['probe_coords.y'].values #u axis is aligned with insertion / depth of probe / probe height dimension
    v_coords = points_df['probe_coords.x'].values #v axis is aligned with width of the probe
    # Project onto 3D plane using the plane's coordinate system
    points_3d = (plane_info['surface_coord'][np.newaxis, :] + 
                 u_coords[:, np.newaxis] * plane_info['u_axis'][np.newaxis, :] + 
                 v_coords[:, np.newaxis] * plane_info['v_axis'][np.newaxis, :])
    return pd.DataFrame(points_3d, columns=['x', 'y', 'z'])

# 8. define cost function for fitting probe geometry to signal
def compute_cost(signal_df: pd.DataFrame, 
                 probe_df: pd.DataFrame, 
                 plane: dict, 
                 param_values: np.ndarray) -> float:
    """Fit the probe geometry to the signal primarily by minimising distance from signal to probe contacts.
    Returns a float cost value.
    """    
    points_2d = transform_2d_probe(probe_df, param_values)
    coords = project_2d_points_to_plane(points_2d, plane)
    probe_tree = KDTree(coords)
    signal_tree = KDTree(signal_df[['x', 'y', 'z']].values)
    #find distance to nearest neighbours
    signal2contact_dist, _ = probe_tree.query(signal_df[['x', 'y', 'z']].values, k=1)
    contact2signal_dist, _ = signal_tree.query(coords, k=1)
    balanced_cost = np.sum(contact2signal_dist)/len(coords)+np.sum(signal2contact_dist*signal_df['norm_value'].values)/len(signal_df)
    return balanced_cost  # weighted by signal value *signal_df['norm_value'].values

# 9. estimate initial parameters for optimization:
def estimate_initial_params(signal_df: pd.DataFrame, probe_df:pd.DataFrame, plane: dict,
                            probe_depth: float = None) -> dict:
    """ Returns dictionary with good initial parameters for optimization.
    Will by default estimate probe_depth from signal"""
    
    if probe_depth is None:
        #Estimate depth of the probe (in sample space):
        centered_points = signal_df[['x','y','z']].values - plane['centroid']
        u_coords = np.dot(centered_points, plane['u_axis'])  # Projection onto u-axis, aligned with probe depth
        probe_depth = np.percentile(u_coords,99.9) - np.percentile(u_coords,0.1)  # Range of u-coordinates gives depth of probe
        
    #estimate the shrinkage of the brain by fitting iteratively and choosing best parameters
    #we can set initial offset and rotation to 0.0 thanks to careful alignment of the plane and projection centered on signal 
    #let's do a cost search
    min_cost = 1e42 #start very high
    for brain_shrinkage_pct in np.linspace(0,5,6): #try 0,1,2,...15 pct shrinkage
        for probe_width_scale in np.linspace(0.8,1,11):
            adjusted_probe_depth = 10*probe_depth * (100 / (100 - brain_shrinkage_pct))  # scale to um and adjust for brain shrinkage.
            params = {'probe_depth':adjusted_probe_depth,  # Adjust depth for brain shrinkage in histology sample space
                    'brain_shrinkage_pct':brain_shrinkage_pct,
                    'probe_width_scaling':probe_width_scale,
                    'theta':1e-3,  # small nonzero rotation initially,
                    'offset_x': 1,}   # small offset initially
            cost = compute_cost(signal_df,probe_df,plane,params.values())
            if cost<min_cost:
                min_cost = cost
                best_params = params.copy()
    return best_params

# 10. Optimize plane-aligned placement
def optimize_probe_plane(signal_df: pd.DataFrame,
                         probe_df: pd.DataFrame,
                         plane: dict,
                         initial_params: dict = None,
                         bounds: dict = {'depth':(500,6000),#depth
                                        'brain_shrinkage_pct':(0,5), #account for some pct brain shrinkage in signal positions
                                        'probe_width_scaling':(0.8,1), #shank width compression for NPXL2.0
                                        'rotation':(-np.pi/2, np.pi/2), #rotation
                                        'offset_x':(-1000, 1000), #offset along probe width in um
                                        }) -> dict:
    if initial_params is None:
        initial_params = estimate_initial_params(signal_df,probe_df, plane)
    def cost(p): #must be given as a function of parameters
        return compute_cost(signal_df, probe_df, plane, p) #turn parameter values into dict again
    res = minimize(cost, 
                   x0 = list(initial_params.values()), 
                   bounds=list(bounds.values()), 
                   options = {'maxiter': 1000, 'maxls': 1000}) #help the constrained optimisation explore parameter space.
    best_params_dict = dict(zip(initial_params.keys(),res.x))
    transformed_points = transform_2d_probe(probe_df, best_params_dict.values())
    fitted = project_2d_points_to_plane(transformed_points, plane)
    return {'result': res,'params':best_params_dict, 'coords': fitted}

# 11. get structure labels
def get_structure_labels(points_array:np.array, data:dict):
    volume_ids = []
    structure_names = []
    acronyms = []
    for each_coord in points_array:
        int_coords = [int(x) for x in each_coord]
        volume_ids.append(data['atlas_registration_data'][int_coords[0],int_coords[1],int_coords[2]])
        if volume_ids[-1]< len(data['volumes_df']):
            structure_names.append(data['volumes_df'].iloc[volume_ids[-1]].structure_name)
            acronyms.append(ALLEN_NAMES2ACRONYM[structure_names[-1]])
        else:
            structure_names.append('outside brain')
            acronyms.append(np.nan)
    return volume_ids, structure_names, acronyms


# 12. transform sample space to allen space
def sample_coords_to_allen_space(points_array:np.array, 
                                 data:dict,
                                 voxel_size:int = 10):
    '''Not super trivial transformation from sample coordinates to allen coords.'''
    # 1. apply affine transformation
    affine = data['affine_matrix'] # 4×4 matrix
    pts_homog = np.hstack([ points_array[:, ::-1],  # (z,y,x) → (x,y,z)
                            np.ones((len(points_array), 1))])
    affine_out = (affine @ pts_homog.T).T  # (N,4)
    affine_vox_atlas = affine_out[:, :3][:, ::-1]  # back to (z,y,x) for deformation
    # 2. apply deformation field
    coords = [affine_vox_atlas[:,i] for i in range(3)]
    disp = np.stack([
        map_coordinates(data['deformation_field_0'], coords, order=1),
        map_coordinates(data['deformation_field_1'], coords, order=1),
        map_coordinates(data['deformation_field_2'], coords, order=1),
    ], axis=1)

    atlas_vox = affine_vox_atlas + disp
    atlas_um = atlas_vox * voxel_size
    return atlas_um
 
# Visualisation: Interactive 3D Plot

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

def adjust_contrast(image, percentile_low=1,percentile_high = 99):
    """Set the contrast, useful for viewng sample data slices at probe centroid"""
     # Convert to float if needed
    if image.dtype != np.float64:
        image = image.astype(np.float64)
    # Calculate percentiles
    p_low, p_high = np.percentile(image, (percentile_low, percentile_high))
    # Clip and rescale
    image_stretched = np.clip(image, p_low, p_high)
    image_stretched = (image_stretched - p_low) / (p_high - p_low)
    return image_stretched

def plot_sample_data_sections(signal_data, plane_centroid, ax= None):
    '''Plot sagittal and coronal sections of the signal data,
    centered the centroid of the plane fit to the probe signal data.'''
    saggital = signal_data[:,:,int(plane_centroid[2])].T
    coronal = data['signal_data'][int(plane_centroid[0]),:,:]
    #transverse = data['signal_data'][:,plane_centroid[1],:] #rarely used, but here in case
    #adjust contrast
    sagittal = adjust_contrast(saggital)
    coronal = adjust_contrast(coronal)
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize =(10,3), width_ratios = [1.5,1])
    fig, ax = plt.subplots(1,2, figsize = (10,3), width_ratios=[1.5,1])
    ax[0].imshow(sagittal,cmap='gray')
    ax[0].axis('off')
    ax[0].xaxis.set_inverted(True)
    ax[0].set(title = 'Sagittal section')

    ax[1].imshow(coronal,cmap = 'gray')
    ax[1].axis('off')
    ax[1].set(title = 'Coronal section')
    return fig

# 10. Demo: Fit PCA plane and plot
if __name__ == '__main__':
    data = get_data(TEST_ATLAS_PATH)
    threshold_signal = threshold_signal_gamma(data['signal_data'])
    signal_df = make_signal_df(data['signal_data'], threshold_signal)
    signal_df = cluster_signal(signal_df, n_clusters=2)
    signal_df1 = signal_df[signal_df['cluster'] == 1]
    probe_df = get_probe_contacts_df()
    fig = go.Figure()
    for cluster in [0,1]:
        signal_cluster_df = signal_df[signal_df['cluster'] == cluster]
        plane = fit_plane_to_signal(signal_cluster_df)
        # sample points on plane to visualise the fit?
        # sample_points = np.stack(np.meshgrid(np.arange(10),np.arange(10), indexing = 'xy')).reshape(-1,100).T - 5
        #points_df = pd.DataFrame(sample_points, columns=['probe_coords.x', 'probe_coords.y'])
        #points_3d = project_2d_points_to_plane(points_df, plane)
        #plot_3d(signal_df, points_3d)
        results = optimize_probe_plane(signal_cluster_df,probe_df,plane)
        if results['result'].success:
            print(f"Fitted parameters: {results['result'].x}")
            plot_3d(signal_cluster_df,results['coords'].values, fig=fig)
