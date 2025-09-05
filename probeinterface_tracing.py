''' Here we essentially want to fit probeinterface probes to the signal data from brainsaw imaging slices.

Note: the code here expects data output from brainreg registered to 10um Allen Brain Atlas.
This data is in 'sample space' which is downsampled and reoriented.
We expect 3D image voxel data with shape (n_i, n_j, n_k).
This data is (re)oriented as 'asr', meaning that the origin is in the anterior, superior, right corner of the 3D data.
The first axis ('i' or 'Z') is anterior-posterior, the second ('j' or 'Y') is superior-inferior, and the third ('k' or 'X') is right-left.

@charlesdgburns
'''
## Setup ## 

#data loading and handling
import json
import tifffile 
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
# plotting
from brainreg_probe import plot_util_func as puf
from brainreg_probe import manually_annotated_points as maps
# probe info and fitting functions
import probeinterface as pi
from skimage.filters import threshold_otsu
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

## Global variables ##

#with open('./brainreg_probe/allen_brain_atlas_info.htsv', 'r') as f:
PREPROCESSED_BRAINREG_PATH = Path('../data/preprocessed_data/brainreg')
ATLAS_NAME = 'allen_mouse_10um'
VOXEL_SIZE = 10 #in um
ALLEN_ATLAS_INFO_DF = pd.read_csv('./brainreg_probe/allen_brain_atlas_info.htsv', sep='\t')
AXIS2ATLAS_VECTOR = {'ap':[1,0,0],'si':[0,1,0],'rl':[0,0,1], # axes in 3D space acccording to 'asr' orientation.
                     'pa':[-1,0,0],'is':[0,-1,0],'lr':[0,0,-1]} #inverses
# option to split probes by probe ordering axis instead of by dbscan clustering.
ORDER_SPLIT = False #can be useful when two probes are very close to each other.

#specify information about brainreg and probes in data.
EXAMPLE_INPUT_DICT = { 'brainreg_signal_channel':2, #commonly 2, should be set to None if there is no signal data (purely manual annotation). 
                        #specifying probe information allowing for multiple probes.
                        'probe_info':[{'label':'ProbeA','manufacturer':'imec','name':'NP2020'},
                                      {'label':'ProbeB','manufacturer':'imec','name':'NP2020'}],
                        'probe_depths_um':[None,None], #insert values in um if known, otherwise set to None and estimate from the signal data.
                        'probe_ordering_axis':'ap', #how are the probe inputs ordered anatomically?
                        'insertion_axis':['si','si'],
                        'contact_face_axis':['lr','rl'],
                        'target_regions':['CA1','CA1'] # only used to highlight regions when plotting 
                        }

## Top level function ##
def run_probeinterface_tracking(subject_IDs:list = [], plotting=True):
    if subject_IDs == []:
        subject_IDs = [p.stem for p in PREPROCESSED_BRAINREG_PATH.iterdir()]
    
    for each_subject in subject_IDs:
        brainreg_atlas_path = PREPROCESSED_BRAINREG_PATH/each_subject/ATLAS_NAME
        input_dict = EXAMPLE_INPUT_DICT #can make subject-level changes here if required
        # SUGGESTION: rewrite this function with subject-level required changes here.
        try:
            print(f'Fitting probe to data from {each_subject}')
            probe_dfs, fit_params, signal_df, data = get_probe_registration_df(brainreg_atlas_path, input_dict)
        except Exception as e:
            print(f'Failed to get probe_df for {each_subject}: {e}')
            continue
        if plotting:
            for idx, probe_df in enumerate(probe_dfs):
                fig, ax = puf.plt.subplots(2,2, figsize = (8,5), width_ratios=[1.5,1])
                fig.suptitle(f"{each_subject} | {fit_params[idx]['probe_name']}")
                if data.__contains__('manual_points'):
                    points_used_for_fit = signal_df[signal_df['cluster']==idx]
                else:
                    points_used_for_fit = None
                _ = puf.plot_sample_data_sections(data['signal_data'],
                                                fit_params[idx]['centroid'],
                                                points_used_for_fit,            #REMOVE THIS IF YOU WANT TO JUST PLOT THE Dil SIGNAL
                                                probe_df.downsample_coords,     #REMOVE THIS IF YOU WANT TO JUST PLOT THE Dil SIGNAL
                                                fig_and_axes=(fig,[ax[0][0],ax[0][1]]))
                allen_plane_centroid = sample_coords_to_allen_space([fit_params[idx]['centroid']],
                                                                        data)[0]
                _ = puf.plot_atlas_data_sections(probe_df.allen_atlas_coords.values,
                                                    allen_plane_centroid,
                                                    fig_and_axes = (fig,[ax[1][0],ax[1][1]]),
                                                    highlight_region=input_dict['target_regions'][idx])
                fig_path = PREPROCESSED_BRAINREG_PATH/each_subject/f"{fit_params[idx]['probe_name']}_sections.png"
                fig.savefig(fig_path)

                print(f'Saved section plots to:\n {fig_path}')
    return None 

def get_probe_registration_df(brainreg_atlas_path: Path, #the path to your subject's brainreg output folder
                              input_dict: dict):
    '''Runs probe registration. 
    Returns 
    -------
    probe_dfs: list() of pd.Dataframe
        probe_df containing probe coords, '''
    probe_info = input_dict['probe_info']
    n_probes = len(probe_info)
    print('Loading data...')
    data = get_data(brainreg_atlas_path, 
                    signal_channel = input_dict['brainreg_signal_channel'])
    if input_dict['brainreg_signal_channel'] is not None:
        print(f'Thresholding signal...')
        threshold_signal = threshold_signal_gamma(data['signal_data'])
        signal_df = make_signal_df(data['signal_data'], threshold_signal)
    else:
        signal_df = pd.DataFrame()
    if data.__contains__('manual_points'):
        signal_df = pd.concat([signal_df, data['manual_points']], ignore_index=True)
    print(f'Clustering signal ({len(signal_df)} points) into {n_probes} clusters...')
    signal_df = separate_probes_from_noise(signal_df, 
                                           n_probes, 
                                           input_dict['probe_ordering_axis'],
                                            DBSCAN_min_samples=100) ## OBS: May need to manually adjust this 
    # Get brain surface:
    surface_df = get_surface_coord_df(data['boundaries'])
    # Get ready to rumble:
    probe_dfs = []
    fit_params = []
    for each_probe in range(n_probes):
        probe_df = get_probe_contacts_df(probe_info[each_probe]['manufacturer'],
                                            probe_info[each_probe]['name'])
        longer_than_wider = probe_df['probe_coords.y'].max()>probe_df['probe_coords.x'].max()
        probe_signal_df = signal_df[signal_df['cluster']==each_probe]
        print(f"Fitting {probe_info[each_probe]['label']} to signal data...")
        fitted_plane = fit_plane_to_signal(probe_signal_df,
                                            input_dict['insertion_axis'][each_probe],
                                            input_dict['contact_face_axis'][each_probe],
                                            longer_than_wider)
        fitted_plane = append_surface_coord(surface_df,fitted_plane)
        best_params_dict = optimize_probe_plane(probe_signal_df,probe_df,fitted_plane,
                                                    fixed_depth_um = input_dict['probe_depths_um'][each_probe])
        transformed_points = transform_2d_probe(probe_df, best_params_dict.values())
        downsampled_coords = project_2d_points_to_plane(transformed_points, fitted_plane)
        atlas_coords = sample_coords_to_allen_space(downsampled_coords.values, data)
        #return the coordinates in downsampled space 
        
        #now append coords to the dataframe
        probe_df = probe_df[probe_df['probe_coords.y']<best_params_dict['probe_depth']]
        for coord_type, coords_data in {'downsample_coords':downsampled_coords.values,'allen_atlas_coords':atlas_coords}.items():
                for idx, coord in enumerate(['i','j','k']):
                    probe_df[f'{coord_type}.{coord}'] = coords_data[:,idx]
        # append the anatomy data
        anatomy_dict = get_structure_labels(downsampled_coords.values, data)
        for label, structure_data in anatomy_dict.items():
            probe_df[f'structure.{label}'] = structure_data
        # add probe name to data
        probe_df['probe_name'] = probe_info[each_probe]['label']
        best_params_dict['probe_name'] = probe_info[each_probe]['label']
        best_params_dict.update({k:list(v) for k,v in fitted_plane.items()})
        print(best_params_dict)
        #save out the data
        probe_df_path = (brainreg_atlas_path.parent)/f"{probe_info[each_probe]['label']}_anatomy.htsv"
        probe_df.to_csv(probe_df_path,sep='\t', index=False)
        params_path = (brainreg_atlas_path.parent)/f"{probe_info[each_probe]['label']}_fit_params.json"
        with open(params_path, "w") as f:      #below we make np.arrays() lists for saving
            json.dump(best_params_dict, f, indent = 4)
        #append to lists
        #make probe_df multiindex before outputting
        probe_df.columns = pd.MultiIndex.from_tuples([col.split('.') if '.' in col else (col,'') for col in probe_df.columns])
        probe_dfs.append(probe_df[['probe_coords','downsample_coords','allen_atlas_coords','structure']]) #simple reordering of columns here.. nothing to see
        fit_params.append(best_params_dict)

        
    return probe_dfs, fit_params, signal_df,data 

## Subfunctions ##

# 0. load necessary data
def get_data(brainreg_atlas_path:Path, signal_channel = 2, control_channel = 3):
    control_channel = str(control_channel); signal_channel= str(signal_channel)
    data = {} #we want to output a structured data dictionary
    # NOTE: Signal_data is what we fit the probe geometry to
    if signal_channel == str(None):
        data.update({'signal_data' :tifffile.imread(brainreg_atlas_path/f"downsampled.tiff")})
        print('No signal channel specified - loading downsampled.tiff')
    else:
        try:
            data.update({'signal_data' :tifffile.imread(brainreg_atlas_path/f"downsampled_{signal_channel}.tiff")})
        except Exception as e:
            print(f'No signal data found: \n {e}')
            
    try: #look for manually annotated points:
        subject_ID = brainreg_atlas_path.parent.stem
        data.update({'manual_points': maps.get_manually_annotated_points_df(subject_ID)})
    except Exception as e:
        print(f'No manual points found: \n {e}')
    #then we need the atlas volume id's in sample space coordinates (we map these using ALLEN_ATLAS_INFO_DF)
    data.update({'atlas_registration_data':tifffile.imread(brainreg_atlas_path/"registered_atlas.tiff")})
    # NOTE: loading data to transform 3D sample space coordinates to allan atlas coordinates
    for i in range(3):
        data.update({f'deformation_field_{i}': tifffile.imread(brainreg_atlas_path/f"deformation_field_{i}.tiff")})
    
    data.update({'boundaries':tifffile.imread(brainreg_atlas_path/'boundaries.tiff')})
    
    return data

# 1. Threshold signal via gamma+Otsu
def threshold_signal_gamma(data: np.ndarray,
                           brain_mask: np.ndarray = None,
                           gamma: float = 1.5) -> np.ndarray:
    '''We threshold the signal to separate dye signal from background autoflorescence.
    Returns: 
    - thresholded_signal: np.ndarray() [i,j,k]] voxel data boolean mask where True indicates signal above threshold.
    Parameters:
    - data: np.ndarray() [i,j,k]  voxel data with dye signal.
    - gamma: float, gamma exponent to apply to the data before thresholding. Default is 1.5.
    '''
    norm = data.astype(np.float32)
    norm -= norm.min(); norm /= norm.max()
    corr = norm ** gamma
    mask = brain_mask if brain_mask is not None else np.ones_like(corr, bool)
    thresh = threshold_otsu(corr[mask])
    thresholded_signal =  (corr > thresh) & mask
    n_signal_points = np.sum(thresholded_signal[:,:,:])
    if n_signal_points>200000:
        raise ValueError(f'{n_signal_points} voxels remain after thresholding. This is unusually high. \n Please take a closer look a the signal data. Consider adjusting gamma.')
    return thresholded_signal


# 2. Build signal DataFrame
def make_signal_df(data: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    coords = np.argwhere(mask)
    values = data[mask]
    df = pd.DataFrame(coords, columns=['i','j','k'])
    df['value'] = values
    df['norm_value'] = (values-values.min())/values.max() #normalise to [0,1]
    
    return df


# 3. Get probe 2D contacts
def get_probe_contacts_df(manufacturer: str = 'imec', probe_name: str = 'NP2014') -> pd.DataFrame:
    probe = pi.get_probe(manufacturer=manufacturer, probe_name=probe_name)
    df = pd.DataFrame(probe.contact_positions, columns=['probe_coords.x','probe_coords.y'])
    return df


# 4. Cluster signal points
def separate_probes_from_noise(signal_df: pd.DataFrame, 
                               n_probes:int, 
                               probe_ordering_axis:str,
                               DBSCAN_eps=50,
                               DBSCAN_min_samples=100):
    '''Separate signal into data. This applies cluster_signal() and reorder_signal_clusters() functions to signal data.'''
    points = signal_df[['i','j','k']].values
    df = signal_df.copy()
    if n_probes==2 and ORDER_SPLIT: # When there are just two probes, 
        clustered_probes = []
        for i in range(n_probes):
        # we have an advantage that we can separate two probes along probe ordering axis
        # then cluster each subset of points to denoise, and at the end append and reorder.
            centroid = np.median(points,axis=0)
            axis_idx = [x!=0 for x in AXIS2ATLAS_VECTOR[probe_ordering_axis]] # True for the axis that is ordered
            probe_mask = points[:,axis_idx]<centroid[axis_idx] # shaped (n_points,1)
            clustered_probe_df = cluster_signal(df.iloc[probe_mask[:,0]].copy(), #weird indexing here just due to mask shape (n_points,1)
                                                n_clusters=1,
                                                eps = DBSCAN_eps, min_samples = DBSCAN_min_samples)
            clustered_probe_df.loc[clustered_probe_df['cluster']==0,'cluster'] = i
            clustered_probes.append(clustered_probe_df)
        df = pd.concat(clustered_probes, ignore_index=True)
        df = reorder_signal_clusters(df,probe_ordering_axis)
    else:
        #if just one probe, or more than two probes, we have to do it all in one go.
        df = cluster_signal(df, n_clusters=n_probes, eps=DBSCAN_eps, min_samples=DBSCAN_min_samples)
        df = reorder_signal_clusters(df,probe_ordering_axis)
    return df
    
def cluster_signal(df, n_clusters:int, eps=50, min_samples=100,):
    """
    Separate signal into data from probes and noise from autoflorescence.
    
    Parameters:
    - df: DataFrame with columns 'i', 'j', 'k'
    - eps: DBSCAN distance required for two points to be considered neighbours. 
        Default is 50x10um = 500um which covers at least two neurpixel 2.0 shanks.
    - min_samples: minimum number of neighbouring points required to be grouped into a cluster.
    - n_clusters_hint: if provided, will auto-adjust eps to try to get this many clusters
    
    Returns:
    - DataFrame with added 'cluster' column (-1 for noise, 0+ for clusters)
    
    """
    df_copy = df.copy()
    coords = df_copy[['i', 'j', 'k']].values
    
    # Auto-tune eps if n_clusters_hint provided
    if n_clusters is not None:
        best_eps = eps
        best_diff = float('inf')
        #we want the smallest eps that gives us the right number of clusters
        for test_eps in tqdm(np.linspace(10, 200, 20)): 
            labels = DBSCAN(eps=test_eps, min_samples=min_samples).fit_predict(coords)
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            diff = abs(n_found - n_clusters)
            if diff < best_diff:
                best_diff = diff
                best_n_found = n_found
                best_eps = test_eps
                if best_diff == 0:
                    break
        eps = best_eps

    # Cluster
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    if best_diff!=0:
        if best_n_found - n_clusters <=2:
            print(f'Found {best_n_found-n_clusters} clusters too many. Assuming smallest cluster(s) to be noise and ignoring.')
            labels[labels>1]=-1 #setting the smaller clusters to -1, since they are ordered by size (0 biggest, 1 smaller, et.c.)
        else:
            raise ValueError(f'Failed to identify {n_clusters} clusters, at best finding {best_n_found}. \n Check signal data after thresholding. You may want to adjust min_samples as well.')

    df_copy['cluster'] = labels
    return df_copy

def reorder_signal_clusters(df:pd.DataFrame, probe_order_axis:str):
    '''Returns array of cluster labels ordered along the probe_order_axis.
    Params:
    ------
    clustered_signal_df: pd.Dataframe
        df with 'i','j','k' columns for voxel indices and 'cluster' column for cluster labels.
    '''
    #we take the median coordinate for each non-noise cluster
    median_cluster_coords = df.groupby('cluster').median(['i','j','k'])[['i','j','k']].loc[0:].values
    #then we project it onto the axis, and order them according to size along the axis.
    cluster_order = np.argsort(np.dot(median_cluster_coords,AXIS2ATLAS_VECTOR[probe_order_axis]))
    for i in range(len(cluster_order)):
        df.loc[df['cluster']==i,'cluster'] = cluster_order[i]
    return df


# 5. Fit plane to signal via PCA
def fit_plane_to_signal(signal_df: pd.DataFrame,
                        insertion_axis:str= 'si',
                        contact_face_axis:str= 'lr',
                        longer_than_wider: bool = True,
                        insertion_bias = 0.01) -> dict:
    """
    Fit a 2D plane to 3D points using PCA, outputting dict with centroid, normal, v_axis, u_axis, where u,v are aligned with probe width, and -depth respectively.

    Parameters:
    ------
    - signal_df: DataFrame with 'i', 'j', 'k' columns.
    - contact_axis: str() specifying axis along which the probe contacts are facing
    - insertion_axis: str() specifying axis along which the probe was inserted
    - longer_than_wider: bool, if True (as for neuropixel probes), assumes the probe is longer than it is wide.
    - insertion_bias: float (0 to 1), indicating how much to bias v_axis towards insertion_axis.
    Returns: dict() with
    ------
    - centroid: np.array() [i,j,k] coordinates in sample space corresponding to the centre of the signal 
    - surface_coord: np.array() [i,j,k] coordinates corresponding to most superior signal along v_axis.
    - v_axis: np.array() [i,j,k] coordinates corresponding to axis pointing from bottom of probe to the top.
    - u_axis: np.array() [i,j,k] coordinates corresponding to axis pointing from left to right of probe.
    - normal: np.array() [i,l,k] coordinates corresponding to axis pointing out from the probe contacts.
    
    Note:
    -----
    axis strings should be one of 'rl' (right-left), 'ap' (anterior-posterior), 'si' (superior-inferior)
    or their inverses 'lr','pa', and'is'. 
    This is for correspondence with probeinterface axes, 
    where 2D origin is the lower left of the probe i.e. the deepest and left-most contact (when looking at contact faces).
    This maps x,y axes of probe coordinates to u,v axes of the plane.
    """
    points = signal_df[['i','j','k']].values
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")
    centroid = np.median(points, axis=0)
    centered = points - centroid

    # Ideally we use a cool trick where PCA extracts length and width of the probe
    # in practice, some probe geometries mean that alignign along diagonal explains more variance
    # we therefore do regularised PCA that aligns more with insertion axis.
    # sample covariance (rowvar=False -> columns are variables i,j,k)
    S = np.cov(centered, rowvar=False)  # 3x3
    # unit vector for the insertion axis (expects your existing mapping)
    a = np.array(AXIS2ATLAS_VECTOR[insertion_axis]).astype(float)
    a = a / (np.linalg.norm(a) + 1e-12)
    # lambda scaled to total variance to keep bias dimensionless
    lam = float(insertion_bias) * np.trace(S)
    # regularized covariance that favors variance along insertion axis
    S_reg = S + lam * np.outer(a, a)
    # ---- eigen-decomposition (ascending from eigh) ----
    # Using eigh for symmetric matrices is more stable
    evals, evecs = np.linalg.eigh(S_reg)
    # sort descending by eigenvalue
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    # By default, first two eigenvectors span the plane; last is the normal
    v_axis = evecs[:, 0] if longer_than_wider else evecs[:, 1]
    u_axis = evecs[:, 1] if longer_than_wider else evecs[:, 0]
    normal = evecs[:, 2]  # least variance direction (orthogonal to plane)

    # ---- orientation standardization (keep your original logic) ----
    def _cosine_similarity(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)

    # v_axis should point from bottom (deep) to top -> opposite the insertion axis direction
    if _cosine_similarity(v_axis, AXIS2ATLAS_VECTOR[insertion_axis]) > 0:
        v_axis = -v_axis

    # normal should point the same way as the contact facing axis
    if _cosine_similarity(normal, AXIS2ATLAS_VECTOR[contact_face_axis]) < 0:
        normal = -normal

    # Set u so that (u, v, normal) is a right-handed frame and u points "left->right"
    # Keep your original sign logic for u:
    possible_v_axes = [ax for ax in AXIS2ATLAS_VECTOR.keys()
                       if ax[0] not in (insertion_axis + contact_face_axis)]
    only_one_axis_misaligned = np.allclose(
        AXIS2ATLAS_VECTOR[insertion_axis] + AXIS2ATLAS_VECTOR[contact_face_axis], 0
    )
    atlas_axis_to_align_to = possible_v_axes[1] if only_one_axis_misaligned else possible_v_axes[0]
    if _cosine_similarity(u_axis, AXIS2ATLAS_VECTOR[atlas_axis_to_align_to]) < 0:
        u_axis = -u_axis

    # Final orthonormal cleanup to avoid small numerical drift
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-12)
    u_axis = u_axis - np.dot(u_axis, v_axis) * v_axis
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-12)
    normal = np.cross(u_axis, v_axis)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    return {
        'centroid': centroid,
        'v_axis': v_axis,
        'u_axis': u_axis,
        'normal': normal
    }


def get_surface_coord_df(boundaries_data:dict)-> pd.DataFrame:
    '''Returns a pandas dataframe with brain surface coordinates ['i','j','k']'''
    surface_coords = []
    for ap_slice in range(boundaries_data.shape[0]):
        coords = np.argwhere(boundaries_data[ap_slice,:,:]==1)
        if len(coords)==0:
            continue
        else:
            for si_slice in np.unique(coords[:,0]):
                rl_indices = np.argwhere(boundaries_data[ap_slice,si_slice]==1)
                surface_coords.append([ap_slice,si_slice,np.min(rl_indices)]),
                surface_coords.append([ap_slice,si_slice,np.max(rl_indices)])

    surface_df = pd.DataFrame(np.array(surface_coords), columns=['i','j','k'])
    return surface_df

def append_surface_coord(surface_df:pd.DataFrame, plane_dict:dict)->dict:
    """Returns plane_dict with 'surface_coord' appended.
    This is the coordinate on the brain surface aligned with v_axis.
    """
    #lastly, we want a surface estimate for a probe, which is essentially the centroid shifted to the brain surface
    surface_coords = surface_df.values.astype(np.float64)
    vectors = surface_coords - plane_dict['centroid']
    # now we do cosine similarity in batch here
    numerator = vectors @ plane_dict['v_axis'] #(v dot u)
    denominator = np.linalg.norm(vectors,axis=1)*np.linalg.norm(plane_dict['v_axis']) #||v||*||u||
    cos_sim = np.divide(numerator,denominator)
    probe_surface_coord = surface_coords[int(np.argmax(cos_sim))]
    plane_dict.update({'surface_coord': probe_surface_coord})

    return plane_dict


# 6. Transform 2d probe contacts before projection
def transform_2d_probe(probe_df: pd.DataFrame, params_values: dict, voxel_size_um = VOXEL_SIZE) -> pd.DataFrame:
    """
    Transform 2D probe contacts by rotating and translating.
    Returns: 
    - pandas DataFrame with transformed 'probe_coords.x', 'probe_coords.y' columns.
    Params:
    - points_df: DataFrame with original 'probe_coords.x', 'probe_coords.y' columns.
    - params: [probe_depth, brain_shrinkage_pct, probe_width_scaling, theta, offset_x]
    
    Note: we want to imagine that we are projecting the probe by its depth starting from the top of the signal.
    """
    probe_depth, brain_shrinkage_pct, probe_width_scaling, theta, offset_x,  = params_values  # transformation parameters
    transformed_df = probe_df.copy() 
    transformed_df = transformed_df[transformed_df['probe_coords.y'] <= probe_depth]  # filter by depth
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
    - plane_info: dict with keys 'surface_coord', 'v_axis', 'u_axis' for a 2D plane in 3D space.
    NOTE: the origin of the projection is 'surface_coord'.
          i.e., the points_df should be transformed accordingly.
    """
    # Extract 2D coordinates
    u_coords = points_df['probe_coords.y'].values #u axis is aligned with insertion / depth of probe / probe height dimension
    v_coords = points_df['probe_coords.x'].values #v axis is aligned with width of the probe
    # Project onto 3D plane using the plane's coordinate system
    points_3d = (plane_info['surface_coord'][np.newaxis, :] + 
                 u_coords[:, np.newaxis] * plane_info['v_axis'][np.newaxis, :] + 
                 v_coords[:, np.newaxis] * plane_info['u_axis'][np.newaxis, :])
    return pd.DataFrame(points_3d, columns=['i', 'j', 'k'])

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
    signal_tree = KDTree(signal_df[['i', 'j', 'k']].values)
    #find distance to nearest neighbours
    signal2contact_dist, _ = probe_tree.query(signal_df[['i', 'j', 'k']].values, k=1)
    contact2signal_dist, _ = signal_tree.query(coords, k=1)
    distance_cost = np.sum(contact2signal_dist)/len(coords)+np.sum(signal2contact_dist*signal_df['norm_value'].values)/len(signal_df)
    #TODO: could add a cost for fitting a set of known probe channel -> brain area matches (based on LFP) {probe_coord : brain_area}.
    return distance_cost

# 9. estimate initial parameters for optimization:
def estimate_initial_params(signal_df: pd.DataFrame, probe_df:pd.DataFrame, plane: dict,
                            probe_depth_um: float = None) -> dict:
    """ Returns dictionary with good initial parameters for optimization.
    Will by default estimate probe_depth from signal"""
    
    if probe_depth_um is None:
        #Estimate depth of the probe (in sample space):
        centered_points = signal_df[['i', 'j', 'k']].values - plane['centroid']
        u_coords = np.dot(centered_points, plane['v_axis'])  # Projection onto u-axis, aligned with probe depth
        probe_depth_um = VOXEL_SIZE*(np.percentile(u_coords,99.9) - np.percentile(u_coords,0.1))  # Range of u-coordinates gives depth of probe
        
    #estimate the shrinkage of the brain by fitting iteratively and choosing best parameters
    #we can set initial offset and rotation to 0.0 thanks to careful alignment of the plane and projection centered on signal 
    #let's do a cost search
    min_cost = 1e42 #start very high
    for brain_shrinkage_pct in np.linspace(0,5,6): #try 0,1,2,...15 pct shrinkage
        for probe_width_scale in np.linspace(0.8,1,11):
            adjusted_probe_depth = probe_depth_um * (100 / (100 - brain_shrinkage_pct))  # scale to um and adjust for brain shrinkage.
            params = {'probe_depth':adjusted_probe_depth,  # Adjust depth for brain shrinkage in histology sample space
                    'brain_shrinkage_pct':brain_shrinkage_pct,
                    'probe_width_scaling':probe_width_scale,
                    'theta':1e-10,  # small nonzero rotation initially,
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
                         fixed_depth_um: float = None,
                         bounds: dict = {'depth':(500,6000),#depth
                                        'brain_shrinkage_pct':(0,5), #account for some pct brain shrinkage in signal positions
                                        'probe_width_scaling':(0.8,1), #shank width compression for NPXL2.0
                                        'rotation':(-np.pi/2, np.pi/2), #rotation
                                        'offset_x':(-1000, 1000), #offset along probe width in um
                                        }) -> dict:
    if fixed_depth_um is None:
        initial_params = estimate_initial_params(signal_df,probe_df, plane)
    else: #if initial parameters are given, assume fixed probe depth.
        initial_params = estimate_initial_params(signal_df,probe_df, plane, 
                                                 probe_depth_um = fixed_depth_um)
        bounds['depth'] = (fixed_depth_um, fixed_depth_um)
    def cost(param_values): #must be given as a function of a list of parameter values
        return compute_cost(signal_df, probe_df, plane, param_values)
    res = minimize(cost, 
                   x0 = list(initial_params.values()), 
                   bounds=list(bounds.values()), 
                   options = {'maxiter': 1000, 'maxls': 1000}) #help the constrained optimisation explore parameter space.
    best_params_dict = dict(zip(initial_params.keys(),res.x))
    return best_params_dict 

# 11. get structure labels
def get_structure_labels(points_array:np.array, data:dict):
    anatomy_dict = {'name':[],
                    'acronym':[],
                    'id':[]}
    for each_coord in points_array:
        int_coords = [int(x) for x in each_coord]
        volume_id = (data['atlas_registration_data'][int_coords[0],int_coords[1],int_coords[2]])
        try:
            info = ALLEN_ATLAS_INFO_DF.query(f'id=={volume_id}')
            anatomy_dict['id'].append(info.id.item())
            anatomy_dict['name'].append(info.name.item())
            anatomy_dict['acronym'].append(info.acronym.item())
        except Exception as e:
            anatomy_dict['name'].append('outside brain')
            anatomy_dict['acronym'].append(np.nan)
            anatomy_dict['id'].append(np.nan)
            #print(f'Failed to get info. assuming outside brain\n {e}')
            continue
    return anatomy_dict

# 12. transform sample space to allen space
def sample_coords_to_allen_space(points:np.array, 
                                 data:dict,):
    '''Not super trivial transformation from sample coordinates to allen coords.'''
    def_fields = [data[f'deformation_field_{x}'] for x in range(3)]
    atlas_mm = []
    for point in points:
        atlas_mm.append([def_fields[0][int(point[0]),int(point[1]),int(point[2])],
                        def_fields[1][int(point[0]),int(point[1]),int(point[2])],
                        def_fields[2][int(point[0]),int(point[1]),int(point[2])]])
    atlas_um = np.array(atlas_mm)*1000
    return atlas_um
