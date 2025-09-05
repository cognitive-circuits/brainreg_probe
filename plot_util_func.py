""" plotting utility functions

@charlesdgburns"""

## Setup ##

import plotly.graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import brainglobe_heatmap as bgh

## Global variables ## 

#Set to `True` to plot straight lines in allen sample space
SIMPLIFIED_PLOTTING = False
ATLAS_VOXEL_SIZE = 10 #in um

## functions ##

def plot_3d(signal_df: pd.DataFrame,
            probe_coords: np.ndarray = None,
            max_points: int = 5000,
            fig: go.Figure = None):
    df = signal_df.copy()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=df.i, y=df.j, z=df.k,
                               mode='markers', marker=dict(size=2, opacity=0.6)))
    if probe_coords is not None:
        fig.add_trace(go.Scatter3d(x=probe_coords[:,0], 
                                   y=probe_coords[:,1], 
                                   z=probe_coords[:,2],
                                   mode='markers', marker=dict(size=4, color='red')))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.show()
    return fig

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

def plot_sample_data_sections(signal_data: np.array, 
                              plane_centroid, 
                              manual_points: pd.DataFrame = None,
                              probe_df: pd.DataFrame = None,
                              fig_and_axes= None, fix_contrast = True):
    '''Plot sagittal and coronal sections of the signal data,
    centered the centroid of the plane fit to the probe signal data.
    
    Parameters:
    ----------
    data_dict: dict
        Dictionary containing 'signal_data' (3D voxel data downsampled to atlas space)
        and optionally 'manual_points' pd.dataframe with columns ['i','j','k']
    plane_centroid: list or np.ndarray
        [i,j,k] coordinates of the point at which the sections will be sliced through.
    m'''

    sagittal = signal_data[:,:,int(plane_centroid[2])].T
    coronal = signal_data[int(plane_centroid[0]),:,:]
    #transverse = data['signal_data'][:,plane_centroid[1],:] #rarely used, but here in case
    if fix_contrast:
        sagittal = adjust_contrast(sagittal)
        coronal = adjust_contrast(coronal)
    if fig_and_axes is None:
        fig, axes = plt.subplots(1,2, figsize =(10,3), width_ratios = [1.5,1])
    else:
        fig, axes = fig_and_axes
    axes[0].imshow(sagittal,cmap='gray')
    axes[0].axis('off')
    axes[0].set(title = 'Sagittal section')

    axes[1].imshow(coronal,cmap = 'gray')
    axes[1].axis('off')
    axes[1].set(title = 'Coronal section')
    
    # add manual points if they have been provided:
    if manual_points is not None:
        axes[0].scatter(manual_points['i'],manual_points['j'],
                        c='white',s=1, label='manual points')
        axes[1].scatter(manual_points['k'],manual_points['j'],
                        c='white',s=1, label='manual points')
    if probe_df is not None:
        axes[0].scatter(probe_df['i'],probe_df['j'],
                        c='r',s=1, label='probe fit', alpha = 0.05)
        axes[1].scatter(probe_df['k'],probe_df['j'],
                        c='r',s=1, label='probe fit', alpha = 0.05)
    return fig

def plot_atlas_data_sections(probe_atlas_coords, 
                             probe_plane_centroid, 
                             voxel_size = ATLAS_VOXEL_SIZE,
                             fig_and_axes:tuple = None, highlight_region = None):
    probe_plane_centroid
    if fig_and_axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, axes = fig_and_axes
    values = {highlight_region:1} #you can add more regions if you want
    scene1 = bgh.Heatmap(
        values,
        position=probe_plane_centroid,
        orientation="sagittal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        title="Sagittal section",
        hemisphere="left",
        cmap="Blues",
        vmin=0,
        vmax=2,
        format="2D",
    ).plot_subplot(fig, axes[0])
    axes[0].scatter(probe_atlas_coords[:,0]-probe_plane_centroid[0], 
                    probe_atlas_coords[:,1]-probe_plane_centroid[1], 
                    color='red',alpha=0.05, s=1)
    
    scene2 = bgh.Heatmap(
        values,
        position=probe_plane_centroid,
        orientation="frontal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        title="Coronal section",
        hemisphere="left",
        cmap="Blues",
        vmin=0,
        vmax=2,
        format="2D",
    ).plot_subplot(fig, axes[1])
    axes[1].scatter(probe_atlas_coords[:,2]-probe_plane_centroid[2], 
                    probe_atlas_coords[:,1]-probe_plane_centroid[1],
                    color='red',alpha=0.05, s=1)
    
    [x.axis('off') for x in axes]
    fig.tight_layout()
    return fig