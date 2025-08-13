""" plotting utility functions

@charlesdgburns"""

## Setup ##

import plotly.graph_objects as go
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


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
    coronal = signal_data[int(plane_centroid[0]),:,:]
    #transverse = data['signal_data'][:,plane_centroid[1],:] #rarely used, but here in case
    #adjust contrast
    sagittal = adjust_contrast(saggital)
    coronal = adjust_contrast(coronal)
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize =(10,3), width_ratios = [1.5,1])
    ax[0].imshow(sagittal,cmap='gray')
    ax[0].axis('off')
    ax[0].xaxis.set_inverted(True)
    ax[0].set(title = 'Sagittal section')

    ax[1].imshow(coronal,cmap = 'gray')
    ax[1].axis('off')
    ax[1].set(title = 'Coronal section')
    return fig
