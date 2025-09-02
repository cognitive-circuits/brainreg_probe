'''Separated some code to extract manually annotated points from brainsaw histology data.
Read from xml file and then resampled into 10um voxel space.

Any questions or help needed? At me: @charlesdgburns'''


import numpy as np
import pandas as pd

from pathlib import Path
import tifffile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Union
from nibabel.orientations import axcodes2ornt, ornt_transform

from brainreg_probe import run_brainreg as rub

## Global variables ##


# example usage

#points_path = rub.PREPROCESSED_BRAINREG_PATH/'MR41'/'MR41_manual.xml'
#recipe_path = '/ceph/behrens/Beatriz/beatriz/histology_Feb2023/BSG_MR41_MR34_MR33/MR41/recipe_MR41_240311_113852.yml'
#points_df = load_cellcounter_markers(points_path)
#voxel_sizes = get_voxel_sizes(recipe_path)
#downsampled_points = downsample_points_to_um_voxels(points_df, voxel_sizes)
#downsampled_points should be a pandas dataframe wtih 'i','j','k', coordinates 
#

## usage

def get_manually_annotated_points_df(subject_ID):
    '''Input: path_to_yaml_recipe, path_to_xml_manual_points
    Returns: pandas dataframe with 'i','j','k' coordinates of manually annotated points in 10um voxel space'''
    path_df = rub.get_brainreg_paths_df()
    subject_paths = path_df[path_df.subject_ID == subject_ID].iloc[0]
    recipe_path = Path(subject_paths.recipe_path)
    manual_path = Path(subject_paths.output_path.parent)/f'{subject_paths.subject_ID}_manual.xml'
    try:
        points_df = load_cellcounter_markers(manual_path)
    except Exception as e:
            FileNotFoundError(f'Could not find manual points xml: \n {e}')
    voxel_sizes = get_voxel_sizes(recipe_path)
    img_shape = get_img_shape(subject_paths.input_path)

    downsampled_points = points_to_atlas_grid(
        points_df,
        img_shape=img_shape,
        sample_vox_um=(voxel_sizes['Z'], voxel_sizes['Y'], voxel_sizes['X']),
        atlas_vox_um=(10, 10, 10),
        icol="z", jcol="y", kcol="x",
        from_axcodes=("P","S","L"),
        to_axcodes=("A","S","R"))
    downsampled_points['norm_value'] = 1 #add normalised value for later clustering
    return downsampled_points
#
#


def load_cellcounter_markers(xml_source: Union[str, Path]) -> pd.DataFrame:
    """
    Load markers from a Fiji/Napari CellCounter-style XML.

    Parameters
    ----------
    xml_source : str | Path
        Filesystem path to the XML, or a raw XML string.

    Returns
    -------
    pd.DataFrame
        Columns: ['x','y','z'] (floats; 'type' int when present).
    """
    # Determine if it's a path or raw XML text
    if isinstance(xml_source, (str, Path)) and Path(str(xml_source)).exists():
        root = ET.parse(str(xml_source)).getroot()
    else:
        root = ET.fromstring(str(xml_source))

    marker_data = root.find("Marker_Data")
    if marker_data is None:
        raise ValueError("No <Marker_Data> section found in XML.")

    rows = []

    # Preferred structure: multiple <Marker_Type> blocks
    for mt in marker_data.findall("Marker_Type"):
##
        for m in mt.findall("Marker"):
            def num(tag):
                node = m.find(tag)
                if node is None or node.text is None or node.text.strip() == "":
                    return np.nan
                try:
                    return float(node.text.strip())
                except ValueError:
                    return np.nan

            rows.append({##
                "x": num("MarkerX"),
                "y": num("MarkerY"),
                "z": num("MarkerZ"),  # may be missing -> NaN
            })

    # Rare variant: markers directly under <Marker_Data>
    if not rows:
        for m in marker_data.findall("Marker"):
            def num(tag):
                node = m.find(tag)
                if node is None or node.text is None or node.text.strip() == "":
                    return np.nan
                try:
                    return float(node.text.strip())
                except ValueError:
                    return np.nan
            rows.append({ "x": num("MarkerX"), "y": num("MarkerY"), "z": num("MarkerZ")})

    df = pd.DataFrame(rows, columns=["x", "y", "z"])
    df[["x", "y","z"]] = df[["x", "y","z"]].astype(float)
    return df

def get_voxel_sizes(recipe_path):
    import yaml

    with open(str(recipe_path), "r") as stream:
        try:
            params = yaml.safe_load(stream)
            return params["VoxelSize"]
        except yaml.YAMLError as exc:
            print(exc)

def get_img_shape(subject_brainreg_input_path: Path) -> tuple:
    # we need to get the img_shape
    shape_i = 0
    for img_path in subject_brainreg_input_path.iterdir():
        #i'th coordinate is as long as the number of files
        shape_i += 1
        if shape_i == 1:
            # j and k are equal to the shape of the first file
            img = tifffile.imread(img_path)
            shape_j, shape_k = img.shape

    img_shape = (shape_i, shape_j, shape_k)
    return img_shape

def points_to_atlas_grid(
    df, 
    img_shape:tuple,                            # <- shape of your *original* image in I,J,K order
    sample_vox_um=(20.0, 4.386, 4.386),          # <- your raw voxel sizes, ordered in (I,J,K) = (Z,Y,X)
    atlas_vox_um=(10.0, 10.0, 10.0),              # <- e.g. Allen 10 µm
    icol="z", jcol="y", kcol="x",                # <- names of columns in df corresponding to i,j,k.
    from_axcodes=("P","S","L"),                  # <- raw image axcodes, ordered as (I,J,K) = (Z,Y,X)
    to_axcodes=("A","S","R"),                    # <- atlas axcodes (set to your atlas, standard Allen is 'asr'')
    
):
    pts = df[[icol, jcol, kcol]].to_numpy()

    # 1) reorient voxel indices
    pts_ro, ornt = _reorient_points_vox(pts, img_shape, from_axcodes, to_axcodes)

    # 2) convert to mm, then to atlas-voxel indices
    sample_vs_mm = np.asarray(sample_vox_um, dtype=float) / 1000.0
    atlas_vs_mm  = np.asarray(atlas_vox_um,  dtype=float) / 1000.0
    vs_ro_mm     = _reorient_voxsizes_mm(sample_vs_mm, ornt)

    atlas_mm  = pts_ro * vs_ro_mm                 # world coords in mm
    atlas_vox = atlas_mm / atlas_vs_mm            # float voxel coords on the atlas grid

    out = df.copy()
    out[["i","j","k"]] = atlas_vox
    return out

def _reorient_points_vox(coords, img_shape, from_axcodes, to_axcodes):
    """
    coords: (N,3) voxel indices in the *original* image orientation
    img_shape: (X, Y, Z) of the original image (needed for orientation flips)
    returns: coords in the target (atlas) orientation, plus the ornt map
    """
    from_ornt = axcodes2ornt(from_axcodes)
    to_ornt   = axcodes2ornt(to_axcodes)
    ornt      = ornt_transform(from_ornt, to_ornt)  # rows = out axes; cols=(in_axis, dir)
    coords = np.asarray(coords, dtype=float)
    out = np.empty_like(coords, dtype=float)
    for out_ax in range(3):
        in_ax = int(ornt[out_ax, 0])
        dirn  = int(ornt[out_ax, 1])  # +1 or -1
        if dirn == 1:
            out[:, out_ax] = coords[:, in_ax]
        else:
            out[:, out_ax] = (img_shape[in_ax] - 1) - coords[:, in_ax]
    return out, ornt

def _reorient_voxsizes_mm(sample_vs_mm, ornt):
    """permute voxel sizes to match the reoriented axes"""
    sample_vs_mm = np.asarray(sample_vs_mm, dtype=float)
    vs = np.empty(3, dtype=float)
    for out_ax in range(3):
        in_ax = int(ornt[out_ax, 0])
        vs[out_ax] = sample_vs_mm[in_ax]
    return vs
