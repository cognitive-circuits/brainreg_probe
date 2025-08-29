'''Separated some code to extract manually annotated points from brainsaw histology data.
Read from xml file and then resampled into 10um voxel space.

Any questions or help needed? At me: @charlesdgburns'''


import numpy as np
import pandas as pd
from typing import Dict, Iterable, Literal, Optional

from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Union

# example usage

#points_path = rub.PREPROCESSED_BRAINREG_PATH/'MR41'/'MR41_manual.xml'
#recipe_path = '/ceph/behrens/Beatriz/beatriz/histology_Feb2023/BSG_MR41_MR34_MR33/MR41/recipe_MR41_240311_113852.yml'
#points_df = load_cellcounter_markers(points_path)
#voxel_sizes = get_voxel_sizes(recipe_path)
#downsampled_points = downsample_points_to_um_voxels(points_df, voxel_sizes)
#downsampled_points should be a pandas dataframe wtih 'i','j','k', coordinates 
#

## usage

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
        Columns: ['type','x','y','z'] (floats; 'type' int when present).
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
        tnode = mt.find("Type")
        type_id = int(tnode.text.strip()) if (tnode is not None and tnode.text) else None

        for m in mt.findall("Marker"):
            def num(tag):
                node = m.find(tag)
                if node is None or node.text is None or node.text.strip() == "":
                    return np.nan
                try:
                    return float(node.text.strip())
                except ValueError:
                    return np.nan

            rows.append({
                "type": type_id,
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
            rows.append({"type": np.nan, "x": num("MarkerX"), "y": num("MarkerY"), "z": num("MarkerZ")})

    df = pd.DataFrame(rows, columns=["type", "x", "y", "z"])
    if df["type"].notna().all():
        df["type"] = df["type"].astype(int)
    df[["x", "y"]] = df[["x", "y"]].astype(float)
    # z stays float to allow NaN
    return df


def get_voxel_sizes(recipe_path):
    import yaml

    with open(str(recipe_path), "r") as stream:
        try:
            params = yaml.safe_load(stream)
            return params["VoxelSize"]
        except yaml.YAMLError as exc:
            print(exc)




BinningMethod = Literal["floor", "round", "ceil"]

def _bin_method(method: BinningMethod):
    if method == "floor":
        return np.floor
    if method == "round":
        return np.rint
    if method == "ceil":
        return np.ceil
    raise ValueError("method must be 'floor', 'round', or 'ceil'")

def downsample_points_to_um_voxels(
    df: pd.DataFrame,
    voxel_sizes_um: Dict[str, float],
    target_um: Iterable[float] = (10.0, 10.0, 10.0),
    *,
    method: BinningMethod = "floor",
    apply_affine_um: Optional[np.ndarray] = None,
    keep_original_cols: bool = True,
    count_col: str = "n_points"
) -> pd.DataFrame:
    """
    Downsample point coordinates to a regular voxel grid in physical (µm) space.

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'x','y','z' measured in *index units* (pixels for X,Y; planes for Z).
    voxel_sizes_um : dict
        Microns per index unit, e.g. {'X': 4.386, 'Y': 4.386, 'Z': 20.0}.
    target_um : (tx, ty, tz)
        Target voxel size in µm (default 10 µm isotropic).
    method : {'floor','round','ceil'}
        How to assign a point to a voxel bin.
    apply_affine_um : np.ndarray, optional
        3x3 or 4x4 affine to apply in *µm space* BEFORE binning.
        - If 3x3, it’s interpreted as a 2D xy affine in homogeneous coords:
          [[a,b,tx],[c,d,ty],[0,0,1]] and z is unchanged.
        - If 4x4, it’s full 3D in homogeneous coords.
    keep_original_cols : bool
        Keep the original columns or not.
    count_col : str
        Name of the aggregated count column.

    Returns
    -------
    DataFrame with integer voxel indices: 'vx10','vy10','vz10' and counts per voxel.
    """
    # 1) Convert index-space coords to µm
    vx = float(voxel_sizes_um.get("X", voxel_sizes_um.get("x", 1.0)))
    vy = float(voxel_sizes_um.get("Y", voxel_sizes_um.get("y", 1.0)))
    vz = float(voxel_sizes_um.get("Z", voxel_sizes_um.get("z", 1.0)))
    coords_idx = df[["x", "y", "z"]].to_numpy(dtype=float)
    coords_um = coords_idx * np.array([vx, vy, vz], dtype=float)[None, :]

    # 2) Optional affine in µm space
    if apply_affine_um is not None:
        A = np.asarray(apply_affine_um, dtype=float)
        if A.shape == (3, 3):
            # 2D affine on XY; leave Z as-is
            xy1 = np.c_[coords_um[:, :2], np.ones(len(coords_um))]
            xy_um = (xy1 @ A.T)[:, :2]
            coords_um = np.c_[xy_um, coords_um[:, 2]]
        elif A.shape == (4, 4):
            xyz1 = np.c_[coords_um, np.ones(len(coords_um))]
            coords_um = (xyz1 @ A.T)[:, :3]
        else:
            raise ValueError("apply_affine_um must be 3x3 (2D) or 4x4 (3D)")

    # 3) Quantize to target grid
    tx, ty, tz = map(float, target_um)
    to_bin = _bin_method(method)
    vox = np.stack([
        to_bin(coords_um[:, 0] / tx),
        to_bin(coords_um[:, 1] / ty),
        to_bin(coords_um[:, 2] / tz),
    ], axis=1).astype(np.int64)

    out = df.copy() if keep_original_cols else pd.DataFrame(index=df.index)
    out["i"] = vox[:, 0]
    out["j"] = vox[:, 1]
    out["k"] = vox[:, 2]

    # 4) Aggregate counts per downsampled voxel
    agg = (
        out.groupby(["i", "j", "k"], as_index=False)
           .size()
           .rename(columns={"value": count_col})
    )
    return agg

# --- Convenience helper to build an affine from your YAML's StitchingParameters.affineMat ---
def affine2d_from_yaml_affineMat(affineMat_3x3: Iterable[Iterable[float]]) -> np.ndarray:
    """
    Returns a 3x3 homogeneous 2D affine usable as `apply_affine_um`.
    The YAML's StitchingParameters.affineMat is already in 3x3 form.
    """
    A = np.array(affineMat_3x3, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("affineMat must be 3x3")
    return A
