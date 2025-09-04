"""
Histology Data Processing - Manual Point Extraction and Reorientation

This script extracts manually annotated points from brainsaw histology data,
reads from XML files, and resamples them into 10μm voxel space with proper
axis reorientation.

Author: @charlesdgburns
Improved for clarity and brainglobe compatibility
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tifffile
import xml.etree.ElementTree as ET
from typing import Union, Tuple
from nibabel.orientations import axcodes2ornt, ornt_transform

from brainreg_probe import run_brainreg as rub

VERBOSE = False
# =============================================================================
# MAIN API FUNCTION
# =============================================================================

def get_manually_annotated_points_df(subject_ID: str) -> pd.DataFrame:
    """
    Extract manually annotated points and convert to 10μm atlas voxel space.
    
    Parameters
    ----------
    subject_ID : str
        Subject identifier to process
        
    Returns
    -------
    pd.DataFrame
        Points with columns ['i', 'j', 'k'] in 10μm atlas voxel coordinates
        and 'norm_value' for downstream clustering
        
    Notes
    -----
    - Input points are assumed to be in 'psl' orientation (Posterior, Superior, Left)
    - Output points are transformed to 'asr' orientation (Anterior, Superior, Right)
    - The anterior-posterior flip happens during this transformation
    """
    # Get file paths for this subject
    path_df = rub.get_brainreg_paths_df()
    subject_paths = path_df[path_df.subject_ID == subject_ID].iloc[0]
    
    recipe_path = Path(subject_paths.recipe_path)
    manual_path = Path(subject_paths.output_path.parent) / f'{subject_paths.subject_ID}_manual_points.xml'
    
    # Load the manually annotated points from XML
    try:
        points_df = load_cellcounter_markers(manual_path)
        print(f"Loaded {len(points_df)} manual points from {manual_path}")
    except Exception as e:
        raise FileNotFoundError(f'Could not find manual points XML: {manual_path}\n{e}')
    
    # Get voxel sizes and image dimensions
    voxel_sizes = get_voxel_sizes(recipe_path)
    img_shape = get_img_shape(subject_paths.input_path)
    if VERBOSE:
        print(f"Original image shape (Z,Y,X): {img_shape}")
        print(f"Original voxel sizes (Z,Y,X): {voxel_sizes['Z']:.3f}, {voxel_sizes['Y']:.3f}, {voxel_sizes['X']:.3f} μm")
        
    # Transform points to atlas grid with proper axis reorientation
    atlas_points = transform_points_to_atlas_grid(
        points_df,
        img_shape=img_shape,
        original_voxel_sizes_um=(voxel_sizes['Z'], voxel_sizes['Y'], voxel_sizes['X']),
        atlas_voxel_sizes_um=(10, 10, 10),
        point_columns=("z", "y", "x"),  # XML points are in z,y,x order
        from_orientation="psl",  # histology orientation in i,j,k, i.e. z,y,x order 
        to_orientation="asr",     # Standard atlas orientation
        )
    
    # Add normalization value for clustering
    atlas_points['norm_value'] = 1.0
    if VERBOSE:
        print(f"Transformed to atlas space: {len(atlas_points)} points")
        print("Atlas coordinates range:")
        print(f"  i (anterior-posterior): {atlas_points['i'].min():.1f} to {atlas_points['i'].max():.1f}")
        print(f"  j (superior-inferior):  {atlas_points['j'].min():.1f} to {atlas_points['j'].max():.1f}")
        print(f"  k (right-left):         {atlas_points['k'].min():.1f} to {atlas_points['k'].max():.1f}")
        
    return atlas_points

# =============================================================================
# COORDINATE TRANSFORMATION
# =============================================================================

def transform_points_to_atlas_grid(
    points_df: pd.DataFrame,
    img_shape: Tuple[int, int, int],
    original_voxel_sizes_um: Tuple[float, float, float],
    atlas_voxel_sizes_um: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    point_columns: Tuple[str, str, str] = ("z", "y", "x"),
    from_orientation: str = "asr",
    to_orientation: str = "asr"
) -> pd.DataFrame:
    """
    Transform points from original image space to atlas voxel space.
    
    This function handles:
    1. Axis reorientation (e.g., posterior→anterior flip)
    2. Voxel size scaling
    3. Coordinate system transformation
    
    Parameters
    ----------
    points_df : pd.DataFrame
        Points with columns specified by point_columns
    img_shape : tuple
        Shape of original image as (dim0, dim1, dim2)
    original_voxel_sizes_um : tuple
        Voxel sizes in micrometers as (size0, size1, size2)
    atlas_voxel_sizes_um : tuple
        Target atlas voxel sizes in micrometers
    point_columns : tuple
        Column names in points_df corresponding to the three dimensions
    from_orientation : str
        Original orientation code (e.g., 'psl' = Posterior, Superior, Left)
    to_orientation : str
        Target orientation code (e.g., 'asr' = Anterior, Superior, Right)
        
    Returns
    -------
    pd.DataFrame
        Transformed points with columns ['i', 'j', 'k'] in atlas space
        
    Notes
    -----
    Orientation codes follow brainglobe convention:
    - First letter: anterior (a) or posterior (p)
    - Second letter: superior (s) or inferior (i) 
    - Third letter: left (l) or right (r)
    
    Common orientations:
    - 'asr': Anterior, Superior, Right (standard atlas)
    - 'psl': Posterior, Superior, Left (common histology)
    """
    print(f"Transforming coordinates from '{from_orientation}' to '{to_orientation}' orientation")
    
    # Extract point coordinates
    coords = points_df[list(point_columns)].to_numpy().astype(float)
    
    # Step 1: Reorient the coordinate axes
    # This is where the anterior-posterior flip happens!
    coords_reoriented, orientation_transform = reorient_coordinates(
        coords, img_shape, from_orientation, to_orientation)
    
    # Step 2: Convert voxel indices to physical coordinates (mm)
    original_voxel_sizes_mm = np.array(original_voxel_sizes_um) / 1000.0
    atlas_voxel_sizes_mm = np.array(atlas_voxel_sizes_um) / 1000.0
    
    # Reorder voxel sizes to match the new orientation
    reoriented_voxel_sizes_mm = reorder_voxel_sizes(
        original_voxel_sizes_mm, orientation_transform
    )
    
    # Convert to physical coordinates, then to atlas voxel indices
    physical_coords_mm = coords_reoriented * reoriented_voxel_sizes_mm
    atlas_voxel_coords = physical_coords_mm / atlas_voxel_sizes_mm
    
    # Create output dataframe
    result = points_df.copy()
    result[["i", "j", "k"]] = atlas_voxel_coords
    if VERBOSE:
        print(f"Applied orientation transform: {orientation_transform}")
        print(f"Reoriented voxel sizes (mm): {reoriented_voxel_sizes_mm}")
        
    return result

def reorient_coordinates(
    coords: np.ndarray, 
    img_shape: Tuple[int, int, int], 
    from_orientation: str, 
    to_orientation: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorient coordinate points between different axis orientations.
    
    This function handles the core axis transformations, including:
    - Axis permutations (e.g., swapping x and z)
    - Axis inversions (e.g., posterior → anterior flip)
    
    Parameters
    ----------
    coords : np.ndarray
        Input coordinates with shape (n_points, 3)
    img_shape : tuple
        Original image dimensions
    from_orientation : str
        Source orientation (e.g., 'psl')
    to_orientation : str
        Target orientation (e.g., 'asr')
        
    Returns
    -------
    reoriented_coords : np.ndarray
        Coordinates in the new orientation
    transform_matrix : np.ndarray
        The transformation matrix applied
    """
    # Convert brainglobe codes to nibabel format
    from_axes = orientation_code_to_nibabel(from_orientation)
    to_axes = orientation_code_to_nibabel(to_orientation)
    
    # Get transformation matrix
    from_ornt = axcodes2ornt(from_axes)
    to_ornt = axcodes2ornt(to_axes)
    transform_matrix = ornt_transform(from_ornt, to_ornt)
    
    # Apply transformation
    coords = np.asarray(coords, dtype=float)
    reoriented_coords = np.empty_like(coords)
    
    for output_axis in range(3):
        input_axis = int(transform_matrix[output_axis, 0])
        direction = int(transform_matrix[output_axis, 1])  # +1 normal, -1 flipped
        
        if direction == 1:
            # Normal direction: copy values directly
            reoriented_coords[:, output_axis] = coords[:, input_axis]
        else:
            # Flipped direction: invert relative to image boundary
            # This is where anterior-posterior flipping occurs!
            max_coord = img_shape[input_axis] - 1
            reoriented_coords[:, output_axis] = max_coord - coords[:, input_axis]
            if VERBOSE:
                print(f"Axis {input_axis} → {output_axis}: FLIPPED (max coord: {max_coord})")
    
    return reoriented_coords, transform_matrix

def reorder_voxel_sizes(voxel_sizes_mm: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Reorder voxel sizes to match reoriented coordinate axes.
    
    When we reorient coordinates, we also need to reorder the voxel sizes
    to match the new axis arrangement.
    """
    reordered_sizes = np.empty(3, dtype=float)
    
    for output_axis in range(3):
        input_axis = int(transform_matrix[output_axis, 0])
        reordered_sizes[output_axis] = voxel_sizes_mm[input_axis]
    
    return reordered_sizes

def orientation_code_to_nibabel(orientation_code: str) -> Tuple[str, str, str]:
    """
    Convert brainglobe orientation code to nibabel axis codes.
    
    Parameters
    ----------
    orientation_code : str
        3-letter orientation code (e.g., 'asr', 'psl')
        
    Returns
    -------
    tuple
        Corresponding nibabel axis codes
    """
    code_map = {
        'a': 'A',  # anterior
        'p': 'P',  # posterior  
        's': 'S',  # superior
        'i': 'I',  # inferior
        'l': 'L',  # left
        'r': 'R'   # right
    }
    
    if len(orientation_code) != 3:
        raise ValueError(f"Orientation code must be 3 letters, got: {orientation_code}")
    
    return tuple(code_map[char.lower()] for char in orientation_code)

# =============================================================================
# FILE I/O FUNCTIONS  
# =============================================================================

def load_cellcounter_markers(xml_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load manually annotated points from Fiji/Napari CellCounter XML format.
    
    Parameters
    ----------
    xml_path : str or Path
        Path to the XML file containing marker annotations
        
    Returns
    -------
    pd.DataFrame
        Points with columns ['x', 'y', 'z'] (floats)
        
    Notes
    -----
    Handles both nested (Marker_Type > Marker) and flat (Marker_Data > Marker)
    XML structures. Missing coordinates are filled with NaN.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # Parse XML
    root = ET.parse(str(xml_path)).getroot()
    marker_data = root.find("Marker_Data")
    
    if marker_data is None:
        raise ValueError("No <Marker_Data> section found in XML file")
    
    points = []
    
    # Try nested structure first (Marker_Type > Marker)
    marker_types = marker_data.findall("Marker_Type")
    if marker_types:
        print(f"Found {len(marker_types)} marker types in XML")
        for marker_type in marker_types:
            markers = marker_type.findall("Marker")
            if VERBOSE:
                print(f"  Processing {len(markers)} markers in type")
            for marker in markers:
                points.append(_extract_marker_coordinates(marker))
    else:
        # Try flat structure (Marker_Data > Marker)
        markers = marker_data.findall("Marker")
        print(f"Found {len(markers)} markers directly under Marker_Data")
        for marker in markers:
            points.append(_extract_marker_coordinates(marker))
    
    if not points:
        raise ValueError("No markers found in XML file")
    
    # Create DataFrame
    df = pd.DataFrame(points, columns=["x", "y", "z"])
    df = df.astype(float)
    
    # Report any missing coordinates
    missing_coords = df.isnull().sum()
    if missing_coords.any():
        print(f"Warning: Missing coordinates found: {dict(missing_coords)}")
    
    return df

def _extract_marker_coordinates(marker_element) -> dict:
    """Extract x, y, z coordinates from a marker XML element."""
    def get_coordinate(tag_name: str) -> float:
        element = marker_element.find(tag_name)
        if element is None or not element.text or element.text.strip() == "":
            return np.nan
        try:
            return float(element.text.strip())
        except ValueError:
            return np.nan
    
    return {
        "x": get_coordinate("MarkerX"),
        "y": get_coordinate("MarkerY"), 
        "z": get_coordinate("MarkerZ")
    }

def get_voxel_sizes(recipe_path: Union[str, Path]) -> dict:
    """
    Extract voxel sizes from brainreg recipe YAML file.
    
    Parameters
    ----------
    recipe_path : str or Path
        Path to the recipe YAML file
        
    Returns
    -------
    dict
        Voxel sizes with keys ['X', 'Y', 'Z'] in micrometers
    """
    import yaml
    
    recipe_path = Path(recipe_path)
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")
    
    with open(recipe_path, 'r') as file:
        try:
            params = yaml.safe_load(file)
            voxel_sizes = params["VoxelSize"]
            print(f"Loaded voxel sizes from {recipe_path}: {voxel_sizes}")
            return voxel_sizes
        except (yaml.YAMLError, KeyError) as e:
            raise ValueError(f"Could not parse voxel sizes from {recipe_path}: {e}")

def get_img_shape(brainreg_input_path: Path) -> Tuple[int, int, int]:
    """
    Determine the shape of the original image stack.
    
    Parameters
    ---------- 
    brainreg_input_path : Path
        Directory containing the image files
        
    Returns
    -------
    tuple
        Image shape as (n_slices, height, width)
        
    Notes
    -----
    Assumes all images have the same height and width, and counts
    the number of files to determine the number of slices.
    """
    input_path = Path(brainreg_input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    
    # Count image files (assuming they're all the same size)
    image_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    if not image_files:
        raise ValueError(f"No TIFF files found in {input_path}")
    
    n_slices = len(image_files)
    
    # Get dimensions from first image
    first_image = tifffile.imread(image_files[0])
    height, width = first_image.shape[:2]  # Handle both 2D and 3D arrays
    
    shape = (n_slices, height, width)
    if VERBOSE:
        print(f"Determined image shape: {shape} from {n_slices} files")
    
    return shape

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    subject_id = "MR41"  # Replace with your subject ID
    
    try:
        points_df = get_manually_annotated_points_df(subject_id)
        print(f"\nSuccessfully processed {len(points_df)} points for subject {subject_id}")
        print("\nFirst few points in atlas space:")
        print(points_df[['i', 'j', 'k', 'norm_value']].head())
        
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")