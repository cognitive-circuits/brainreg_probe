# brainreg_probe
This repository aims to automate the process of registering a probe to brain anatomy on par with manual tracing of signal data, while maintaining reproducibility. 

This is done by fitting probe geometry (loaded from probe interface) to signal data from Dil coated probes (see https://brainglobe.info/tutorials/silicon-probe-tracking.html).


[insert nice set of figures and GIF here]

# Tutorial
The following tutorial will be a step-by-step for processing histology data as efficiently as possible, organised as follows:

1. running brainreg (HPC cluster via SLURM)
2. probeinterface tracing (HPC cluster via SLURM)

We assume that the data is stored on a HPC server managed via SLURM. We recommend running the notebook for a smooth step-by-step walkthrough.
The tutorial here is meant to supplement this notebook.

## Step 0: installation and setup

Navigate to your code folder (see tip below) and clone the current repository:
```
git clone https://github.com/charlesdgburns/brainreg_probe.git
```

Next, you want to set up a a [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) environment called `histology` with all the required python packages:
```
conda create -f environment.yml 
conda activate histology
``` 


>[!TIP]
> All the code will run seamlessly if your data is organised as below. Otherwise please change the paths in the scripts.
>```
>.
>└── experiment/ 
>    ├── code/    <-- (make this your working directory)
>    │   └── brainreg_probe/  
>    └── data/ 
>        ├── raw_data/
>        │   └── histology/    <-- (RAW_HISTOLOGY_PATH)
>        │       └── <subject_ID>/
>        │           └── <brainsaw outputs>
>        └── preprocessed_data/
>            └── brainreg/    <-- (PREPROCESSED_BRAINREG_PATH)
>                └── <subject_ID>/ 
>```

In order to run the notebook or 
``` ```

## Step 1: run brainreg

>[!TIP] Data orientation and signal channel assignment can also be checked locally using `napari`. See the section **Manual data check and annotations** below.

Before running `brainreg` it is crucial to specify the orientation of the inputs. There's a helper function to check this:

```
from brainreg_probe import run_brainreg as rub  
rub.check_brain_orientations() 
```

> [!NOTE] 
> **Let's reorient ourselves**
> Orientation of data is important here and can be confusing. Voxel data is a big 3D stack of images which can be indexed in each dimension. However, the dimension of the voxel data is not always ordered as X,Y,Z.
>Brainreg output is reoriented to the allen brain atlas, which follows an image axis convention with origin at 'RAS', the right, anterior, superior corner. This corresponds to a brainglobe orientation of `asr` (if the brain is sliced along the anterior-posterior axis).
>
> | Direction | Acronym | Voxel Dimension | Image Axis | Example |
> |-----------|---------|-----------------|------------|---------|
> | Anterior → Posterior | `ap` | `[0]` or `i` | **Z** | `data[z, :, :]` |
> | Superior → Inferior | `si` | `[1]` or `j` | **Y** | `data[:, y, :]` |
> | Right → Left | `rl` | `[2]` or `k` | **X** | `data[:, :, x]` |
>
> The reason for this is that packages like `numpy` and `skimage` follow the same image data convention for data transfomrations (inhereting this convention from C++ image processing libraries).
> To avoid confusion with `X`, `Y`, `Z` ordering in the output, we refer to `downsample_coords` in voxel data dimension order `i`, `j`, `k`.


---
## Step 2: run `brainreg`
The conda environment should have brainreg 

---

# Manual data check and annotations
You may need to manually check the histology and annotate where the probe was, for example in case of a poor Dil signal.
This will have to be done locally.

### Step 1 install napari
install `napari` following their [instructions](https://napari.org/dev/tutorials/fundamentals/installation.html). 

### Step 2.1: check 

You may run `brainreg` on a local machine, making sure to use the following command:
```brainreg <input_path> <output_path> --additional <dye_channel_path> -v <Z voxel size> <Y voxel size> <X voxel size> --orientation <orientation> --atlas allen_mouse_10um --debug```

We refer to [brainglobe orientations docs](https://brainglobe.info/documentation/setting-up/image-definition.html).

