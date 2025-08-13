# brainreg_probe
This repository aims to automate the process of registering a probe to brain anatomy on par with manual tracing of signal data, while maintaining reproducibility. 

This is done by fitting probe geometry (loaded from probe interface) to signal data from Dil coated probes (see https://brainglobe.info/tutorials/silicon-probe-tracking.html).


[insert nice set of figures and GIF here]

# Tutorial

## Step 1: setting up conda environment

## Step 2: run brainreg


We refer to brainglobe orientations here (https://brainglobe.info/documentation/setting-up/image-definition.html).



> [!NOTE]
> ### Let's reorient ourselves
> Orientation of data is important here and can be confusing. Voxel data is a big 3D stack of images which can be indexed in each dimension. However, the dimension of the voxel data is not always ordered as X,Y,Z.
>Brainreg output is reoriented to the allen brain atlas, which follows an image axis convention with origin at 'RAS', the right, anterior, superior corner. This corresponds to a brainglobe orientation of `asr` (if the brain is sliced along the anterior-posterior axis).
>
> | Direction | Acronym | Voxel Dimension | Image Axis | Example |
> |-----------|---------|-----------------|------------|---------|
> | Anterior → Posterior | `ap` | `[0]` | **Z** | `data[z, :, :]` |
> | Superior → Inferior | `si` | `[1]` | **Y** | `data[:, y, :]` |
> | Right → Left | `rl` | `[2]` | **X** | `data[:, :, x]` |
>
> The reason for this is that `numpy` and `skimage` packages follow the same image data convention for data transfomrations (inhereting this convention from C++ packages).
