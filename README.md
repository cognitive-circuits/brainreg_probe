# Probe auto-registration
This repository aims to automate the process of registering a probe to brain anatomy on par with manual tracing of signal data, while maintaining reproducibility. 

This is done by fitting probe geometry (loaded from probe interface) to signal data from Dil coated probes (see https://brainglobe.info/tutorials/silicon-probe-tracking.html).


[insert nice set of figures and GIF here]

## Step 1: setting up conda environment

## Step 2: run brainreg
We refer to brainglobe orientations here (https://brainglobe.info/documentation/setting-up/image-definition.html).

> [!TIP]
> Orientation of data is important here and can be confusing. Voxel data is a big 3D stack of images which can be indexed in each dimension. Image data conventions index out as `stack[Z,Y,X]`, for a coordinate system where `X` corresponds to right-left axis, `Y` is superior-inferior, and `Z` is . The allen brain atlas uses image data conventions of 'RAS', so the origin is in  anterior, superior, right corner of the data. In Brainglobe convention this is referred to as `asr`.

> [!NOTE]
> ## Coordinate System Quick Reference
> 
> | Axis | Direction | Data dimension | Example |
> |------|-----------|-------------|---------|
> | **Z** | Anterior → Posterior | `0` | `data[z, :, :]` |
> | **Y** | Superior → Inferior | `1` | `data[:, y, :]` |
> | **X** | Right → Left | `2` | `data[:, :, x]` |