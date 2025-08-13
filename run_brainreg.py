'''A script to send jobs to HPC cluster via SLURM for parallel processing of brainreg.
Code here also included to check orientation from downsampled stacks.
@charlesdgburns'''

## setup ## 
import os
import sys
import pandas as pd
from pathlib import Path
#to check data orientation:
import tifffile 
import matplotlib.pyplot as plt
from brainreg_probe import plot_util_func as puf

## Global variables ##
RAW_HISTOLOGY_PATH = Path("../data/raw_data/histology")               # Contains /<subject_ID>/brainsaw_output
PREPROCESSED_HISTOLOGY_PATH = Path("../data/preprocessed_data/brainreg")  # Will contain brainreg/<subject_ID>/<atlas_name>/outputs
JOBS_PATH = Path("./Jobs/brainreg")               # Stores SLURM scripts and logs
for jobs_folder in ["slurm", "out", "err"]:
        if not (JOBS_PATH/jobs_folder).exists():
            os.mkdir(JOBS_PATH/jobs_folder)
            
ATLAS_NAME = 'allen_mouse_10um'
# specify subjects here as a list, otherwise assume that they are subdirs to raw_histology_path
SUBJECT_IDS = [p.stem for p in RAW_HISTOLOGY_PATH.iterdir()]
# MAKE SURE TO CHECK THE RAW DATA ORIENTATION ON NAPARI. OPTION HERE TO SPECIFY PER SUBJECT.
# PLEASE RUN  'check_brain_orientations()' in accordance with brainglobe image space definition.
SUBJECT_ID2ORIENTATION = {each_subject:'psl' for each_subject in SUBJECT_IDS}
# SELECT CHANNELS: usually 2 is red, 3 is green, 4 is blue.
SIGNAL_CHANNEL = 2 #this is the channel with probe dye in it
REGISTRATION_CHANNEL = 3 #this is the channel we want to register to; green is less flourescent than red and less noisy than blue.


## top level function ##
def run_brainreg(overwrite = False):
    '''Submit jobs to HPC cluster via slurm to run brainreg'''
    brainreg_paths_df = get_brainreg_paths_df()
    if not overwrite:
        brainreg_paths_df = brainreg_paths_df[~brainreg_paths_df.brainreg_completed]
    if brainreg_paths_df.empty:
        print("All files have been registered. No new videos to track.")
        return
    for session_info in brainreg_paths_df.itertuples():
        print(f"Submitting brainreg for {session_info.subject_ID} to HPC")
        script_path = get_brainreg_SLURM_script(session_info)
        os.system(f"sbatch {script_path}")
    print("All brainreg jobs submitted to HPC. Check progress with 'squeue -u <username>'")


## subfunctions ##

def check_brain_orientations():
    for subject_ID in SUBJECT_IDS:
        subject_path = RAW_HISTOLOGY_PATH/subject_ID
        downsampled_files = [x for x in (subject_path/'downsampled_stacks/025_micron').iterdir() if '.tif' in str(x)]
        data = tifffile.imread(downsampled_files[0])
        first_axis_coords = [0,50,100,200]
        fig, ax = plt.subplots(1,len(first_axis_coords), figsize = (10, 10*len(first_axis_coords)))
        for i, first_idx in enumerate(first_axis_coords):
            contrast_adjusted = puf.adjust_contrast(data[first_idx])
            ax[i].imshow(contrast_adjusted, cmap = 'gray')
            ax[i].set_title(f'{subject_ID} | data[{first_idx},:,:]')
            ax[i].scatter(0,0, color='red', s=100, zorder=3)
        fig.tight_layout()
        plt.show()


def get_brainreg_paths_df(raw_data_path = RAW_HISTOLOGY_PATH):

    #initialise dataframe as a dictionary
    paths_dict = {'subject_ID':[], 
                  'input_path':[],
                  'output_path':[],
                  'signal_path':[],
                  'recipe_path':[],
                  'brainreg_completed':[],
                  'orientation':[]}
    for each_subject in SUBJECT_IDS:
        subject_histology_path = RAW_HISTOLOGY_PATH/each_subject/'stitchedImages_100'
        
        #for each_channel in os.listdir(subject_histology_path):
        paths_dict['subject_ID'].append(each_subject)
        paths_dict['input_path'].append(subject_histology_path/str(REGISTRATION_CHANNEL))
        paths_dict['output_path'].append(PREPROCESSED_HISTOLOGY_PATH/each_subject/ATLAS_NAME)
        paths_dict['signal_path'].append(subject_histology_path/str(SIGNAL_CHANNEL))
        paths_dict['recipe_path'].append(list(paths_dict['input_path'][-1].parent.parent.glob("recipe*"))[0])
        paths_dict['brainreg_completed'].append(paths_dict['output_path'][-1].exists())
        paths_dict['orientation'].append(SUBJECT_ID2ORIENTATION[each_subject])

    return pd.DataFrame(paths_dict)
    

def get_brainreg_SLURM_script(br_info, RAM="64GB", time_limit="23:59:00"):
    """
    Writes a SLURM script to run sleap tracking on the video from a session specified in video_info.
    Input: video_info: pd.Series, with columns: subject_ID, session_type, datetime, video_path (row from the output of get_video_paths_df())
    Output: script_path: str, path to the SLURM script (saved in mazeSLEAP/jobs/slurm/)
    """
    session_ID = f"{br_info.subject_ID}"
    voxel_sizes = get_voxel_sizes(br_info.recipe_path)
    script = f"""#!/bin/bash
#SBATCH --job-name=brainreg_{session_ID}
#SBATCH --output='{JOBS_PATH}/out/brainreg_{session_ID}.out'
#SBATCH --error='{JOBS_PATH}/err/brainreg_{session_ID}.err'
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

echo $SLURMD_NODENAME
source /etc/profile.d/modules.sh

echo "Loading Brainglobe module"
module load brainglobe/2024-03-01
nvidia-smi

echo "Running brainreg"

brainreg {br_info.input_path} {br_info.output_path} --additional {br_info.signal_path} -v {voxel_sizes['Z']} {voxel_sizes['Y']} {voxel_sizes['X']} --orientation {br_info.orientation} --atlas {ATLAS_NAME} --debug"""
    script_path = JOBS_PATH/'slurm'/f'brainreg_{session_ID}.sh'
    with open(script_path, "w") as f:
        f.write(script)

    return script_path

def get_voxel_sizes(recipe_path):
    import yaml

    with open(str(recipe_path), "r") as stream:
        try:
            params = yaml.safe_load(stream)
            return params["VoxelSize"]
        except yaml.YAMLError as exc:
            print(exc)

    