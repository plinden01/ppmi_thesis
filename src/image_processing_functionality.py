import os
import sys
import random
import numpy as np
import nibabel as nib
import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import pyelastix
from scipy import ndimage
import string
#import csv
#import SimpleITK as sitk
#import matplotlib.pyplot as plt
#import PIL.Image

# def read_nifti_file(filepath):
#     scan = nib.load(filepath)
#     scan = scan.get_fdata()
#     return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

# def resize_volume(img):
#     desired_depth = 64
#     desired_width = 128
#     desired_height = 128
#     # Get current depth
#     current_depth = img.shape[-1]
#     current_width = img.shape[0]
#     current_height = img.shape[1]
#     # Compute depth factor
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     # Rotate
#     img = ndimage.rotate(img, 90, reshape=False)
#     # Resize across z-axis
#     img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
#     return img

def register_and_difference_image(im_moving_arr, im_fixed_arr):
    params = pyelastix.get_default_params(type='RIGID')
    params.MaximumNumberOfIterations = 1000
    (im_registered_arr, field) = pyelastix.register(im_moving_arr, im_fixed_arr, params)
    im_difference_arr = abs(im_registered_arr - im_fixed_arr)
    return im_difference_arr

# def process_scan(path):
#     """Read and resize volume"""
#     # Read scan
#     volume = read_nifti_file(path)
#     # Normalize
#     volume = normalize(volume)
#     # Resize width, height and depth
#     volume = resize_volume(volume)
#     return volume
#######################################################################################
def process_and_subtract(path1,path2):
    im_fixed_arr = []
    im_moving_arr = []
    for baseline in os.scandir(path1):
        if (baseline.path.endswith(".nii")):
            baseline_file = nib.load(baseline)
            baseline_data = np.ascontiguousarray(baseline_file.get_fdata())
            im_fixed_arr.append(baseline_data)
    for followup in os.scandir(path2):
        if (followup.path.endswith(".nii")):
            followup_file = nib.load(followup)
            followup_data = np.ascontiguousarray(followup_file.get_fdata())
            im_moving_arr.append(followup_data)
    print("The number of images in BL are: ", len(im_fixed_arr))
    print("The number of images in 48m are: ", len(im_moving_arr))
    im_difference_arr = []
    if(len(im_fixed_arr) == len(im_moving_arr)):
        i = 0
        while i < len(im_moving_arr):
            im_difference = register_and_difference(im_moving_arr[i], im_fixed_arr[i])
            im_difference_arr.append(abs(im_difference))
            i += 1
        print("The number of differenced images are: ",len(im_difference_arr))
    else:
        sys.exit("array mismatch between BL and 48mo")
    return(im_difference_arr)

########################################################################################


##################
##################
##################
##################
# do registration and all that business, and then add to master list [filename[5:9], numpy_array_of_filename]
# Read csv
# iterate through master list, for each patno [el[0]]:
    # if said pathno is above threshold, label as 1
    # if said patno is below threshold, label as 0

directory1 = r'PD_FA/BL'
directory2 = r'PD_FA/48mo'
difference_array = process_and_subtract(directory1,directory2)
# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.

patno_reference = {}
worse_outcomes = []
better_outcomes = []
with open('PD_FA/patientlist.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        patno_reference[row[0]] = row[7]
for el in difference_array:
    if(patno_reference[el] == 'TRUE'):
        worse_outcomes.append(el)
    else:
        better_outcomes.append(el)

worse_scans = np.array([scan for scan in worse_outcomes])
better_scans = np.array([scan for scan in worse_outcomes])

worse_labels = np.array([1 for _ in range(len(worse_scans))])
better_labels = np.array([1 for _ in range(len(better_scans))])
# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((worse_scans[:26], better_scans[:26]), axis=0)
y_train = np.concatenate((worse_labels[:26], better_labels[:26]), axis=0)
x_val = np.concatenate((worse_scans[26:38], better_scans[26:38]), axis=0)
y_val = np.concatenate((worse_labels[26:38], better_labels[26:38]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)
