import os
import sys
import numpy as np
import nibabel as nib
import csv
import SimpleITK as sitk
import pyelastix
from scipy import ndimage
import matplotlib.pyplot as plt
import PIL.Image
import string
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan
def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
# Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
# Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
# Rotate
    img = ndimage.rotate(img, 90, reshape=False)
# Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    #volume = normalize(volume)
    # Resize width, height and depth
    #volume = resize_volume(volume)
    return volume
data_path = ''
#'C:/Users/Peter/Documents/Thesis/sampleT1FAmap_0-48_s3107/PPMI/3107/FA_map-MRI/2011-04-13_12_07_16.0/S107254'
from nibabel.testing import data_path
filename1 = os.path.join(data_path,'PPMI_3107_MR_FA_map-MRI_Br_20130506104300521_S107254_I370155.nii')
filename2 = os.path.join(data_path,'PPMI_3107_MR_FA_map-MRI_Br_20160428115933183_S264294_I695755.nii')
#registered_file2 = sitk.Elastix(sitk.ReadImage(filename1),sitk.ReadImage(filename2),"translation")
#print(example_filename)
print(filename1)
# img1 = process_scan(filename1)
# img2 = process_scan(filename2)
file1 = nib.load(filename1)
file2 = nib.load(filename2)
im_fixed_arr = np.ascontiguousarray(file1.get_fdata())
im_moving_arr = np.ascontiguousarray(file2.get_fdata())

params = pyelastix.get_default_params(type='RIGID')
#params.transform = 'AffineTransform'
params.MaximumNumberOfIterations = 1000
#params.FinalGridSpacingInVoxels = 10

# Apply the registration (im1 and im2 can be 2D or 3D)
def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume
def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
(im_registered_arr, field) = pyelastix.register(im_moving_arr, im_fixed_arr, params)
im_difference_arr2 = im_registered_arr - im_fixed_arr
im_difference_arr = abs(im_difference_arr2)
im_diff_norm = normalize(im_difference_arr)
im_diff_norm = resize_volume(im_diff_norm)

img = nib.Nifti1Image(abs(im_difference_arr2), np.eye(4))
filename = os.path.join(os.getcwd(), 'saved_nifi_im.nii.gz')
nib.save(img, filename)
file = nib.load(filename)
sample_arr = np.ascontiguousarray(file.get_fdata())
print(sample_arr.shape())
#### ACCESSING ONE IMAGE FROM 3D nii
# from PIL import Image
# img = Image.fromarray(img_data_arr_norm[:,:,0], 'L')
# img.save("image.jpeg")


# dipy
# moving_data = img2.get_data()
# moving_affine = img2.affine
# template_data = img1.get_data()
# template_affine = img1.affine
# identity = np.eye(4)

# affine_map = AffineMap(identity,template_data.shape, template_affine,moving_data.shape, moving_affine)
# resampled = affine_map.transform(moving_data)
# regtools.overlay_slices(template_data, resampled, None, 0,"Template", "Moving")
# regtools.overlay_slices(template_data, resampled, None, 1,"Template", "Moving")
# regtools.overlay_slices(template_data, resampled, None, 2, "Template", "Moving")


# nbins = 32
# sampling_prop = None
# metric = MutualInformationMetric(nbins, sampling_prop)
# # The optimization strategy
# level_iters = [10, 10, 5]
# sigmas = [3.0, 1.0, 0.0]
# factors = [4, 2, 1]
# affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas,factors=factors)
#
# transform = TranslationTransform3D()
# params0 = None
# translation = affreg.optimize(template_data, moving_data, transform, params0, template_affine, moving_affine)
#
# transformed = translation.transform(moving_data)
# # regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Transformed")
# # regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Transformed")
# # regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Transformed")
#
# transform = RigidTransform3D()
# rigid = affreg.optimize(template_data, moving_data, transform, params0, template_affine, moving_affine, starting_affine=translation.affine)
#
# transformed = rigid.transform(moving_data)
# # regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Transformed")
# # regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Transformed")
# # regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Transformed")
#
# transform = AffineTransform3D()
# affreg.level_iters = [1000, 1000, 100]
# affine = affreg.optimize(template_data, moving_data, transform, params0, template_affine, moving_affine, starting_affine=rigid.affine)
# transformed = affine.transform(moving_data)
# # regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Transformed")
# # regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Transformed")regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Transformed")
#
# # # nonlinear time
# # from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
# # from dipy.align.imwarp import DiffeomorphicMap
# # from dipy.align.metrics import CCMetric
# # metric = CCMetric(3)
# # # The optimization strategy:
# # level_iters = [10, 10, 5]
# # # Registration object
# # sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
# # mapping = sdr.optimize(template_data, moving_data, template_affine, moving_affine, affine.affine)
# # warped_moving = mapping.transform(moving_data)
#
# # img1 = np.ascontiguousarray(template_data)
# # img2 = np.ascontiguousarray(warped_moving)
#
# img1 = template_data
# img2 = transformed
# img2_nonregistered = moving_data
#
# params = pyelastix.get_default_params()
# params.Transform = 'AffineTransform'

# (img1_registered, fields) = pyelastix.register(img1,img1,params)
# (img2_registered, fields) = pyelastix.register(img1,img2,params)
#img_dif = abs(img2_registered - img1)
#print(img1.shape,img2_registered.shape)
#img2_registered = alter_arr(img2_registered)
from skimage.util import montage

# BL
fig, ax1 = plt.subplots(1,1, 'all')
i = 0
for el in im_fixed_arr[90:100]:
    ax1.imshow(el,cmap = 'bone')
    addr = 'ppmi_bl/BL_normalized'+str(i)+'.png'
    print(addr)
    fig.savefig(addr)
    i += 1

# 48 mo unregistered
fig, ax1 = plt.subplots(1,1, 'all')
for el in im_registered_arr[90:100]:
    ax1.imshow(el,cmap = 'bone')
    addr = 'ppmi_48mo/48mo_registered_normalized'+str(i)+'.png'
    fig.savefig(addr)
    i += 1

# # 48mo registered
# fig, ax1 = plt.subplots(1,1, 'all')
# i = 0
# for el in im_diff_norm[90:100]:
#     ax1.imshow(el,cmap = 'bone')
#     addr = 'ppmi_difference_noabs/imagedifference_normalized'+str(i)+'.png'
#     fig.savefig(addr)
#     i += 1

#img3 = img2_registered - img1
# img3 = img2  - img1
# img3 = abs(img3)
i = 0
fig, ax1 = plt.subplots(1,1, 'all')
for el in sample_arr[90:100]:
    ax1.imshow(el,cmap = 'bone')
    addr = 'ppmi_difference/loaded_sample_nifi'+str(i)+'.png'
    fig.savefig(addr)
    i += 1

#img4 = abs(img2_registered - img1_registered)

# i = 0
# fig, ax1 = plt.subplots(1,1, 'all')
# for el in img4[90:100]:
#     ax1.imshow(el,cmap = 'bone')
#     addr = 'ppmi_difference_nonregistered/bothimagesregistered'+str(i)+'.png'
#     fig.savefig(addr)
#     i += 1

# def pix_average(arr):
#     summ = 0
#     avg = 0
#     for el in arr:
#         summ = summ + el
#     avg = summ / len(arr)
#     return avg
#im = (Image.open('imagedifference_normalized.png'))
#im = Image.eval(im,(lambda x: x*(165/200)))
#im = im.save('imagedifference2_normalized.png')
# el1 in im.getdata:
 #   for el2 in el1:
  #      for el3 in el2:
   #         if el3 > 0:
    #            el3 = el3 - 40
#print(im.getdata)
# pix_val = list(im.getdata())
# pix_val_flat = [x for sets in pix_val for x in sets]
# #for el in pix_val_flat:
# #    el = el - 40
# im2 = Image.open('BL_normalized.png')
# pix_val2 = list(im2.getdata())
# pix_val_flat2 = [x for sets in pix_val2 for x in sets]
# print("average for BL normalized is ",pix_average(pix_val_flat2))
# print("average for image difference is ",pix_average(pix_val_flat))
