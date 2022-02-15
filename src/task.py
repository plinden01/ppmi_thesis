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
def register_image(im_moving_arr, im_fixed_arr):
    params = pyelastix.get_default_params(type='RIGID')
    params.MaximumNumberOfIterations = 1000
    (im_registered_arr, field) = pyelastix.register(im_moving_arr, im_fixed_arr, params)
    im_difference_arr = abs(im_registered_arr - im_fixed_arr)
    return im_difference_arr
def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
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
im1 = file1.get_fdata()
im1_ = np.ascontiguousarray(file1.get_fdata())

# fig, ax1 = plt.subplots(1,1, 'all')
# i = 0
# for el in im1[90:91]:
#     ax1.imshow(el,cmap = 'bone')
#     addr = 'ppmi_ex/BL_getfdata'+str(i)+'.png'
#     print(addr)
#     fig.savefig(addr)
#     i += 1

# # 48 mo unregistered
# fig, ax1 = plt.subplots(1,1, 'all')
# for el in im1_[90:91]:
#     ax1.imshow(el,cmap = 'bone')
#     addr = 'ppmi_ex/BL_ascontiguous_arr'+str(i)+'.png'
#     fig.savefig(addr)
#     i += 1



##
#
#
#
#
##



def get_model(width=128, height=128, depth=64):
    #"""Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

# Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
fig, ax = plt.subplots(1,2,figsize=(20,3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )