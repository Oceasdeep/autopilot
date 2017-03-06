"""Image preprocessor for the deep learning model for self-driving cars.

This script preprocesses the dataset by first cropping and then scaling
the images to the final image size of 200x66. This script needs to be run
once.
"""
import os
import scipy.misc
import shutil

# Folder locations
INPUTDIR = './driving_dataset'
OUTPUTDIR = './driving_dataset/scaled'

# Delete the old output directory
shutil.rmtree(OUTPUTDIR, ignore_errors=True)

# Create output directory if not present
if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

# Copy label file from the image dataset to the output directory
shutil.copy(INPUTDIR+"/data.txt",OUTPUTDIR)

i = 0;

# Loop through all images in the image dataset
while(True):

    # Read 256 x 455 RGB image
    try:
        full_image = scipy.misc.imread(INPUTDIR + "/" + str(i) + ".jpg", mode="RGB")
    except:
        break

    # Crop to last 150 rows
    cropped_image = full_image[-150:]

    # Resize to 66 x 200 and scale to interval [0,1]
    image = scipy.misc.imresize(cropped_image, [66, 200])

    # Write the cropped and scaled image to a file
    scipy.misc.imsave(OUTPUTDIR + "/" + str(i) + ".jpg", image)

    # increment image index
    i += 1
