import os
import scipy.misc
import shutil


INPUTDIR = './driving_dataset'
OUTPUTDIR = './driving_dataset/scaled'

shutil.rmtree(OUTPUTDIR, ignore_errors=True)

if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

shutil.copy(INPUTDIR+"/data.txt",OUTPUTDIR)

i = 0;

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

    scipy.misc.imsave(OUTPUTDIR + "/" + str(i) + ".jpg", image)

    i += 1
