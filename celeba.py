# File to generate the celeba dataset from a zip file
# First download zip file (align and cropped) from
# https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
# The deform part is not ready, we need to first remove background
from dataset import Dataset
import zipfile
from PIL import Image
import numpy as np
import cv2
import os


if __name__ == '__main__':
    n_images = 5  # number of images to extract
    path_to_images = 'data/celeba/img_align_celeba.zip'  # path to zip file
    path_to_save = 'data/celeba/test'  # path to extract (will save a bunch .jpg to use Praneet Dataset class)
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Loop to extract from zip file without unzipping all
    with zipfile.ZipFile(path_to_images, "r") as z:
        i = 0
        for name in z.namelist()[1:n_images]:  # Don't take the first because it's the folder not a file
            img_bytes = z.open(name)
            img_data = Image.open(img_bytes)
            image_as_array = np.array(img_data, np.uint8)
            print(image_as_array.shape)
            image_as_array = cv2.cvtColor(image_as_array, cv2.COLOR_RGB2BGR)
            # Next line was introduce directly to Dataset class
            # X_deformed = elasticdeform.deform_random_grid(np.mean(image_as_array, axis=-1), sigma=8, points=3)

            cv2.imwrite(os.path.join(path_to_save, str(i)+'.jpg'), image_as_array)
            # cv2.imwrite(os.path.join(path_to_save, str(i) + '_deformed.jpg'), X_deformed)
            i += 1

    data = Dataset(path_to_save, img_size=(218, 178), n_images=n_images, sigmaX=3, sigmaY=3, ksize=21, deform=False)
    data.transform_all_images()
    data.save_sketches('sketches')
    data.save_as_pickle()
