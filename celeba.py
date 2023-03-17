# File to generate the celeba dataset from a zip file
# First download zip file (align and cropped) from
# https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
import shutil

from dataset import Dataset, distortion, compile_into_npy_and_zip
import zipfile
from PIL import Image
import numpy as np
import cv2
import os
import argparse
import multiprocessing


def check_dir(path):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


def loop_function(batch_idx):
    if not os.path.exists(path_to_save + f'_{batch_idx}'):
        os.mkdir(path_to_save + f'_{batch_idx}')
    # First we extract the batch
    begin = batch_idx * n_images
    end = (batch_idx + 1) * n_images
    if (partial == 1) & (batch_idx == n_batches):
        end = -1
    for file_path in list_of_files[begin:end]:
        img_bytes = z.open(file_path)
        img_data = Image.open(img_bytes)
        image_as_array = np.array(img_data, np.uint8)
        print(image_as_array.shape)
        image_as_array = cv2.cvtColor(image_as_array, cv2.COLOR_RGB2BGR)
        file_name = os.path.split(file_path)[1]
        cv2.imwrite(os.path.join(path_to_save + f'_{batch_idx}', file_name), image_as_array)

    # Now we transform the batch
    data = Dataset(path_to_save + f'_{batch_idx}', source_image_size=(218, 178), n_images=n_images, simpler=True)
    data.transform_all_images()
    data.save_sketches(f'sketches')  # Save as .jpg in 'original' and 'sketches/sketches_simpler'

    # Delete the extraction folder (images are saved in the global folder named original)
    shutil.rmtree(path_to_save + f'_{batch_idx}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data", type=check_dir, help="Directory to read the zip file")
    args = parser.parse_args()
    n_images = 500  # number of images per batch
    path_to_images = args.path_to_data+'/img_align_celeba.zip'  # path to zip file
    path_to_save = 'extracted'  # path to extract (temporary folders that will be deleted)
    #path_to_images = 'data/celeba/img_align_celeba.zip'
    #path_to_save = 'test'

    # Loop to extract from zip file without unzipping all
    with zipfile.ZipFile(path_to_images, "r") as z:
        list_of_files = z.namelist()
        list_of_files = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg'),
                                    list_of_files))
        total = len(list_of_files)
        n_batches = total // n_images
        if total % n_images != 0:
            partial = 1
        else:
            partial = 0
        # n_batches, partial = 2, 0 # For testing
        for batch_idx in range(n_batches + 1*partial):
            loop_function(batch_idx)

    # Add distortion to NOT compressed sketch files
    distortion(source_path=f'sketches_simpler', save_path=f'distorted')

    # Zip files
    shutil.make_archive('original', 'zip', 'original')
    shutil.make_archive('sketches_simpler', 'zip', 'sketches_simpler')
    shutil.make_archive('distorted', 'zip', 'distorted')

    # Remove not compressed folders
    shutil.rmtree('original')
    shutil.rmtree(f'sketches_simpler')
    shutil.rmtree(f'distorted')
