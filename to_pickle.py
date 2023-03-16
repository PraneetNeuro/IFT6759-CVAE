import os
import cv2
import numpy as np
from tqdm import tqdm 


def to_preprocessed_pickle(img_path, dest_path, input_size, output_size, architecture, type):
    
    
    if type == 'sketch':
        if os.path.exists(dest_path + '/sketches_'+architecture+'.npy'):
            return
        sketches = []
        for img in tqdm(os.listdir(img_path + '/sketches')):
            img = cv2.imread(os.path.join(img_path + '/sketches', img), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, input_size)
            img_resized = (img_resized - np.mean(img_resized))/np.std(img_resized)
            sketches.append(img_resized)
        sketches = np.array(sketches)
        np.save(dest_path + '/sketches_'+architecture+'.npy', sketches)
        return
    
    if type == 'original':    
        if os.path.exists(dest_path + '/originals_'+architecture+'.npy'):
            return
        images = []
        for img in tqdm(os.listdir(img_path + '/original')):
            img = cv2.imread(os.path.join(img_path + '/original', img))
            img_resized = cv2.resize(img, output_size)
            img_resized = (img_resized - np.mean(img_resized))/np.std(img_resized)
            images.append(img_resized)
        images = np.array(images)
        np.save(dest_path + '/originals_'+architecture+'.npy', images)
        return
    
    return 