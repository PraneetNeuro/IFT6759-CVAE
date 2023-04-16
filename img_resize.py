import os
import cv2
from tqdm import tqdm
import numpy as np


# os.mkdir('resized_sketches')
# os.mkdir('resized_original')
input_size = 128
output_size = 268


# for i in tqdm(os.listdir('cvae_dataset/sketches')):
#     img = cv2.imread(os.path.join('cvae_dataset/sketches', i), cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (input_size, input_size))
#     cv2.imwrite(os.path.join('resized_sketches', i), img) 

for j in tqdm(os.listdir('cvae_dataset/original')):
    img = cv2.imread(os.path.join('cvae_dataset/original', j))
    img = cv2.resize(img, (output_size, output_size))
    cv2.imwrite(os.path.join('resized_original', j), img)


sketches = []
originals = []
# for i in tqdm(os.listdir('resized_sketches')):
#     img = cv2.imread(os.path.join('resized_sketches', i), cv2.IMREAD_GRAYSCALE)
#     sketches.append(img)
for j in tqdm(os.listdir('resized_original')):
    img = cv2.imread(os.path.join('resized_original', j))
    originals.append(img)


sketches = np.array(sketches)
originals = np.array(originals)

    
# np.save('npy_files/sketches.npy', sketches)
np.save('npy_files/originals.npy', originals)
