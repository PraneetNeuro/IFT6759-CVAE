import os
import numpy as np
import cv2
from tqdm import tqdm
from rembg import remove


class Dataset:

    def __init__(self,
                 path_to_images,
                 source_image_size=None,
                 target_image_size=None,
                 n_images=5000,
                 lazy_load=False,
                 resize=False,
                 simpler=False,
                 remove_background=False):

        self.resize, self.simpler, self.remove_bg = resize, simpler, remove_background
        self.target_image_size = target_image_size
        self.imagefiles = [os.path.join(path_to_images, img_name) for img_name in os.listdir(path_to_images)]
        self.imagefiles = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg'),
                                      self.imagefiles
                                      ))[:n_images]

        if type(source_image_size) == int:
            self.img_size = (source_image_size, source_image_size)
        elif type(source_image_size) == tuple:
            self.img_size = source_image_size
        else:
            raise ValueError('img_size must be an integer or a tuple')
        
        self.lazy_load = lazy_load
        self.images = []
        if not lazy_load:
            self.load_images()

    def load_images(self):
        try:
            self.images = [cv2.imread(img) for img in self.imagefiles]
        except Exception as e:
            print(e)
    
    def img_to_sketch(self, img):

        # Consider this implementation as a starter
        # Could be tweaked further like increasing contrast for the edges, etc.
        # Let us study the transformations and work on improvising it before we move on to training the model
        # img = img - np.mean(img) / np.std(img)
        if self.remove_bg:
            img = remove(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_inv = 255 - img_gray
        if self.simpler:
            img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(3, 3), sigmaX=0, sigmaY=0)
            t1, t2 = 100, 150
            img_blend = cv2.Canny(image=img_blur, threshold1=t1, threshold2=t2)
            img_blend = 255 - img_blend
        else:
            img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
            img_blend = cv2.divide(img_gray, 255 - img_blur, scale=256)
        if self.resize:
            img_blend = cv2.resize(img_blend, self.img_size)

        return img_blend

    def transform_all_images(self):
        if self.lazy_load:
            self.load_images()
        self.dataset = [(img, self.img_to_sketch(img)) for img in self.images]
    
    def save_sketches(self, path_to_save):
        if not hasattr(self, 'dataset'):
            self.transform_all_images()
        if not os.path.exists('original'):
            os.mkdir('original')
        if self.simpler:
            path_to_save = path_to_save + '_simpler'
        if self.remove_bg:
            path_to_save = path_to_save + '_no_bg'
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        for i, (img, sketch) in tqdm(enumerate(self.dataset[::-1])):
            img = img - np.mean(img) / np.std(img)
            cv2.imwrite(os.path.join('original', f'{i}.jpg'), img if not self.resize else cv2.resize(img, (self.target_image_size, self.target_image_size)))
            cv2.imwrite(os.path.join(path_to_save, f'{i}.jpg'), sketch)
    
    def save_as_pickle(self):
        if not hasattr(self, 'dataset'):
            self.transform_all_images()

        originals = np.array([(img - np.mean(img) / np.std(img)) if not self.resize else cv2.resize((img - np.mean(img) / np.std(img)), (self.target_image_size, self.target_image_size)) for img, _ in self.dataset])
        sketches = np.array([sketch for _, sketch in self.dataset])

        np.save('originals.npy', originals)
        np.save('sketches.npy', sketches)
