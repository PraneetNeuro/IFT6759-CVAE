import os
import numpy as np
import cv2

class Dataset:

    def __init__(self,
                 path_to_images,
                 img_size,
                 n_images,
                 lazy_load=False):
        
        self.imagefiles = [os.path.join(path_to_images, img_name) for img_name in os.listdir(path_to_images)][:n_images]
        self.imagefiles = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg'), self.imagefiles))

        if type(img_size) == int:
            self.img_size = (img_size, img_size)
        elif type(img_size) == tuple:
            self.img_size = img_size
        else:
            raise ValueError('img_size must be an integer or a tuple')
        
        self.lazy_load = lazy_load
        if not lazy_load:
            self.load_images()
        
    
    def load_images(self):
        try:
            print(self.img_size)
            print(self.imagefiles)

            self.images = [cv2.resize(cv2.imread(img), self.img_size) for img in self.imagefiles]
        except Exception as e:
            print(e)
    
    def img_to_sketch(self, img):

        # Consider this implementation as a starter
        # Could be tweaked further like increasing contrast for the edges, etc.
        # Let us study the transformations and work on improvising it before we move on to training the model

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_inv = 255 - img_gray
        img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
        img_blend = cv2.divide(img_gray, 255 - img_blur, scale=256)

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
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        for i, (img, sketch) in enumerate(self.dataset):
            cv2.imwrite(os.path.join('original', f'{i}.jpg'), img)
            cv2.imwrite(os.path.join(path_to_save, f'{i}.jpg'), sketch)
    
    def save_as_pickle(self):
        if not hasattr(self, 'dataset'):
            self.transform_all_images()
        
        originals = np.array([img for img, _ in self.dataset])
        sketches = np.array([sketch for _, sketch in self.dataset])

        np.save('originals.npy', originals)
        np.save('sketches.npy', sketches)

dataset = Dataset(path_to_images='images', img_size=256, n_images=1000, lazy_load=False)
dataset.save_sketches('sketches')



