import os
import numpy as np
import cv2
from tqdm import tqdm
from rembg import remove
import Augmentor
import shutil

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
    
    def save_as_pickle(self, path_to_save='sketches'):
        if not hasattr(self, 'dataset'):
            self.transform_all_images()
        originals = np.array([(img - np.mean(img) / np.std(img)) if not self.resize else cv2.resize((img - np.mean(img) / np.std(img)), (self.target_image_size, self.target_image_size)) for img, _ in self.dataset])
        sketches = np.array([sketch for _, sketch in self.dataset])
        print(sketches.shape)
        np.save('originals.npy', originals)
        if self.simpler:
            path_to_save = path_to_save + '_simpler'
        if self.remove_bg:
            path_to_save = path_to_save + '_no_bg'
        np.save(path_to_save+'.npy', sketches)

def distortion(source_path, n_images=-1, save_path='distorted', replace=True):
    '''
    To generate distorted sketches from sketches folder
    :param source_path (str): Folder containing all sketches to be treated
    :param n_images (int): Number of sketches to transform (default is all)
    :param save_path: Where to save distorted sketches
    :return: None
    '''
    # We need a temporary folder to make sure every sketch is treated exactly one time
    # This is because that Augmentor normally picks a file randomly from the source folder to apply a deformation
    temp_folder = os.path.join(source_path, 'temp')
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
        os.mkdir(temp_folder)
    else:
        os.mkdir(temp_folder)  # Create a temporary folder
        print('folder created')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file_name in os.listdir(source_path)[:n_images]:
        file_path = os.path.join(source_path, file_name)
        temp_file_path = os.path.join(temp_folder, file_name)
        # Move the original sketch to the temp folder
        os.rename(file_path, temp_file_path)
        # Setting the pipeline to the folder containing only one sketch
        p = Augmentor.Pipeline(temp_folder)
        p.random_distortion(probability=1, grid_width=3, grid_height=3, magnitude=5)
        # p.gaussian_distortion(probability=1, grid_width=3, grid_height=3, magnitude=5, corner='bell', method='in')
        # Generate one random deformation
        p.sample(1)
        # Return the original sketch to the source folder, the distorted is in source/output
        os.rename(temp_file_path, file_path)

        # Move the distorted file and rename it
        for distorted_name in os.listdir(temp_folder + '/output'):
            distorted_path = os.path.join(temp_folder + '/output', distorted_name)
            new_distorted_path = save_path + '/' + file_name
            try:
                os.rename(distorted_path, new_distorted_path)
            except Exception as e:
                if replace:
                    os.remove(new_distorted_path)
                    try:
                        os.rename(distorted_path, new_distorted_path)
                    except Exception as e:
                        print(e)
                else:
                    print(f'{new_distorted_path} already exists, set replace to True or delete it. Deleting new file.')
                    os.remove(distorted_path)
    os.rmdir(temp_folder + '/output')
    os.rmdir(temp_folder)


def compile_into_npy(save_path):
    distorted = [cv2.imread(os.path.join(save_path, file)) for file in os.listdir(save_path)]
    distorted = np.array(distorted)
    print(distorted.shape)
    np.save('distorted.npy', distorted)
