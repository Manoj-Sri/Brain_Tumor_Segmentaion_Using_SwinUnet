import nibabel as nib
import pandas as pd
import numpy as np
from glob import glob
import os

from config import cfg

#returns 3D image from image path
def load_image(path):
    data = nib.load(path)
    image = data.get_fdata()

    return image

#resizes 240*240*depth shape to 256*256*depth images
def resize_image(image):
    im_depth = image.shape[2]
    shape = [256,256,im_depth]
    
    resized_image = np.empty(shape, dtype = float)
    
    for depth in range(0,im_depth):
        image_2d = image[:,:,depth]
        resized_2d = np.pad(image_2d,((8,8),(8,8)),'constant',constant_values=0)
        resized_image[:,:,depth] = resized_2d

    return resized_image



#returns standardized image
def standardize(image):
    mean = image.mean()
    sd = image.std()

    return (image - mean)/(sd + 1e-8)


#save all image paths to a file
def save_to_file():
    flair_files = []
    t1_files = []
    t1ce_files = []
    t2_files = []

    images_path = cfg['data_dir']

    mask_files = glob(os.path.join(images_path,'*/*/*_seg*'))

    for i in mask_files:
        flair_files.append(i.replace('_seg','_flair'))
        t1_files.append(i.replace('_seg','_t1'))
        t1ce_files.append(i.replace('_seg','_t1ce'))
        t2_files.append(i.replace('_seg','_t2'))

    

    df = pd.DataFrame(data={"flair": flair_files,"t1": t1_files,"t1ce": t1ce_files,"t2": t2_files, 'mask' : mask_files})
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    
    assert(len(shuffled_df) == 335)

    return shuffled_df



def data_frame_subset(df,start,length):
    return df[start:start+length]



