import numpy as np
from tensorflow.keras.utils import Sequence
import nibabel as nib

import config
from preprocess import load_image,resize_image,standardize,save_to_file
from augment import random_flip_rotate

breaks = [3,13,23,33,43,53,63,73,83,93,103,113,123,133,143,153]

class CustomDataGen(Sequence):

    def __init__(self,
                df,
                to_fit=True,batch_size=1,dim=(256,256),
                    n_channels=1,n_classes=1,shuffle=True):
        

        self.df = df.copy()
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes=n_classes
        self.shuffle = shuffle
        #self.on_epoch_end()
        self.present_depth=0


    def __len__(self):
        return int(np.floor(len(self.df) * 15))

    def on_epoch_end():
        pass

    def __getitem__(self,index):
        
        present_index = int(index/15)
        self.present_depth = index%15

        i=present_index
        flair_list= self.df.iloc[i]['flair']
        t1_list   = self.df.iloc[i]['t1']
        t1ce_list = self.df.iloc[i]['t1ce']
        t2_list   = self.df.iloc[i]['t2']
        mask_list = self.df.iloc[i]['mask']
        
       
        X = self._generate_X(flair_list,t1_list,t1ce_list,t2_list)
        
        if self.to_fit:
            y = self._generate_y(mask_list)
            return X,y
        else:
            return X


    def _generate_X(self,flair_path,t1_path,t1ce_path,t2_path):
        
        paths = [flair_path,t1_path,t1ce_path,t2_path]
        X = np.empty((10,256,256,4))

        for modality in range(len(paths)):
            ID = paths[modality]
            image_3d = self._load_image(ID)
            
            for depth in range(image_3d.shape[2]):
                image_2d = image_3d[:,:,depth]
                X[depth,:,:,modality] = image_2d
        return X
    
    
        

    def _generate_y(self,mask_path):
        
        y = np.empty((10,256,256,1))
        

        ID = mask_path
        image_3d = self._load_image(ID)
        
        image_bool = image_3d>0
        image_binary = image_bool.astype(int)
        
        for depth in range(image_3d.shape[2]):
            image_2d = image_3d[:,:,depth]
            y[depth,:,:,0] = image_2d

        return y
    
    

    def _load_image(self,image_path):
        
        image= load_image(image_path)
        
        present_depth = self.present_depth
        depth_first = breaks[present_depth]
        depth_last = breaks[present_depth+1]-1
        new_image = image[:,:,depth_first:depth_last]
        
        new_image = standardize(new_image)
        resized_image = resize_image(new_image)
        
        return resized_image
