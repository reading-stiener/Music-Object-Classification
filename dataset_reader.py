from past.builtins import xrange
import numpy as np
import scipy.misc as misc
import pandas as pa
import re
import os
import sys
from cv2 import imread, imwrite
import psutil
import random 

class class_dataset_reader:

    def __init__(self, records_list, dest_path, seed = 444, split = 0.2, min_nr = 2):
        """
        Initialize a file reader for the DeepScores classification data
        :param records_list: path to the dataset
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        """
        
        print("Initializing DeepScores Classification Batch Dataset Reader...")

        self.path = records_list
        self.dest_path = dest_path
        self.seed = seed

        self.class_names = pa.read_csv(self.path+"/class_names.csv", header=None)

        #print('class names', self.class_names)

        config = open(self.path+"/config.txt", "r")
        config_str = config.read()
        self.tile_size = re.split('\)|,|\(', config_str)[4:6]
        
        #print('tile size', self.tile_size)

        self.tile_size[0] = int(self.tile_size[0])
        self.tile_size[1] = int(self.tile_size[1])

        self.min_nr = min_nr
        self.split = split
        self.images =[] 
        self.annotations = [] 

    def read_images(self, train_test=False):
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
            for folder in os.listdir(self.path):
                if os.path.isdir(self.path +"/"+folder) and max(self.class_names[1].isin([folder])):
                        class_index = int(self.class_names[self.class_names[1] == folder][0])
                        self.load_class(folder,class_index)


            np.save(self.dest_path+"/template paths.npy", self.images) 
            np.save(self.dest_path+"/annotation.npy", self.annotations)

        else:  
            if train_test:  
                self.images = np.load(self.dest_path+"/train_images.npy") 
                self.annotations =  np.load(self.dest_path+"/train_annotations.npy")
                self.test_images = np.load(self.dest_path+"/test_images.npy") 
                self.test_annotations =  np.load(self.dest_path+"/test_annotations.npy")
                
                return None

            self.images = np.load(self.dest_path+"/template paths.npy")
            self.annotations = np.load(self.dest_path+"/annotation.npy")

        
        print("arrays were made")
        # extract test data
        test_indices = []
        train_indices = []
        
        print("splitting data: " + str(1 - self.split) + "-training " + str(self.split) + "-testing")

        # going through unique annotations
        for cla in np.unique(self.annotations):
            if sum(self.annotations == cla) < self.min_nr:
                print("Less than " + str(self.min_nr) + " occurences - removing class " + self.class_names[1][cla])
            else:
                # do split
                cla_indices = np.where(self.annotations == cla)[0]
                np.random.shuffle(cla_indices)
                train_indices.append(cla_indices[0:int(len(cla_indices) * (1 - self.split))])
                test_indices.append(cla_indices[int(len(cla_indices) * (1 - self.split)):len(cla_indices)])
    

        train_indices = np.concatenate(train_indices)
        test_indices = np.concatenate(test_indices)
        
        print("train indices", train_indices) 
        print("test indices", test_indices) 

        # setting up train and test set
        self.test_images = self.images[test_indices]
        self.test_annotations = self.annotations[test_indices]

        self.images = self.images[train_indices]
        self.annotations = self.annotations[train_indices]

        print(len(test_indices))

        # Shuffle the test data
        perm = np.arange(self.test_images.shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(perm)
        self.test_images = self.test_images[perm]
        self.test_annotations = self.test_annotations[perm]

        np.save(self.dest_path+"/train_images.npy", self.images) 
        np.save(self.dest_path+"/train_annotations.npy", self.annotations)
        np.save(self.dest_path+"/test_images.npy", self.test_images) 
        np.save(self.dest_path+"/test_annotations.npy", self.test_annotations)
  

    def load_class(self, folder, class_index):
        # move trough images in folder
        for image in os.listdir(self.path +"/"+folder):
            self.load_image(folder, image, class_index)
        return None

    def load_image(self,folder,image, class_index):
        img = imread(self.path + "/" + folder + "/" + image)

        nr_y = img.shape[0] // self.tile_size[0]
        nr_x = img.shape[1] // self.tile_size[1]
        count = 0

        for x_i in xrange(0, nr_x):
            for y_i in xrange(0, nr_y):

                dest_image_path = self.dest_path+"/"+str(class_index)+"_"+str(count)+"_"+image
                self.images.append(dest_image_path)

                print("destination path", dest_image_path)
                
                # write a grayscale image to disk
                im = img[y_i*self.tile_size[0]:(y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1]]
                print(im.shape)

                imwrite(dest_image_path, im)
                self.annotations.append(class_index)
                count += 1

        return None


    
    def test_set(self):  
        return {"test_images": self.test_images, 
                "test_annotations": self.test_annotations}

    def train_set(self):  
        return {"train_images": self.images, 
                "train_annotations": self.annotations}


if __name__ == "__main__":
    data_reader = class_dataset_reader(records_list="/home/abi-osler/Documents/CV_final_project/DeepScoresClassification",
                                        dest_path= "/home/abi-osler/Documents/CV_final_project/final_project/images_template_matching" )
    data_reader.read_images()