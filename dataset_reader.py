from past.builtins import xrange
import numpy as np
import scipy.misc as misc
import pandas as pa
import re
import os
import sys
from cv2 import imread, imwrite
import random 

class class_dataset_reader:
    """
    Class for processing dataset. Used this for both detecting using Pearson's correalation and 
    CNN model 

    Attributes
    ----------

    self.path (str): 
        Path for DeepScores-Example directory 
    
    self.dest_path (str): 
        Path for all processed images from the dataset
    
    self.file_path (str): 
        Path for list of file_images paths with its annotations for train, test, validation sets

    Methods
    --------
    
    read_images(): 
        Reads through folders for each class 
    
    load_class(folder, class_index): 
        Goes through unprocessed files for each class folder 
    
    load_image(folder, class_index): 
        Splits each image using of tile size (220,120,3)

    """


    def __init__(self, data_path, seed = 444, split = (0.7,0.2,0.1), min_nr = 2, train_test=False):
        """
        Initialize a file reader for the DeepScores classification data
        
        Parameters 
        ----------
        
        data_path (str): 
            Path to the dataset

        seed (int): 
            Value for shuffling data 
        
        split (int tuple): 
            Train, validation, test split ratio 
        
        min_nr (int): 
            Minimum number of samples for splitting data of a class 
        
        train_test (bool): 
            Flag if set True uses train, validation, test split already
            stored on disk. 
        

        """
        
        print("Initializing DeepScores Classification Batch Dataset Reader...")

        self.path = data_path
        self.dest_path = "processed_data_path"
        self.seed = seed
        self.file_path = "processed_data_files"
        self.train_test = train_test
        if not os.path.exists(self.file_path): 
            os.makedirs(self.file_path)


        self.class_names = pa.read_csv(self.path+"/class_names.csv", header=None)
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

    def read_images(self):
        """
        Reads through folders for each class. Using methods below to process files and store them in dest_path. 
        The list of each processed image path with its annotation format is stored in file_path. File formats used 
        are .npy and .csv files

        Parameters
        -----------
        None 

        Returns
        -------- 
        None 
        
        """

        # checks to see dest doesn't exist
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
            print("Preparing dataset folders...")
            for folder in os.listdir(self.path):
                if os.path.isdir(self.path +"/"+folder) and max(self.class_names[1].isin([folder])):
                        class_index = int(self.class_names[self.class_names[1] == folder][0])
                        self.load_class(folder,class_index)
                        
            np.save(self.file_path+"/data_paths.npy", self.images) 
            np.save(self.file_path+"/data_annotations.npy", self.annotations)

            print("total files", len(self.images))


        else:  
            # check to see file_path exists. Loads the prepopulated data if train test 
            # flag is True
                    
           

            if len(os.listdir(self.file_path)) and self.train_test:  
                self.train_images = np.load(self.file_path+"/train_images.npy") 
                self.train_annotations = np.load(self.file_path+"/train_annotations.npy")
                self.test_images = np.load(self.file_path+"/test_images.npy") 
                self.test_annotations = np.load(self.file_path+"/test_annotations.npy")
                self.val_images = np.load(self.file_path+"/val_images.npy") 
                self.val_annotations = np.load(self.file_path+"/val_annotations.npy")

                if os.path.exists(self.file_path+"/train_image_annotation.csv"):
                    train_img_annotation = np.vstack((self.train_images, self.train_annotations)).T
                    np.savetxt(self.file_path+"/train_image_annotation.csv", train_img_annotation, 
                            delimiter=",",
                            fmt=["%s", "%s"],
                            header='filename,annotation',
                            comments='')

                    test_img_annotation = np.vstack((self.test_images, self.test_annotations)).T
                    np.savetxt(self.file_path+"/test_image_annotation.csv", test_img_annotation, 
                            delimiter=",",
                            fmt=["%s", "%s"],
                            header='filename,annotation',
                            comments='')
                    
                    val_img_annotation = np.vstack((self.val_images, self.val_annotations)).T
                    np.savetxt(self.file_path+"/val_image_annotation.csv", val_img_annotation, 
                            delimiter=",",
                            fmt=["%s", "%s"],
                            header='filename,annotation',
                            comments='')
                    return None

        self.images = np.load(self.file_path+"/data_paths.npy")
        self.annotations = np.load(self.file_path+"/data_annotations.npy")

            
        print("arrays were made")
        # extract test data
        test_indices = []
        val_indices = [] 
        train_indices = []
        
        print("splitting data: " + str(self.split[0]) + "-training " \
            + str(self.split[1]) + "-validation "+ str(self.split[2]) + "-testing")

        # going through unique annotations 
        for cla in np.unique(self.annotations):
            if sum(self.annotations == cla) < self.min_nr:
                print("Less than " + str(self.min_nr) + " occurences - removing class " + self.class_names[1][cla])
            else:
                # do split into test train and validation sets
                cla_indices = np.where(self.annotations == cla)[0]
                np.random.shuffle(cla_indices)
                
                num_idx = len(cla_indices)

                train_indices.append(cla_indices[0:int(num_idx * self.split[0])])
                
                val_indices.append(cla_indices[int(num_idx * self.split[0]): \
                                    int(num_idx * (self.split[0]+self.split[1]))])
                
                test_indices.append(cla_indices[int(num_idx* (self.split[0]+self.split[1])): num_idx])
        
        # concatenating the numpy arrays for every piece
        train_indices = np.concatenate(train_indices)
        val_indices = np.concatenate(val_indices)
        test_indices = np.concatenate(test_indices)
        
      

        # setting up train, validation, and test set
       
        self.train_images = self.images[train_indices]
        self.train_annotations = self.annotations[train_indices]

        self.val_images = self.images[val_indices]
        self.val_annotations = self.annotations[val_indices]

        self.test_images = self.images[test_indices]
        self.test_annotations = self.annotations[test_indices]
       
        

        # Shuffle the test data
        perm = np.arange(self.test_images.shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(perm)
        self.test_images = self.test_images[perm]
        self.test_annotations = self.test_annotations[perm]

        #saving new train, test, validation split into file_path

        np.save(self.file_path+"/images.npy", self.images) 
        np.save(self.file_path+"/annotations.npy", self.annotations)

        np.save(self.file_path+"/train_images.npy", self.train_images) 
        np.save(self.file_path+"/train_annotations.npy", self.train_annotations)
        np.save(self.file_path+"/test_images.npy", self.test_images) 
        np.save(self.file_path+"/test_annotations.npy", self.test_annotations)
        np.save(self.file_path+"/val_images.npy", self.val_images) 
        np.save(self.file_path+"/val_annotations.npy", self.val_annotations)

        train_img_annotation = np.vstack((self.train_images, self.train_annotations)).T
        np.savetxt(self.file_path+"/train_image_annotation.csv", train_img_annotation, 
                           delimiter=",",
                           fmt=["%s", "%s"],
                           header='filename,annotation',
                           comments='')
        
        test_img_annotation = np.vstack((self.test_images, self.test_annotations)).T
        np.savetxt(self.file_path+"/test_image_annotation.csv", test_img_annotation, 
                           delimiter=",",
                           fmt=["%s", "%s"],
                           header='filename,annotation',
                           comments='')

        val_img_annotation = np.vstack((self.val_images, self.val_annotations)).T
        np.savetxt(self.file_path+"/val_image_annotation.csv", val_img_annotation, 
                           delimiter=",",
                           fmt=["%s", "%s"],
                           header='filename,annotation',
                           comments='')

    def load_class(self, folder, class_index):
        """
        Goes through unprocessed files for each class folder 

        Parameters
        ----------
        
        folder (str): 
            path for class folder 

        class_index (int): 
            annotation for the class files

        Returns
        --------
        
        None

        """
        for image in os.listdir(self.path +"/"+folder):
            self.load_image(folder, image, class_index)

    def load_image(self,folder,image, class_index):
        """
        Splits each image using of tile size (220,120,3)

        Parameters
        ----------

        folder (str):
            path for class folder
        
        image (str): 
            Filename for image in folder

        class_index (int): 
            Image annotation

        """


        img = imread(self.path + "/" + folder + "/" + image)
        nr_y = img.shape[0] // self.tile_size[0]
        nr_x = img.shape[1] // self.tile_size[1]
        count = 0

        # splitting images into tiles and saving them to file_path
        for x_i in xrange(0, nr_x):
            for y_i in xrange(0, nr_y):

                dest_image_path = os.path.abspath(self.dest_path+"/"+str(class_index)+"_"+str(count)+"_"+image)
                self.images.append(dest_image_path)
                
                # write the cropped image of tile size to disk
                im = img[y_i*self.tile_size[0]:(y_i+1)*self.tile_size[0], x_i*self.tile_size[1]:(x_i+1)*self.tile_size[1],:]

                imwrite(dest_image_path, im)
                self.annotations.append(class_index)
                count += 1

    
    def test_set(self): 
        """

        Creates a dictionary of test set
        """

        return {"test_images": self.test_images, 
                "test_annotations": self.test_annotations}

    def train_set(self):
        """

        Creates a dictionary of train set 
        """  
        return {"train_images": self.train_images, 
                "train_annotations": self.train_annotations}


if __name__ == "__main__":
    data_reader = class_dataset_reader(data_path="/home/abi-osler/Documents/CV_final_project/DeepScoresClassification", train_test=True)
    data_reader.read_images()