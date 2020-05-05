import numpy as np
import dataset_reader
import sys 
import argparse
from cv2 import imread, imshow, waitKey
import random

FLAGS = None

class detection_cor_coeff: 
    
    def __init__(self, train_set, test_set, random=True):
        if random:
            self.templates = self.random_templates_gen(train_set) 
        else: 
            self.templates = self.average_templates_gen(train_set)
        self.test_set = test_set


    def random_templates_gen(self, train_set):
        """
        Random template generator.  
        """
        train_images = train_set["train_images"]
        train_annotations = train_set["train_annotations"]
        templates = {}
        for cla in np.unique(train_annotations):
            cla_indices = np.where(train_annotations == cla)[0]
            template_index = random.choice(cla_indices)
            templates[cla] = imread(train_images[template_index]) 
        
        return templates

    def average_templates_gen(self, train_set):  
        """ 
        Averaged template generator.
        """ 
        train_images = train_set["train_images"]
        train_annotations = train_set["train_annotations"]
        templates = {}
        for cla in np.unique(train_annotations):
            cla_indices = np.where(train_annotations == cla)[0]
            sample_indices = np.random.choice(cla_indices, 10)
            images = []

            for train_img in train_images[sample_indices]: 
                img = imread(train_img)
                images.append(img)
            images = np.stack(images)
            
            templates[cla] = np.mean(images, axis=0, dtype=np.int16)
        
        return templates


    def template_detection(self, template, test_img):

        """ 
        Takes in a template and test image and returns the pearson correlation coefficient
        @param: template and test_img both are numpy image arrays 
        @return:  Boolean based on value of correlation   
        """   
        template_flat  = np.ravel(template)
        image_flat = np.ravel(test_img) 
        #print("template", template.shape)
        #print("image", test_img.shape)
        #print("image flat", image_flat.shape)
        imshow("template img", template)
        imshow("test img", test_img)
        corr_coef = np.corrcoef(template_flat, image_flat) 
        #print(corr_coef)
        waitKey(0)
        if corr_coef[0,1] > 0.40: 
            return True 
        return False       

    def one_to_one_template_detection(self, batch_size):
        """
        Runs template dectection on particular batch size of test set 
        @param:  a batch size 
        @return: an accuracy
        """ 

        count = 0 
    
        for i in range(batch_size): 
            annotation = self.test_set["test_annotations"][i]
            test_img = imread(self.test_set["test_images"][i])
          
            template = self.templates[annotation]

            if self.template_detection(template, test_img):  
                count += 1 
        
        print("accuracy", count/batch_size) 
    

def main():  
    data_set = dataset_reader.class_dataset_reader(FLAGS.data_dir, FLAGS.dest_dir)
    # reads images in batches to aviod running out of memory 
    data_set.read_images(train_test=True)
    test_set = data_set.test_set()
    train_set = data_set.train_set()

    simple_cor_coeff = detection_cor_coeff(train_set, test_set, random=False) 
    
    # maximum possible batch size
    max_batch_size = len(test_set["test_annotations"])
    simple_cor_coeff.one_to_one_template_detection(batch_size=10000)
    #simple_cor_coeff.average_templates_gen(train_set)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/home/abi-osler/Documents/CV_final_project/DeepScoresClassification',
                      help='Directory for storing input data')

  parser.add_argument('--dest_dir', type=str,
                      default='/home/abi-osler/Documents/CV_final_project/final_project/images_template_matching',
                      help='Directory for storing processed dataset')
  FLAGS, unparsed = parser.parse_known_args()
  main() 