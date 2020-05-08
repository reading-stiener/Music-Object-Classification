import numpy as np
import dataset_reader
import sys 
import argparse
from cv2 import imread, imshow, waitKey
import random

FLAGS = None

class detection_cor_coeff: 
    
    def __init__(self, train_set, test_set, random):
        """
        Class definition. 

        Parameters
        ----------
         
        train_set (str):
            Dictionary of file paths and annotations 
         
        test_set (str):
            Dictionary of file paths and annotations 

        
        random (boolean):
            flag to either define templates at random or 
            average them out 


        """
        if random:
            self.templates = self.random_templates_gen(train_set) 
        else: 
            self.templates = self.average_templates_gen(train_set)
        self.test_set = test_set


    def random_templates_gen(self, train_set):
        """
        Random template generator.  

        Parameters:
        -----------
        train_set (dict): 
            Dictionary 
        
        Returns: 
        --------
        A templates dictionary with a random template each class 
        
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

        @param: train_set dictionary 
        @return: A templates dictionary with an averaged out template of random 
                 sample of size 10 for each class. 
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


    def template_detection_threshold(self, template, test_img):

        """ 
        Takes in a template and test image of same annotation and 
        returns the pearson correlation coefficient
        
        Parameters:
        -----------
        
        template (dict): 
            Numpy image arrays
        
        test_img (dict): 
            Numpy image arrays 
        
        Returns:
        ---------
        
        Boolean based on value of correlation. Returns 
        True is correlation is above a threshold   
        """   
        template_flat  = np.ravel(template)
        image_flat = np.ravel(test_img) 
      
        #imshow("template img", template)
        #imshow("test img", test_img)
        corr_coef = np.corrcoef(template_flat, image_flat) 
        #print(corr_coef)
      
        if corr_coef[0,1] > 0.40: 
            return True 
        return False       

    def one_to_one_template_detection(self, batch_size):
        """
        Runs one to one template dectection on particular 
        batch size of test set 
        
        Parameters:
        -----------
        
        batch_size (int): A batch size 


        Return:
        -------
        
        An accuracy
        """ 

        count = 0 
    
        for i in range(batch_size): 
            annotation = self.test_set["test_annotations"][i]
            test_img = imread(self.test_set["test_images"][i])
          
            template = self.templates[annotation]

            if self.template_detection_threshold(template, test_img):  
                count += 1 
        
        return count/batch_size 
    
    def template_detection_best_match(self, batch_size):
        """
        Runs one to one template dectection on particular 
        batch size of test set 
        
        Parameters:
        -----------
        
        batch_size (int): A batch size 


        Return:
        -------
        
        An accuracy
        """ 
        
        count = 0 
        for i in range(batch_size): 
            test_annotation = self.test_set["test_annotations"][i]
            test_img = imread(self.test_set["test_images"][i])
            pred_annotation = -1
            max_corr = -1
            for annotation, template in self.templates.items():
                template_flat  = np.ravel(template)
                image_flat = np.ravel(test_img)
                corr_coef = np.corrcoef(template_flat, image_flat)
                if corr_coef[0,1] > max_corr:
                    pred_annotation = annotation
                    max_corr = corr_coef[0,1]
            if pred_annotation == test_annotation:  
                count += 1 
        return count/batch_size



def main():  
    data_set = dataset_reader.class_dataset_reader(FLAGS.data_dir, train_test=True)
    
    # reads images in batches to aviod running out of memory 
    data_set.read_images()

    test_set = data_set.test_set()
    train_set = data_set.train_set()
    simple_cor_coeff = detection_cor_coeff(train_set, test_set, random=FLAGS.randomized) 
    
    # maximum possible batch size
    max_batch_size = len(test_set["test_annotations"])
    print("Maximum batch size. ", max_batch_size)

    if FLAGS.test_mode == "one_to_one": 
        accuracy = simple_cor_coeff.one_to_one_template_detection(batch_size=FLAGS.batch_size)
        print("Accuracy of one to one method is {0} for batch size of {1}".format(accuracy, FLAGS.batch_size))
        
    elif FLAGS.test_mode == "best_match": 
        accuracy = simple_cor_coeff.template_detection_best_match(batch_size=FLAGS.batch_size)
        print("Accuracy of best match method is {0} for batch size of {1}.".format(accuracy, FLAGS.batch_size))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", 
                      type=str,
                      default="/home/abi-osler/Documents/CV_final_project/DeepScoresClassification",
                      help="Directory for storing input data")


  parser.add_argument("--batch_size", 
                      type=int,
                      default=100, 
                      help= "Set the batch size of test data to evaluate on")
                      

  
  parser.add_argument("--test_mode", 
                       type=str, 
                       default="best_match", 
                       choices=["best_match", "one_to_one"], 
                       help="Pick the mode for this test")

  parser.add_argument('--randomized', 
                      default=False, 
                      type=lambda x: (str(x).lower() == 'true'),
                      help="Setting randomized to True sets the template at random. \
                      Setting it true sets a template from a random batch of 10 from train set")

  FLAGS, unparsed = parser.parse_known_args()
  main() 