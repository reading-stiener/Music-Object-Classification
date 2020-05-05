
# Python program to explain cv2.imshow() method  
  
# importing cv2  
import cv2  
  
# path  
path = '/home/abi-osler/Documents/CV_final_project/DeepScoresClassification/accidentalDoubleFlat/0.png'
  
# Reading an image in grayscale mode 
image = cv2.imread(path, 0) 
print(image.shape)
  
# Window name in which image is displayed 
window_name = 'image'
  
# Using cv2.imshow() method  
# Displaying the image  
cv2.imshow(window_name, image)  
cv2.waitKey(0)
