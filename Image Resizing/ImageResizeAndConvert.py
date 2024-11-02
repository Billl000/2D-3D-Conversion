import cv2 as cv2 #Image Detection 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os #Interact with the computer files


def detectAndCropWalls(file):
    #Read Image, convert to grayscale, apply thresholding
    img = cv2.imread(file)
    grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(grayscaleImg, 100, 255, cv2.THRESH_BINARY_INV)
    
    #Find edges through Canny Edge
    edges = cv2.Canny(threshold, 50, 150)
    
    
    #Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Find largest contours
    largestContours = max(contours, key=cv2.contourArea)
    
    imgWithContours = img.copy()
    cv2.drawContours(imgWithContours, [largestContours], -1, (0, 255, 0), 3)
    
    #Get the bounding box
    x, y, w, h = cv2.boundingRect(largestContours)
    
    #Crop the inside of the box
    croppped_img = img[y : y + h, x : x + w]
    
    #Display original image and the cropped result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imgWithContours, cv2.COLOR_BGR2RGB))
    plt.title("Detected wall")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(croppped_img, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Interior")
    plt.show()
    return croppped_img
    
    
    

def resizeImage(file):
    #Read image
    img = cv2.imread(file)
    
    #Printing image size
    print("Image width is ", img.shape[1])
    print("Image height is ", img.shape[0])
    
    resized = cv2.resize(img, (600, 600))
    
    img_75 = cv2.resize(resized, None, fx=.75, fy=0.75)
    
    return img_75



#Create the image directory
#img = Image.open(r"C:\Users\animu\OneDrive\Documents\VSCode Files\2D-3D Model Conversion\Image Resizing\cat.jpg")
img2 = r"C:\Users\animu\OneDrive\Documents\VSCode Files\2D-3D Model Conversion\Image Resizing\1Room1Bed.jpg"

newImg = detectAndCropWalls(img2)   

plt.imshow(newImg)
plt.waitforbuttonpress()

# file_name, file_format = os.path.splitext(os.path.basename(img2))

# #Create the directory
# directory = r"C:\Users\animu\OneDrive\Documents\VSCode Files\2D-3D Model Conversion\Image Resizing"

# #Change the directory
# os.chdir(directory)

# #Create new file, save the image to the file witht the directory above
# resized = resizeImage(img2)
# cv2.imwrite(file_name + "_resized.jpg", resized)

# #Show the image
# # cv2.imshow(file_name + "_resized.jpg", resized)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
