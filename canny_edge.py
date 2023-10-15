import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image,ImageFilter
import math


def show_img(link):
  img = mpimg.imread(link)
  plt.imshow(img)
  plt.show()

#Grayscale Conversion: I = 0.299 * R + 0.587 * G + 0.114 * B
def gray_scale_img(link):
  gray_img = Image.open(link).convert('L')
  return gray_img

#Gaussian Blurring
def gaussian_blur_img(gray_img,r=1.5):
  blurred_img = gray_img.filter(ImageFilter.GaussianBlur(radius=r))
  blurred_arr = np.asarray(blurred_img)
  return blurred_arr

#Gradient magnitude and direction calculation 
def gradient_calc(blurred_arr):
  Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  [rows, columns] = np.shape(blurred_arr) 
  sobel_filtered_image = np.zeros(shape=(rows, columns)) 
  sobel_filtered_direction = np.zeros(shape=(rows, columns)) 

  for i in range(rows - 2):
      for j in range(columns - 2):
          gx = np.sum(np.multiply(Gx, blurred_arr[i:i + 3, j:j + 3]))  # x direction
          gy = np.sum(np.multiply(Gy, blurred_arr[i:i + 3, j:j + 3]))  # y direction
          sobel_filtered_direction[i+1,j+1] = math.degrees(math.atan2(gy,gx))
          sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
  return sobel_filtered_image, sobel_filtered_direction