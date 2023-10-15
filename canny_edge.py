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

