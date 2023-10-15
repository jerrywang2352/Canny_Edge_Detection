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
  gray_arr = np.asarray(gray_img)
  return gray_arr