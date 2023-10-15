import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image,ImageFilter
import math

#Show the img
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

#Non-maximum suppression 
def non_max_suppression(img,direction):
  rows,columns = img.shape
  suppressed_matrix = np.zeros((rows,columns), dtype=np.int32)

  for i in range(1,rows-1):
      for j in range(1,columns-1):
          neg_direction = 255
          pos_direction = 255

          #check direction for gradient
          if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
              neg_direction = img[i, j+1]
              pos_direction = img[i, j-1]
          elif (22.5 <= direction[i,j] < 67.5):
              neg_direction = img[i+1, j-1]
              pos_direction = img[i-1, j+1]
          elif (67.5 <= direction[i,j] < 112.5):
              neg_direction = img[i+1, j]
              pos_direction = img[i-1, j]
          elif (112.5 <= direction[i,j] < 157.5):
              neg_direction = img[i-1, j-1]
              pos_direction = img[i+1, j+1]
          
          #keep pixel if it is a local maximum 
          if (img[i,j] >= pos_direction) and (img[i,j] >= neg_direction):
              suppressed_matrix[i,j] = img[i,j]
          else:
              suppressed_matrix[i,j] = 0
  return suppressed_matrix

#double threshold
def double_threshold(suppressed_matrix,lowRatio=0.05,highRatio=0.09,weak=25,strong=255):
  highThreshold = suppressed_matrix.max() * highRatio 
  lowThreshold = highThreshold  * lowRatio

  rows,columns = suppressed_matrix.shape
  threshold_matrix = np.zeros((rows,columns),dtype=np.int32)


  strong_i, strong_j = np.where(suppressed_matrix >= highThreshold)
  zeros_i, zeros_j = np.where(suppressed_matrix < lowThreshold)
  weak_i, weak_j = np.where((suppressed_matrix <= highThreshold) & (suppressed_matrix >= lowThreshold))

  threshold_matrix[strong_i,strong_j] = strong
  threshold_matrix[weak_i,weak_j] = weak
  threshold_matrix[zeros_i,zeros_j] = 0
  return threshold_matrix

def hysteresis(threshold_matrix,weak=25,strong=255):
  rows,cols = threshold_matrix.shape
  final = np.zeros((rows,cols))
  for r in range(1,rows-1):
      for c in range(1,cols-1):
          if (threshold_matrix[r,c] == weak):
              if ((threshold_matrix[r+1, c-1] == strong) or (threshold_matrix[r+1, c] == strong) or (threshold_matrix[r+1, c+1] == strong)
                          or (threshold_matrix[r, c-1] == strong) or (threshold_matrix[r, c+1] == strong)
                          or (threshold_matrix[r-1, c-1] == strong) or (threshold_matrix[r-1, c] == strong) or (threshold_matrix[r-1, c+1] == strong)):
                  final[r, c] = strong
              else:
                  final[r, c] = 0
          elif (threshold_matrix[r,c] == strong):
              final[r,c] = strong
  return final

def canny_edge_detection(link):
  gray_scale = gray_scale_img(link)
  blurred_matrix = gaussian_blur_img(gray_scale)
  G,direction = gradient_calc(blurred_matrix)
  suppressed_matrix = non_max_suppression(G,direction)
  threshold_matrix = double_threshold(suppressed_matrix)
  final_matrix = hysteresis(threshold_matrix)

  original = mpimg.imread(link)
  fig, (ax1, ax2) = plt.subplots(1, 2)

  ax1.imshow(original)
  ax2.imshow(final_matrix,cmap='gray',vmin=0,vmax=255)
  plt.show()  
  ax1.axis('off')
  ax2.axis('off')

  return final_matrix

while True:
  image_path = input("Enter the path of the image: ")
  try: 
    canny_edge_detection(image_path)
    break 
  except Exception as e:
     print(f"Error: {e}. Please enter a valid image path.")