Edge detection is a computer vision algorithm for finding boundaries and important features in images, like object outlines, contours, and texture changes. Here we specifically implement the Canny edge detector.

This is a multi-stage algorithm that involves several steps outlined below.

1. Grayscale and Gaussian Smoothing:
    Grayscale the image and smooth it to reduce image noise 
2. Gradient Calculation:
    Using sobel operators for the x and y direction, we calculate angle and magnitude 
    of the image gradient.
3. Non-Maximum Suppression:
    Suppresses non-maximum pixels. Pixels are only kept if it is a local maximum 
    determined by the gradient magnitude and angle. Otherwise it is set to zero
4. Double Thresholding:
    Classify each pixel into strong, weak, or irreleavant edges based on a low and high threshold value
5. Edge Tracking by Hysteresis:
    Retain weak edges and transform them into strong edges if they are connected to a strong edge. Otherwise, discard and set the pixel to zero.
    
Challenges faced during implementation:

1. Parameter Tuning:
   It is difficult to choose good parameters for double thresholds, reducing noisy images, and coloring of hysteresis pixels

2. Handling Noisy Images:
   While encoding the noisy function for images, it is discovered that high levels of noise can lead to false edge detections. Proper preprocessing or denoising techniques is needed.

3. Vague directions:
   The directions given for canny edge detector outline algorithms in a vague manner. We have to figure out the specifics through testing, researching package documentations, and looking at online tutorials.

4. Performance Optimization:
   Depending on the size of the image, the computation involved in edge detection can be resource-intensive.
   Using nested for loops may not have been the most computationally efficient way to handle matrices as 
   there may be numpy functionalities that speeds up this process.