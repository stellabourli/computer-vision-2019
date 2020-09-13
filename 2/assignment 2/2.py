from sys import argv
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
from PIL import Image
import numpy as np
import math


def sobel_edges(imageName, threshold_rate):
    #Load and show image
    image = np.array(Image.open(imageName))
    print("Shape: "+str(image.shape))
    #plt.imshow(image, cmap = "gray")
    #plt.show()
    grayImage = np.zeros((image.shape[0], image.shape[1]))

    #Create gray image with average of red, green, blue for each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grayImage[i][j] = int((sum(image[i][j])) / 3)

    #Filter sobel
    sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    x_edges = signal.convolve2d(grayImage, sobel_x, mode='same', boundary='fill', fillvalue=0)
    y_edges = signal.convolve2d(grayImage, sobel_y, mode='same', boundary='fill', fillvalue=0)

    Sobel_edges = []
    for i in range(x_edges.shape[0]):
        row = []
        for j in range(x_edges.shape[1]):
            row.append( abs(x_edges[i][j]) + abs(y_edges[i][j]))
        Sobel_edges.append(row)
    Sobel_edges = np.asarray(Sobel_edges)
    #plt.imshow(Sobel_edges, cmap = "gray")
    #plt.show()

    #Find threshold value

    maxValue = 0
    for i in range(Sobel_edges.shape[0]):
        for j in range(Sobel_edges.shape[1]):
            if Sobel_edges[i][j] > maxValue:
                maxValue = Sobel_edges[i][j]
    threshold = maxValue * threshold_rate
    print("Threshold: "+str(threshold))

    #Create binary image
    resultImage = np.zeros((image.shape[0], image.shape[1]))
    for i in range(Sobel_edges.shape[0]):
        for j in range(Sobel_edges.shape[1]):
            if Sobel_edges[i][j] > threshold:
                resultImage[i][j] = 255
            else:
                resultImage[i][j] = 0
    plt.imshow(resultImage, cmap = "gray")
    plt.show()

    #Save image
    result = Image.fromarray(resultImage.astype(np.uint8))
    result.save("result"+imageName[:-4]+"-threshold-"+str(threshold_rate)+".jpg")

    return result


#Check if number of arguments are 3
if len(argv) != 3:
    print("Wrong number of arguments!")
    print("Use: image_file threshold")
    exit()
#Check if threshold is between 0-1
if float(argv[2]) <0 or float(argv[2])>1:
    print("Wrong threshold!")
    print("Use a threshold between 0 and 1")
    exit()

imageName = argv[1]
threshold_rate = float(argv[2])
sobel_edges(imageName, threshold_rate)
