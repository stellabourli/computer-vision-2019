from sys import argv
import numpy as np
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from math import pi, cos, sin
from scipy import signal
from scipy import misc

#from askhsh2.py
def sobel_edges(imageName, threshold_rate):
    #Load and show image
    image = np.array(Image.open(imageName))
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

    #Find threshold value
    maxValue = 0
    for i in range(Sobel_edges.shape[0]):
        for j in range(Sobel_edges.shape[1]):
            if Sobel_edges[i][j] > maxValue:
                maxValue = Sobel_edges[i][j]
    threshold = maxValue * threshold_rate

    #Create binary image
    resultImage = np.zeros((image.shape[0], image.shape[1]))
    for i in range(Sobel_edges.shape[0]):
        for j in range(Sobel_edges.shape[1]):
            if Sobel_edges[i][j] > threshold:
                resultImage[i][j] = 255
            else:
                resultImage[i][j] = 0
    result = Image.fromarray(resultImage.astype(np.uint8))
    return result

#Check if number of arguments are 3
if len(argv) != 2:
    print("Wrong number of arguments!")
    print("Use: image_file")
    exit()

#Load image and create binary image with edges
imageName = argv[1]
input_image = Image.open(imageName)
edges_image = sobel_edges(imageName, 0.2)
image = np.array(edges_image)
print("Shape: "+str(image.shape))
#plt.imshow(image, cmap = "gray")
#plt.show()

rmin, rmax = 10, 20 #radius limits
steps = 250

A = {} #Hough dictionary
for x in range(image.shape[0]): #for each pixel
    for y in range(image.shape[1]):
        if image[x][y] == 255: #if pixel is white
            for r in range(rmin, rmax + 1): #for each radius
                for t in range(steps): #for each point in circle
                    a = x - int(r * cos(t * 2 * pi / steps))
                    b = y - int(r * sin(t * 2 * pi / steps))
                    #voting
                    if (a,b,r) in A:
                        A[(a,b,r)] += 1
                    else:
                        A[(a,b,r)] = 1

#select circles
circles = []
for k, v in sorted(A.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps > 0.55: #check if there are enough points in circle
        not_close = False
        for xc, yc, rc in circles: #check if it is not close to other circle
            if (x - xc) ** 2 + (y - yc) ** 2 <= rc ** 2:
                not_close = True
        if not_close == False:
            circles.append((x, y, r))


print("Draw circles...")
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)
for x, y, r in circles:
    draw_result.ellipse((y-r, x-r, y+r, x+r), outline='lawngreen')
output_image.save("circles.png")
plt.imshow(output_image, cmap = "gray")
plt.show()
