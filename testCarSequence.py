import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade   #importing the lucas Kanade module
from LucasKanadeBasis import LucasKanadeBasis   #importing the lucas Kanade Basis module
import scipy
import cv2 as cv
# write your script here, we recommend the above libraries for making your animation


car_data = np.load('../data/carseq.npy')    

H,W, num_frames = car_data.shape
rect = np.transpose([59,116,145,151])

p = np.zeros(2)

fig, ax = plt.subplots(1)
coordinates = []
coordinates.append(rect)    

plt.ion()
random_frame = car_data[:,:,0]
plot_frame = plt.imshow(random_frame, cmap = "gray") 


for i in range(num_frames - 1):
    It = car_data[:,:,i]        
    It1 = car_data[:,:,i+1]     
    p0 = np.zeros(2)
    x1 = int(round(rect[0]))
    y1 = int(round(rect[1]))
    x2 = int(round(rect[2]))
    y2 = int(round(rect[3]))
    It = It[y1:y2+1,x1:x2+1]   
    p0 = LucasKanade(It, It1, rect, p0)

    y1 = rect[0] + p0[0] 
    y2 = rect[2] + p0[0]
    x1 = rect[1] + p0[1]
    x2 = rect[3] + p0[1]
    rect = np.transpose([y1, x1, y2, x2])

# : for getting the image at particular frame
    # if (i == 125):
    plot_frame.set_data(It1) 


    rect2 = patches.Rectangle((y1,x1), rect[2] - rect[0],  rect[3] - rect[1],  edgecolor = 'y', fill = False)
    coordinates.append(rect)

    ax.add_patch(rect2)  
    plt.pause(0.01) 
    rect2.remove()

coordinates = np.array(coordinates)
np.save('../result/carseqrects.npy',coordinates)



plt.ioff()