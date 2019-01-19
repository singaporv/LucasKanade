import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade   #importing the lucas Kanade module
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
temp_1 = random_frame[116:152, 59:146]
plot_frame = plt.imshow(random_frame, cmap = "gray") 
eps_2 = 0.01
p_c = np.zeros(2)



for i in range(num_frames - 1):
    It = car_data[:,:,i]        
    It1 = car_data[:,:,i+1]     
    x1 = int(round(rect[0]))
    y1 = int(round(rect[1]))
    x2 = int(round(rect[2]))
    y2 = int(round(rect[3]))
    It = It[y1:y2+1,x1:x2+1]   

    p_s = LucasKanade(It, It1, rect, p_c)

    y1 = rect[0] + p_s[0]
    y2 = rect[2] + p_s[0]
    x1 = rect[1] + p_s[1]
    x2 = rect[3] + p_s[1]
    rect_temp = np.transpose([y1, x1, y2, x2])

    p_c = LucasKanade(temp_1, It1, rect_temp, p_s)

    y1 = rect[0] + p_c[0]
    y2 = rect[2] + p_c[0]
    x1 = rect[1] + p_c[1]
    x2 = rect[3] + p_c[1]
    rect = np.transpose([y1, x1, y2, x2])

    # for getting the image at particular frame
    # if (i == 0): 
    # if (i == 99): 
    # if (i == 199): 
    # if (i == 299):
    # if (i == 399):
    plot_frame.set_data(It1) 


    rect2 = patches.Rectangle((y1,x1), rect[2] - rect[0],  rect[3] - rect[1],  edgecolor = 'r', fill = False)
    coordinates.append(rect)
    ax.add_patch(rect2)  
    plt.pause(0.01) 
    rect2.remove()

coordinates = np.array(coordinates)
np.save('../result/carseqrects-wcrt.npy',coordinates)

plt.ioff()