import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade   #importing the lucas Kanade module
from LucasKanadeBasis import LucasKanadeBasis   #importing the lucas Kanade Basis module
import scipy
import cv2 as cv
# write your script here, we recommend the above libraries for making your animation


sylv_data = np.load('../data/sylvseq.npy')  
bases1 = np.load('../data/sylvbases.npy') 
a,b,c = bases1.shape

# bases --> (47, 55, 10)
bases = bases1.reshape(a*b, c)

H,W, num_frames = sylv_data.shape
# H = 240, W = 320, num_frames = 451

x = np.shape([num_frames-400, 4])

rect_1 = np.transpose([101,61,155,107])
rect_2 = np.transpose([101,61,155,107])


p = np.zeros(2)
p0 = np.zeros(2)
p02 = np.zeros(2)

fig, ax = plt.subplots(1)
coordinates = []
coordinates.append(rect_1)    

plt.ion()
random_frame = sylv_data[:,:,0]
plot_frame = plt.imshow(random_frame, cmap = "gray") 


for i in range(num_frames - 1):
	# template frame
    It = sylv_data[:,:,i]        
    # Current frame
    It1 = sylv_data[:,:,i+1]  

    p01 = np.zeros(2)
    p02 = np.zeros(2)

    x1 = int(round(rect_1[0]))
    y1 = int(round(rect_1[1]))
    x2 = int(round(rect_1[2]))
    y2 = int(round(rect_1[3]))
    It = It[y1:y2+1,x1:x2+1]

    # For LKB
    p01 = LucasKanadeBasis(It, It1, rect_1, bases, p0)

    y1 = rect_1[0] + p01[0] 
    y2 = rect_1[2] + p01[0]
    x1 = rect_1[1] + p01[1]
    x2 = rect_1[3] + p01[1]
    rect_1 = np.transpose([y1, x1, y2, x2])

    rect_12 = patches.Rectangle((y1,x1), rect_1[2] - rect_1[0],  rect_1[3] - rect_1[1], linewidth = 3, edgecolor = 'g', fill = False)
    coordinates.append(rect_1)



    # for LK
 
    x12 = int(round(rect_2[0]))
    y12 = int(round(rect_2[1]))
    x22 = int(round(rect_2[2]))
    y22 = int(round(rect_2[3]))


    p02 = LucasKanade(It, It1, rect_2, p0)

    y12 = rect_2[0] + p02[0] 
    y22 = rect_2[2] + p02[0]
    x12 = rect_2[1] + p02[1]
    x22 = rect_2[3] + p02[1]
    rect_2 = np.transpose([y12, x12, y22, x22])


    rect_22 = patches.Rectangle((y12,x12), rect_2[2] - rect_2[0],  rect_2[3] - rect_2[1], linewidth = 1,  edgecolor = 'y', fill = False)


    #for getting the image at particular frame
    # if (i == 0):
    # if (i == 199):
    # if (i == 299):
    # if (i == 349):
    if (i == 399):   	
	    plot_frame.set_data(It1) 
	    ax.add_patch(rect_12)  
	    ax.add_patch(rect_22)  

	    plt.pause(5) 

	    rect_12.remove()
	    rect_22.remove()


coordinates = np.array(coordinates)
np.save('../result/sylvseqrects.npy',coordinates)
# a = np.load('../result/sylvseqrects.npy')
# print (a.shape)


plt.ioff()