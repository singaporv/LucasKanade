import numpy as np
from scipy.ndimage import morphology
import matplotlib.pyplot as plt 


def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    mask = 0

    diff = image1 - image2
    diff = np.absolute(diff)
    a,b = diff.shape


    for i in range(a):
    	for j in range(b):
    		if (diff[i,j] <= 0.15):
    			diff[i,j] = 0
    		else:
    			diff[i,j] = 1


    # Try different combinations of dilation and erosion to get the proper image
    diff = morphology.binary_dilation(diff)
    # diff = morphology.binary_erosion(diff)
    mask = morphology.binary_dilation(diff)

    return mask
