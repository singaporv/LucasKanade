import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from InverseCompositionAffine import InverseCompositionAffine
from LucasKanadeAffine import LucasKanadeAffine  
from SubtractDominantMotion import SubtractDominantMotion
import scipy.ndimage

# write your script here, we recommend the above libraries for making your animation

aerial_data = np.load('../data/aerialseq.npy')    

fig, ax = plt.subplots(1)    

plt.ion()
random_frame = aerial_data[:,:,0]
plot_frame = plt.imshow(random_frame, cmap = "gray") 

for i in range(aerial_data.shape[2] -1):
	#template image
	It = aerial_data[:,:,i]        
	It1 = aerial_data[:,:,i+1] 

	# M = LucasKanadeAffine(It, It1)
	M = InverseCompositionAffine(It, It1)

	It1 = scipy.ndimage.affine_transform(It1, M)
	mask = SubtractDominantMotion(It, It1)

	for r in range(It.shape[0]):
		for s in range(It.shape[1]):
			It1[r,s] = It1[r,s] + mask[r,s]

	# for getting the image at particular frame
	if (i == 119):
		plot_frame.set_data(It1) 
		plt.pause(5)


plt.ioff()

