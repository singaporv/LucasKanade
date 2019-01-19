import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage

def InverseCompositionAffine(It, It1):
	# Input: 
	# 	It: template image
	# 	It1: Current image

	# Output:
	# 	M: the Affine warp matrix [2x3 numpy array]

 #    put your implementation here

 #    Initializers
    d_p = np.zeros([6])

    p0 = np.zeros([6])
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0,0.0,1.0]])
    M_d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  [0.0,0.0,1.0]])

    tol = 5e-2

    # Finding the x and y for equation (10)
    # Increasing uniformly in x and y direction uniformly respectively.
    # Also converting to int type.
    h,w = It.shape
    y_temp = np.linspace(0, h, h).astype(int)
    x_temp = np.linspace(0, w, w).astype(int)
    y, x = np.meshgrid(y_temp,x_temp, sparse = False, indexing = 'ij')
    y = y.flatten()
    x = x.flatten()
    # Mask with ones
    mask= np.ones([It.shape[0],It.shape[1]])

    # Gradient of It1 in x and y directions and then flatten them:
    g_x = np.gradient(It, axis = 1)
    g_y = np.gradient(It, axis = 0)
    g_x = g_x.flatten()
    g_y = g_y.flatten()

    # Calculating the below steps outside the loop (Changed)
    steep_descent = np.zeros([len(x), 6])
    for q in range(len(x)):
        steep_descent[q] = [x[q]*g_x[q], y[q]*g_x[q], g_x[q], x[q]*g_y[q], y[q]*g_y[q], g_y[q]]

    Hessian = np.matmul((steep_descent.T), steep_descent)     


    while True:
    	# Warping the image and mask
        It1_w = ndimage.affine_transform(It1, M)
        w_mask = ndimage.affine_transform(mask, M)

        #Corrected It - For corners
        It_c = It*w_mask

		# Finding the error
        error = It1_w-It_c 
        error = error.flatten()

        # Finding comp
        comp = np.matmul((steep_descent.T), error)

		# Hessian update and multiplication with comp
        d_p = np.matmul(np.linalg.inv(Hessian), comp)

		# Updating M in loop
       	M_d= np.array([[1+d_p[4], d_p[3], d_p[5]],[d_p[1],1+d_p[0], d_p[2]],[0.0,0.0,1.0]])
       	M = np.matmul(M,np.linalg.inv(M_d))

        if(np.linalg.norm(d_p)<tol):
            break

    return M[:2,:]