import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here

    # Initializers
    d_p = np.zeros([6])
    p0 = np.zeros([6])
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tol = 1e-2

    # Finding the x and y for equation (10)
    # Increasing uniformly in x and y direction uniformly respectively.
    # Also converting to int type.
    h,w = It.shape
    y_temp = np.linspace(0, h, h).astype(int)
    x_temp = np.linspace(0, w, w).astype(int)
    y, x = np.meshgrid(y_temp,x_temp, sparse = False)
    y = y.flatten()
    x = x.flatten()
    # print (y.shape)
    # print (x.shape)

    # Gradient of It1 in x and y directions
    g_x = np.gradient(It1, axis = 1)
    g_y = np.gradient(It1, axis = 0)
    while True:
        print (It1.shape, M.shape)
        It1 = scipy.ndimage.affine_transform(It1,M)
        g_x = scipy.ndimage.affine_transform(g_x,M)
        g_y = scipy.ndimage.affine_transform(g_y,M)
		# Final grad values
        g_x = g_x.flatten()
        g_y = g_y.flatten()

		# For taking care of corner zeros that are incuded while warping 
        mask = np.ones([h, w])
        mask = scipy.ndimage.affine_transform(mask, M)
        It = It * mask 

		# Finding the error
        error = It-It1 
        error = error.flatten()

		# Finding the steep_descent by multiplying Jacobian and gradient
        steep_descent = np.zeros([len(x), 6])
        for q in range(len(x)):
            steep_descent[q] = [x[q]*g_x[q], y[q]*g_x[q], g_x[q], x[q]*g_y[q], y[q]*g_y[q], g_y[q]]

        # Finding Hessian and comp
        Hessian = np.matmul((steep_descent.T), steep_descent) 
        comp = np.matmul(steep_descent.T, error)

		# Hessian update and multiplication with comp
        d_p = np.matmul(np.linalg.inv(Hessian), comp)
        p0 += d_p

		# Updating M in loop
        M += [[d_p[4], d_p[3], d_p[5]],[d_p[1],d_p[0], d_p[2]]]

        if(np.linalg.norm(d_p)<tol):
            break

        return M



