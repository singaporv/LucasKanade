import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
    
    
def LucasKanadeBasis(It, It1, rect, bases, p0 = np.zeros(2)):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the car
    #   (top left, bot right coordinates)
    #   p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #   p: movement vector [dp_x, dp_y]
    
    # Put your implementation here
    tol = 1e-1
    p = 0
    W = np.zeros([bases.shape[0], bases.shape[1]])
    bs = np.zeros([bases.shape[0], bases.shape[1]]).flatten()
    while True: 




        x1 = int(round(rect[0]))
        y1 = int(round(rect[1]))
        x2 = int(round(rect[2]))
        y2 = int(round(rect[3]))

        error = It-It1[y1:y2+1,x1:x2+1]
        error = error.flatten()


        for z in range(bases.shape[1]):
            W[:,z] = np.sum(bases[:,z]*error)


        bs = np.sum(bases*W, axis = 1)
        error += bs

        # Gradient of in x and y directions
        grad_x = np.gradient(It1, axis = 1)
        grad_y = np.gradient(It1, axis = 0)

        # flattening gradient images
        grad_x = grad_x[y1:y2+1,x1:x2+1].flatten()
        grad_y = grad_y[y1:y2+1,x1:x2+1].flatten()

        # Convert in array form
        grad = np.vstack((grad_x, grad_y)).T

        # Jacobian identity matrix
        Jacob = np.array([[1, 0],[0, 1]])

        # Find steep descent
        steep_descent = np.matmul(grad, Jacob)


        #Find Hessian
        Hessian = np.matmul(np.transpose(steep_descent), steep_descent) 
        comp = np.matmul(steep_descent.T, error)

        #Hessian update and multiplication with com
        d_p = np.matmul(np.linalg.inv(Hessian), comp)
        It1 = scipy.ndimage.shift(It1, (-d_p[1], -d_p[0]))  

        # Updating p in loop
        p = p + d_p
        if(np.linalg.norm(d_p)<tol):
            break
            
    p0 = p0 + p
    return (p0)