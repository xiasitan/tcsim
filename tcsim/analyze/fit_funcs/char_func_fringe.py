import numpy as np
def char_func_fringe(xy_tuple, amplitude, sigma_x,sigma_y, beta_amp, theta, offset):
    theta = theta+np.pi/2
    (x,y) = xy_tuple
    alpha = x+1j*y
    beta = beta_amp*np.exp(-1j*theta)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2) 
    
    g = offset + amplitude*np.exp( - (a*((x)**2) + 2*b*(x)*(y) 
                            + c*((y)**2)))*np.real(np.exp(np.conjugate(alpha)*beta-np.conjugate(beta)*alpha))
    return g.ravel()
    
    
    
    