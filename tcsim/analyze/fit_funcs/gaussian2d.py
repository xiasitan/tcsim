import numpy as np


def func(xs, ys, x0=0, y0=0, sigmax=1, sigmay=1, area=1, ofs=0):
    r2 = (xs - x0) ** 2 / (2 * sigmax ** 2) + (ys - y0) ** 2 / (2 * sigmay ** 2)
    return ofs + area / (2 * sigmax * sigmay) / np.sqrt(np.pi / 2) * np.exp(-r2)


def guess(xs, ys, zs):
    zofs = np.mean([zs[0, :], zs[-1, :], zs[:, 0], zs[:, -1]])
    zs = zs - zofs
    maxidxy = np.argmax(np.abs(zs).sum(axis=1))
    maxidxx = np.argmax(np.abs(zs).sum(axis=0))

    xspan = xs[-1, 0] - xs[0, 0]
    yspan = ys[0, -1] - ys[0, 0]

    sigmax = xspan / 5
    sigmay = yspan / 5

    maxidx0 = np.argmax(np.abs(zs))
    dmin = (np.max(xs) - np.min(xs)) / 8
    mask = (
        (xs - xs.flatten()[maxidxx]) ** 2 + (ys - ys.flatten()[maxidx0]) ** 2
    ) > dmin ** 2
    area = np.sum(zs[mask])

    return dict(
        x0=xs[maxidxy, maxidxx],
        y0=ys[maxidxy, maxidxx],
        ofs=zofs,
        area=area,
        sigmax=sigmax,
        sigmay=sigmay,
    )

def twoD_Gaussian(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xy_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()