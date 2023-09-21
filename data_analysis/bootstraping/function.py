
import numpy  as np

import h5py
import matplotlib.pyplot as plt
import numpy.ma as ma
import os
from qutip import*
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap as lsc
import time
from matplotlib.pyplot import figure, show
from matplotlib import gridspec
# from paper_figs.plotconfig import*
import scipy.optimize as opt
import matplotlib.pyplot as plt

from scipy.interpolate import interp2d
import h5py

dx=0.085
scale = 3
normal= 1
d_avg_key = [1000.0, 10000.0, 20000.0, 35000.0, 50000.0,70000.0, 100000.0, 150000.0, 200000.0]


def load_data(d1,fname,scale,normal, thresh = 0.00018275862068965518, background=False,max_file=10):
    fps = []
    fp1 = ([d1 + f for f in os.listdir(d1) if f.endswith(fname +'.h5')])
    fp1.sort(key=lambda x: os.path.getmtime(x))
    fps = fp1
    dg = []
    de = []
    dts = []
    
    for kk in range(len(fps)):
#         fname = d + files[kk]
        df = h5py.File(fps[kk], "r")
        data = df["data"]
        data_i = data["I"][:]
        x = data["x"][:, 0, 0] * scale
        y = data["y"][0, :, 0] * scale
        dt = df.attrs['decay_time']
        thresh = thresh 
        
        raw_m0 = data_i[:, 0::3]
        raw_m1 = data_i[:, 1::3] 
        raw_m2 = data_i[:, 2::3]
         
        m0 = np.where(raw_m0 < thresh, 1, 0)
        m1 = np.where(raw_m1 < thresh, 1, 0)
        m2 = np.where(raw_m2 < thresh, 1, 0)
        
        m1_g = ma.masked_array(m1, mask=m0)
        m2_g = ma.masked_array(m2, mask=m0)

        ## only care about last two measurements
        proj_g = ma.masked_array(m2, mask=m1).mean(axis=0).reshape(len(x), len(y)) * 2 - 1
        proj_e = ma.masked_array(m2, mask=np.logical_not(m1)).mean(axis=0).reshape(len(x), len(y)) * 2 - 1

        
        #double check
        double_ps_g = ma.masked_array(m2_g, mask=m1).mean(axis=0).reshape(len(x), len(y)) * 2 - 1
        double_ps_e = ma.masked_array(m2_g, mask=np.logical_not(m1)).mean(axis=0).reshape(len(x), len(y)) * 2 - 1

        
        dg.append(double_ps_g)
        de.append(double_ps_e)
        dts.append(dt)

    dg = np.array(dg)
    de = np.array(de)
    dts = np.array(dts)
    decay_times = np.unique(dts)//1000*1000
    dd = {}
    for dt in decay_times:
        dd[str(dt)] = []

    for n, dt in enumerate(dts):
        dd[str(dt//1000*1000)].append(dg[n])
    d_avg = {}
    d_centre = {}
    for dt in decay_times:
        data = np.array(dd[str(dt)])[:max_file,:].mean(axis=0)
        data/=normal
        if background:
            n=5
#             back=np.mean((data[0:n,0:n]+data[0:n,-n:]+data[-n:,0:n]+data[-n:,-n:])/4)
            back=find_background(data)
            data-=back
            print(back)
        d_avg[str(dt)]=data
        d_centre[str(dt)] = np.array(dd[str(dt)]).mean(axis=0).flatten().max()
    return x, y, d_avg





def wigner_from_char(char,dx,scale,padding=True,pad_n=1001):
    if padding:
        char_new=np.zeros((pad_n,pad_n))
        char_new[pad_n//2-char.shape[-1]//2:pad_n//2+char.shape[-1]//2+1,pad_n//2-char.shape[-1]//2:pad_n//2+char.shape[-1]//2+1]=char
    else:
        char_new=char
    center=np.zeros(np.shape(char_new))
    center[char_new.shape[-1]//2,char_new.shape[-1]//2]=1
    f=np.fft.fftfreq(char_new.shape[-1],dx)*np.pi/scale
    f=np.fft.fftshift(f)
    wig=np.fft.fft2(char_new)/np.fft.fft2(center)
    wig=np.fft.fftshift(wig)*(dx**2/np.pi**2*scale**2)
    return f,wig    
    
def wigner_origin_from_char(char,dx,scale):
    return np.sum(char)*(dx*scale)**2/np.pi**2

def find_background(data):
    def portion_sea(thresh):
        return np.sum(data>thresh)
    half=np.size(data)/2
    up,down=1.,-1.
    for i in range(100):
        val=(up+down)/2
        if portion_sea(val)>half:
            down=val
        else:
            up=val
    return up

def plot_cats(x: list,y: list, d_avg:list):
    rows = 1
    cols = len(d_avg_key)
    fig, axes = plt.subplots(rows, cols, figsize=(30, 20))
    for j in range(cols):
        axes[j].pcolormesh(x,x, d_avg[str(d_avg_key[j])] , cmap="seismic", shading = 'auto', vmax=1, vmin=-1)
        axes[j].set_aspect("equal")
        axes[j].set_title(str(d_avg_key[j]))
        
def calc_wigners(x,y,d_avg,padding=False, pad_n=1001): #1001
    f={}
    w_fft ={}
    for dt in d_avg_key:
        f,w_fft[str(dt)] = wigner_from_char(d_avg[str(dt)], dx, scale,padding,pad_n)
        w_fft[str(dt)]=w_fft[str(dt)].real# abandon the imag part
    return f,w_fft

def calc_negativity(wigners):
    negativity=[]
    for dt in d_avg_key:
        negativity.append(wigners[str(dt)].flatten().min())
    return negativity

def calc_negativity_around_origin(wigners):
    negativity=[]
    for dt in d_avg_key:
        wig=wigners[str(dt)]
        N=np.shape(wig)[0]
        print(N)
        print(wig[N//2,N//2])
        neighbors=[wig[N//2-1,N//2-1],wig[N//2-1,N//2+1],wig[N//2+1,N//2-1],wig[N//2+1,N//2+1]]
        print(neighbors)
        negativity.append(np.mean(neighbors))
    return negativity

def calc_negativity_origin(d_avg):
    negativity=[]
    for dt in d_avg_key:
        negativity.append(wigner_origin_from_char(d_avg[str(dt)], dx,scale))
    return negativity

def calc_purity(char,dx,scale):
    return np.sum(np.abs(char)**2*(dx*scale)**2)/np.pi
    
def calc_puritys(d_avg):
    purity=[]
    for dt in d_avg_key:
        purity.append(calc_purity(d_avg[str(dt)], dx,scale))
    return purity

def interplation(f, w_fft: dict):
    w_fft_interp2d = {}
    for dt in d_avg_key:
        w_fft_interp2d[str(dt)] = []
        
    for n, dt in enumerate(d_avg_key):
        x_inter = np.linspace(np.min(f),np.max(f), 2001)
        y_inter = np.linspace(np.min(f),np.max(f), 2001)
        func = interp2d(f, f, w_fft[str(dt)], kind='cubic')
        w_fft_interp2d[str(dt)] = func(x_inter, y_inter)
    return w_fft_interp2d

def plot_wigner_dynamic_with_threshold_array(d, fname, scale, normal, threshold_array, padding = False):
    fig, ax = plt.subplots(figsize=(9, 6))
    # test = []
    for n, threshod in enumerate( threshold_array):
        x, y, d_avg_array=load_data(d,fname,scale,normal, thresh=threshod)
        print(padding)
        f, w_fft_array=calc_wigners(x,y,d_avg_array, padding)
        w_fft_array_interp2d = interplation(f, w_fft_array)
        negativity_w_fft_array_interp2d = calc_negativity(w_fft_array_interp2d)
        ax.plot(d_avg_key, negativity_w_fft_array_interp2d, "*-", label = f'FFT_interp2d_{n}' )
        print(negativity_w_fft_array_interp2d)
    plt.title('Sweeping threshold,' + fname)
    plt.ylim(-2/np.pi, 0.2)
    plt.xlabel('Decay time')
    plt.ylabel('Minimun of Wigner function')
    plt.legend(fontsize = 8)
    plt.show()
    return 

 
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


def gauss_3(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3):
    """ Fitting Function"""
    return amp1 * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2))) + \
           amp2 * (np.exp((-1.0 / 2.0) * (((x - cen2) / sigma2) ** 2))) + \
           amp3 * (np.exp((-1.0 / 2.0) * (((x - cen3) / sigma3) ** 2)))

def gauss_1(x, amp1, cen1, sigma1):
    return amp1 * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2)))
    

def cut_indexes(y_inter,angle):
    k = np.tan(angle)
    max_y_index = np.max(y_inter) # define maximal y_index allowed
    max_x_index = max_y_index
    xvec_fine = np.linspace(0, max_x_index, len(y_inter))
    for i, x in enumerate(xvec_fine):
        if np.abs(x*k)>max_y_index:
            max_x_index = xvec_fine[i-1]
            break
    x_indicies = np.linspace(-max_x_index, max_x_index, len(y_inter))
    y_indicies = x_indicies*k
    return x_indicies, y_indicies

def get_data_1D_cut_w_fft(f, w_fft, angles, chose_fig = [], interp_point = 2001):
    cat_cut_array = []
    fig, axs = plt.subplots(2, len(d_avg_key), figsize = (30,6))

    for i in range(len(d_avg_key)):

        angle = angles[i]
        if i in chose_fig:
            angle = angles[i] - np.pi/2

        func = interp2d(f, f, w_fft[str(d_avg_key[i])], kind='cubic')
        
        # # define finer x,y arrays with same end values as data x,y
        x_interp2d = np.linspace(np.min(f),np.max(f), interp_point)
        y_interp2d = np.linspace(np.min(f),np.max(f), interp_point)
        
        # # get x and y indicies for line cut and load the line cut in cut
        x_index, y_index = cut_indexes(y_interp2d, -angle)
    
        cut = []
        for j in range(len(x_index)):
            cut.append(float(func(x_index[j],y_index[j])))
        cat_cut_array.append(cut)

        axs[0][i].set_aspect("equal")
        # print( len(x_interp2d) ) 
        # print( shape(func(x_interp2d, y_interp2d)) )
        axs[0][i].pcolormesh(x_interp2d, y_interp2d, func(x_interp2d, x_interp2d), vmin=-1, vmax=1 , cmap="seismic", shading='auto')
        axs[0][i].scatter(x_index, y_index, c='k', marker = '.')
        axs[0][i].set_title(str(d_avg_key[i]))
        axs[1][i].plot(x_interp2d,cut)
        
        plt.tight_layout()
    return  x_interp2d, cat_cut_array 