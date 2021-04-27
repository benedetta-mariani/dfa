# author: Dominik Krzeminski (dokato)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

# detrended fluctuation analysis

def calc_rms(x, scale):
    """
    windowed Root Mean Square (RMS) with linear detrending.
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (x.shape[0]//scale, scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
    return rms

def dfa(x, scale_lim=[5,9], scale_dens=0.25, show=False):
    """
    Detrended Fluctuation Analysis - measures power law scaling coefficient
    of the given signal *x*.

    More details about the algorithm you can find e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free 
    view on neuronal oscillations, (2012).

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of length 2 
        boundaries of the scale, where scale means windows among which RMS
        is calculated. Numbers from list are exponents of 2 to the power
        of X, eg. [5,9] is in fact [2**5, 2**9].
        You can think of it that if your signal is sampled with F_s = 128 Hz,
        then the lowest considered scale would be 2**5/128 = 32/128 = 0.25,
        so 250 ms.
      *scale_dens* = 0.25 : float
        density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ] 
      *show* = False
        if True it shows matplotlib log-log plot.
    Returns:
    --------
      *scales* : numpy.array
        vector of scales (x axis)
      *fluct* : numpy.array
        fluctuation function values (y axis)
      *alpha* : float
        estimation of DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc)**2))
    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    if show:
        fluctfit = 2**np.polyval(coeff,np.log2(scales))
        plt.loglog(scales, fluct, 'bo')
        plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
        plt.title('DFA')
        plt.xlabel(r'$\log_{10}$(time window)')
        plt.ylabel(r'$\log_{10}$<F(t)>')
        plt.legend()
        plt.show()
    return scales, fluct, coeff[0]

def calc_rms2(x, scale, overlap,minscale, maxscale):
    """
    Root Mean Square in windows with linear detrending.
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
      *overlap*: 
        admitted overlap between windows
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    
    """
    
    ## Signal profile
    # making an array with data divided in windows
    #shape = (x.shape[0]//scale, scale) # 50 % of overlap is not present...
    #X = np.lib.stride_tricks.as_strided(x,shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    overlap = int((1 - overlap/100)*minscale) # percentage of the scale

    rms = []
    i = 0
    while i + maxscale < len(x):
        xcut = x[i:i+ scale]
        #print(e, xcut)
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms.append(np.sqrt(np.mean((xcut-xfit)**2)))  #?
        i += overlap
        
    rms = np.array(rms)
    print(rms.shape)
    return rms

def dfa2(x, scale_lim=[5,9], scale_dens=0.25, show=False, overlap = 50):
    """
    Detrended Fluctuation Analysis - algorithm with measures power law
    scaling of the given signal *x*.
    More details about algorithm can be found e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free 
    view on neuronal oscillations, (2012).
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of lenght 2 
        boundaries of the scale where scale means windows in which RMS
        is calculated. Numbers from list are indexes of 2 to the power
        of range.
      *scale_dens* = 0.25 : float
        density of scale divisions
      *show* = False
        if True it shows matplotlib picture
    Returns:
    --------
      *scales* : numpy.array
        vector of scales
      *fluct* : numpy.array
        fluctuation function
      *alpha* : float
        DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        fluct[e] = np.mean(np.sqrt(calc_rms(y, sc, overlap, min(scales),max(scales))**2)) # inutile... 
    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    if show:
        fluctfit = 2**np.polyval(coeff,np.log2(scales))
        plt.loglog(scales, fluct, 'bo')
        plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
        plt.title('DFA')
        plt.xlabel(r'$\log_{10}$(time window)')
        plt.ylabel(r'$\log_{10}$<F(t)>')
        plt.legend()
        plt.show()
    return scales, fluct, coeff[0]


def power_law_noise(n, alpha, var=1):
    '''
    Generale power law noise. 
    
    Args:
    -----
      *n* : int
        number of data points
      *alpha* : float
        DFA exponent
      *var* = 1 : float
        variance
    Returns:
    --------
      *x* : numpy.array
        generated noisy data with exponent *alpha*

    Based on:
    N. Jeremy Kasdin, Discrete simulation of power law noise (for
    oscillator stability evaluation)
    '''
    # computing standard deviation from variance
    stdev = np.sqrt(np.abs(var))
    beta = 2*alpha-1
    hfa = np.zeros(2*n)
    hfa[0] = 1
    for i in range(1,n):
        hfa[i] = hfa[i-1] * (0.5*beta + (i-1))/i
    # sample white noise
    wfa = np.hstack((-stdev +2*stdev * np.random.rand(n), np.zeros(n)))
    fh = np.fft.fft(hfa)
    fw = np.fft.fft(wfa)
    fh = fh[1:n+1]
    fw = fw[1:n+1]
    ftot = fh * fw
    # matching the conventions of the Numerical Recipes
    ftot = np.hstack((ftot, np.zeros(n-1)))
    x = np.fft.ifft(ftot)    
    return np.real(x[:n])



if __name__=='__main__':
    n = 1000
    x = np.random.randn(n)
    # computing DFA of signal envelope
    x = np.abs(ss.hilbert(x))
    scales, fluct, alpha = dfa(x, show=1)
    print(scales)
    print(fluct)
    print("DFA exponent: {}".format(alpha))



