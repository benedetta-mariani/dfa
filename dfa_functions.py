import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)

def weight_average(arr,weights,axis):
    return np.average(arr,axis = axis, weights = weights), 1/np.sqrt(np.sum(weights, axis = axis))

def calc_rms(x,scale,overlap,minscale,maxscale,ordd):
    
    """
    Root Mean Square in windows with linear detrending.
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
      *overlap*: percentage of allowed overlap between windows
      *minscale*: minumum length of the windows considered
      *maxscale*: maximum length of the windows considered
      
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    
    """
    
    scale_ax = np.arange(scale)
    if not (overlap > 0):
        shape = (x.shape[0]//scale, scale)
        ln = (x.shape[0]//scale)*scale
        X = np.reshape(x[:ln],shape)
        rms = np.zeros(X.shape[0])
        coeff = np.polyfit(scale_ax, X.T, ordd)
        #print(coeff.shape)

        for e in range(len(rms)):
            xfit = np.polyval(coeff[:,e], scale_ax)
            # detrending and computing RMS of each window
            rms[e] = np.sqrt(np.mean((X[e]-xfit)**2))
            
    else:
        rms = []
        i = 0
        while i + scale < len(x):
            xcut = x[i:i + scale]
            coeff = np.polyfit(scale_ax, xcut, ordd) ## try also another detrending 
            xfit = np.polyval(coeff, scale_ax)
            # detrending and computing RMS of each window
            rms.append(np.sqrt(np.mean((xcut-xfit)**2)))
            i += overlap
        rms = np.array(rms)

    return rms
    
def dfa(x,scale_lim=[5,9],scale_dens=0.25,overlap = 0,ordd  = 1,xmin = 'default',xmax = 'default'):
    
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
      *overlap*: percentage of allowed overlap between windows
      
    Returns:
    --------
      *scales* : numpy.array
        vector of scales
      *fluct* : numpy.array
        fluctuation function
      *alpha* : float
        DFA exponent
    """
    # Signal profile
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(int)
    
    if xmin =='default': 
        xmin = scales[0]
    else:
        xmin = 2**xmin
    if xmax == 'default':
        xmax = scales[-1]
    else:
        xmax = 2**xmax 
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    err = np.zeros(len(scales))
    for e, sc in enumerate(scales):
        c = calc_rms(y, sc, int(overlap*sc), min(scales),max(scales), ordd)
        fluct[e] = np.mean(c)
        err[e] = np.std(c)/np.sqrt(len(c))
    
    # fitting a line to rms data
    import copy
    scales2 = scales.copy()
    fluct2 = fluct.copy()
    fluct2 = fluct2[scales2>= xmin]
    scales2 = scales2[scales2>= xmin]
    fluct2 =fluct2[scales2<= xmax]
    scales2 = scales2[scales2<= xmax]
    x = sm.add_constant(np.log2(scales2), prepend=False)
    mod = sm.OLS(np.log2(fluct2),x)
    v =mod.fit()
    coeff = [v.params[0], v.bse[0]]
    return scales, fluct, coeff, err

def plot_fluct(scales,fluct,err,show = 0,ax = None,xmin = 'default',xmax = 'default'):
    
    if xmin == 'default':
        xmin = min(scales)
    else:
        xmin = 2**xmin
    if xmax == 'default':
        xmax = max(scales)
    else:
        xmax =  2**xmax
              
    if show:
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)    
        ax.errorbar(scales, fluct,yerr = np.array(1*err), fmt = '^', color = 'royalblue', ecolor = 'royalblue')
    fluct = fluct[scales >= xmin]
    scales = scales[scales >= xmin]
    
    fluct = fluct[scales <= xmax]
    scales = scales[scales <= xmax]
 
    x = sm.add_constant(np.log2(scales), prepend=False)
    mod = sm.OLS(np.log2(fluct),x)
    v =mod.fit()
    coeff = [v.params[0], v.bse[0]]
    ordd = v.params[1]
    #print(coeff[0])
    #coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    #print(coeff[0])
    fluctfit = 2**np.polyval([coeff[0], ordd],np.log2(scales))
    
    if show:
        ax.plot(scales, fluctfit, color = 'tab:red',lw =3, label=r'$\alpha$ = %0.2f'%coeff[0])
        ax.set_xlabel(r'$\log_{10}$(time window)')
        ax.set_ylabel(r'$\log_{10}\langle F(t)\rangle$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    return coeff

def returnH(alpha):
    
    """
    returns Hurst exponent calculated from DFA exponent
    
    if alpha < 0.5 anticorrelations are present and H = alpha
    if alpha = 0.5 the signal has no memory and H = alpha
    if 0.5 < alpha < 1 positive correlations are present and H = alpha
    if alpha > 1 the process is non stationary and H = alpha - 1
    """
    if alpha < 1:
        return alpha, 0
    else:
        return alpha - 1, 1