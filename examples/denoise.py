from __future__ import division
import numpy as np
from rwt import dwt, idwt, rdwt, irdwt
from rwt.utilities import softThreshold, hardThreshold


def denoise(
    x,
    h,
    wavelet_type=0,
    low_pass=False,
    c=None,
    var_estimator=0,
    threshold_type=0,
    L=None,
    threshold=0
    ):
    """
    Generic program for wavelet based denoising.
    The program will denoise the signal x using the 2-band wavelet
    system described by the filter h using either the traditional 
    discrete wavelet transform (DWT) or the linear shift invariant 
    discrete wavelet transform (also known as the undecimated DWT
    (UDWT)). 

    Parameters
    ----------
    x : array-like, shape = [n] or [m, n]
        Finite length 1D or 2D signal (implicitly periodized)
    h : array-like, shape = [n]
        Scaling filter
    wavelet_type : [0, 1], optional (default=0)
        Type of transform:
        0 - Discrete wavelet transform (DWT)
        1 - Undecimated DWT (UDWT)
    low_pass : bool, optional (default=False)
        Whether to threshold low-pass part
        False - Don't threshold low pass component 
        True - Threshold low pass component
    c : float, optional (default=3.0 for DWT, 3.6 for UDWT)
        Threshold multiplier, c. The threshold is
        computed as: thld = c*MAD(noise_estimate)).
    var_estimator : [0, 1], optional (default=0)
        Type of variance estimator
        0 - MAD (mean absolute deviation)
        1 - STD (classical numerical std estimate)
    threshold_type : [0, 1], optional (default=0)
        Type of thresholding
        0 - Soft thresholding
        1 - Hard thresholding
    L : integer, optional (default=None)
        Number of levels. In the case of a 1D signal, length(x) must be
        divisible by 2**L; in the case of a 2D signal, the row and the
        column dimension must be divisible by 2**L. If no argument is
        specified, a full DWT is returned for maximal possible L.
    threshold : float, optional (default=0)
       Actual threshold to use (setting this to anything but 0 will
       mean that var_estimator)

    Returns
    -------
    xd : array-like, shape = [n] or [m, n]
        Estimate of noise free signal 
    xn : array-like, shape = [n] or [m, n]
        The estimated noise signal (x-xd)
       
    """

    assert wavelet_type in (0, 1), "Unknown denoising method. If it is any good we need to have a serious talk :-)"

    if c == None:
        if wavelet_type == 0:
            c = 3.0
        else:
            c = 3.6

    assert threshold_type in (0, 1), 'Unknown threshold rule. Use either Soft (0) or Hard (1)'

    if x.ndim == 1:
        n = nx = x.size
        mx = 1
    else:
        mx, nx = x.shape
        n = min(mx, nx)

    if L == None:
        L = np.floor(np.log2(n))

    if wavelet_type == 0:
        #
        # Denoising by DWT
        #
        xd, L = dwt(x, h, L)
        if threshold == 0:
            assert var_estimator in (0, 1), 'Unknown threshold estimator, Use either MAD (0) or STD (1)'

            tmp = xd[int(mx/2):mx, int(nx/2):nx]
            if var_estimator == 0:
                threshold = c * np.median(np.abs(tmp))/.67
            else:
                threshold = c * np.std(tmp)

        if x.ndim == 1:
            ix = np.arange(int(n/(2**L)))
            ykeep = xd[ix]
        else:
            ix = np.arange(int(mx/(2**L)))
            jx = np.arange(int(nx/(2**L)))
            ykeep = xd[ix, jx]

        if threshold_type == 0:
            xd = softThreshold(xd, threshold)
        else:
            xd = hardThreshold(xd, threshold)

        if not low_pass:
            if x.ndim == 1:
                xd[ix] = ykeep
            else:
                xd[ix, jx] = ykeep

        xd, L = idwt(xd, h, L)
    else:
        #
        # Denoising by UDWT
        #
        xl, xh, L = rdwt(x, h, L)
        
        if x.ndim == 1:
            c_offset = 1
        else:
            c_offset = 2*nx + 1

        if threshold == 0:
            assert var_estimator in (0, 1), 'Unknown threshold estimator, Use either MAD (0) or STD (1)'

            tmp = xh[:, c_offset:c_offset+nx-1]
            if var_estimator == 0:
                threshold = c * np.median(np.abs(tmp))/.67
            else:
                threshold = c * np.std(tmp)

        if threshold_type == 0:
            xh = softThreshold(xh, threshold)
            
            if low_pass:
                xl = softThreshold(xl, threshold)
        else:
            xh = hardThreshold(xh, threshold)

            if low_pass:
                xl = hardThreshold(xl, threshold)
        
        xd, L = irdwt(xl, xh, h, L)

    xn = x - xd

    return xd, xn


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import scipy.misc as spm
    from rwt.wavelets import daubcqf
    
    lena = spm.lena()
    noisy_lena = lena + 25 * np.random.randn(*lena.shape)
    
    h = daubcqf(6)[0]
    denoised_lena, xn = denoise(noisy_lena, h)
    
    plt.figure()
    plt.gray()
    plt.subplot(131)
    plt.imshow(lena)
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(noisy_lena)
    plt.title('Noisy Image')
    plt.subplot(133)
    plt.imshow(denoised_lena)
    plt.title('Denoised Image')
    plt.show()