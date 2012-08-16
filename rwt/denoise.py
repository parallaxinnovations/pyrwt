from __future__ import division
import numpy as np
from rwt import mdwt, midwt, mrdwt, mirdwt
from rwt.utilities import softThreshold, hardThreshold


def denoise(
    x,
    h,
    wavelet_type=0,
    low_pass=False,
    c=None,
    var_estimator=0,
    threshold_type=None,
    L=0,
    threshold=0
    ):
    """
    DENOISE is a generic program for wavelet based denoising.
    The program will denoise the signal x using the 2-band wavelet
    system described by the filter h using either the traditional 
    discrete wavelet transform (DWT) or the linear shift invariant 
    discrete wavelet transform (also known as the undecimated DWT
    (UDWT)). 

    Input:  
       x         : 1D or 2D signal to be denoised
       h         : Scaling filter to be applied
       type      : Type of transform (Default: type = 0)
                   0 --> Discrete wavelet transform (DWT)
                   1 --> Undecimated DWT (UDWT)
       option    : Default settings is marked with '*':
                   *type = 0 --> option = [0 3.0 0 0 0 0]
                   type = 1 --> option = [0 3.6 0 1 0 0]
       option(1) : Whether to threshold low-pass part
                   0 --> Don't threshold low pass component 
                   1 --> Threshold low pass component
       option(2) : Threshold multiplier, c. The threshold is
                   computed as: 
                     thld = c*MAD(noise_estimate)). 
                   The default values are:
                     c = 3.0 for the DWT based denoising
                     c = 3.6 for the UDWT based denoising
       option(3) : Type of variance estimator
                   0 --> MAD (mean absolute deviation)
                   1 --> STD (classical numerical std estimate)
       option(4) : Type of thresholding
                   0 --> Soft thresholding
                   1 --> Hard thresholding
       option(5) : Number of levels, L, in wavelet decomposition. By
                   setting this to the default value '0' a maximal
                   decomposition is used.
       option(6) : Actual threshold to use (setting this to
                   anything but 0 will mean that option(3)
                   is ignored)

    Output: 
       xd     : Estimate of noise free signal 
       xn     : The estimated noise signal (x-xd)
       option : A vector of actual parameters used by the
                program. The vector is configured the same way as
                the input option vector with one added element
                option(7) = type.

    HERE'S AN EASY WAY TO RUN THE EXAMPLES:
    Cut-and-paste the example you want to run to a new file 
    called ex.m, for example. Delete out the  at the beginning 
    of each line in ex.m (Can use search-and-replace in your editor
    to replace it with a space). Type 'ex' in matlab and hit return.

    Example 1: 
       h = daubcqf(6); [s,N] = makesig('Doppler'); n = randn(1,N);
       x = s + n/10;      (approximately 10dB SNR)
       figure;plot(x);hold on;plot(s,'r');

       Denoise x with the default method based on the DWT
       [xd,xn,opt1] = denoise(x,h);
       figure;plot(xd);hold on;plot(s,'r');

       Denoise x using the undecimated (LSI) wavelet transform
       [yd,yn,opt2] = denoise(x,h,1);
       figure;plot(yd);hold on;plot(s,'r');

    Example 2: (on an image)  
      h = daubcqf(6);  load lena; 
      noisyLena = lena + 25 * randn(size(lena));
      figure; colormap(gray); imagesc(lena); title('Original Image');
       figure; colormap(gray); imagesc(noisyLena); title('Noisy Image'); 
       Denoise lena with the default method based on the DWT
      [denoisedLena,xn,opt1] = denoise(noisyLena,h);
      figure; colormap(gray); imagesc(denoisedLena); title('denoised Image');


    See also: mdwt, midwt, mrdwt, mirdwt, SoftTh, HardTh, setopt

    """

    assert wavelet_type in (0, 1), "Unknown denoising method. If it is any good we need to have a serious talk :-)"

    if c == None:
        if type == 0:
            c = 3.0
        else:
            c = 3.6

    if threshold_type == None:
        if type == 0:
            threshold_type = 0
        else:
            threshold_type = 1

    assert threshold_type in (0, 1), 'Unknown threshold rule. Use either Soft (0) or Hard (1)'

    if x.ndim == 1:
        n = nx = x.size
        mx = 1
    else:
        mx, nx = x.shape
        n = min(mx, nx)

    if L == 0:
        L = np.floor(np.log2(n))

    if wavelet_type == 0:
        #
        # Denoising by DWT
        #
        xd, L = mdwt(x, h, L)
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
                x[ix, jx] = ykeep

        xd, L = midwt(xd, h, L)
    else:
        #
        # Denoising by UDWT
        #
        xl, xh, L = mrdwt(x, h, L)
        
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
        
        xd, L = mirdwt(xl, xh, h, L)

    xn = x - xd

    return xd, xn


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import scipy.misc as spm
    from rwt import daubcqf
    
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