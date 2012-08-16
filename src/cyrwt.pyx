from __future__ import division
import numpy as np
cimport numpy as np
from rwt cimport *

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t


def _prepareInputs(src_array, h, L):

    src_array = np.array(src_array, dtype=DTYPEd)
    if src_array.ndim > 1:
        m, n = src_array.shape
    else:
        m = 1
        n = src_array.size

    h = np.array(h, dtype=DTYPEd)
    if h.ndim > 1:
        h_row, h_col = h.shape
    else:
        h_row = 1
        h_col = h.size
    
    if h_col > h_row:
        lh = h_col
    else:    
        lh = h_row
        
    if L == None:
        #
        # Estimate L
        #
        i = n
        j = 0
        while i % 2 == 0:
            i >>= 1
            j += 1

        L = m
        i = 0
        while L % 2 == 0:
            L >>= 1
            i += 1
	    
        if min(m, n) == 1:
            L = max(i, j)
        else:
            L = min(i, j)
        
    assert(L != 0, "Maximum number of levels is zero; no decomposition can be performed!")
    assert(L > 0, "The number of levels, L, must be a non-negative integer")

    #
    # Check the ROW dimension of input
    #
    if m > 1:
        mtest = m / 2.0**L
        assert(mtest == int(mtest), "The matrix row dimension must be of size m*2^(L)")

    #
    # Check the COLUMN dimension of input
    #
    if n > 1:
        ntest = n / 2.0**L
        assert(ntest == int(ntest), "The matrix column dimension must be of size n*2^(L)")
        
    return src_array, L, m, n, lh


def mdwt(x, h, L=None):
    """	
    Function computes the discrete wavelet transform y for a 1D or 2D input
    signal x using the scaling filter h.

    Input:
       x : finite length 1D or 2D signal (implicitly periodized)
       h : scaling filter
       L : number of levels. In the case of a 1D signal, length(x) must be
           divisible by 2^L; in the case of a 2D signal, the row and the
           column dimension must be divisible by 2^L. If no argument is
           specified, a full DWT is returned for maximal possible L.

    Output:
       y : the wavelet transform of the signal 
           (see example to understand the coefficients)
       L : number of decomposition levels

    1D Example:
       x = makesig('LinChirp',8);
       h = daubcqf(4,'min');
       L = 2;
       [y,L] = mdwt(x,h,L)

    1D Example's  output and explanation:

       y = [1.1097 0.8767 0.8204 -0.5201 -0.0339 0.1001 0.2201 -0.1401]
       L = 2

    The coefficients in output y are arranged as follows

       y(1) and y(2) : Scaling coefficients (lowest frequency)
       y(3) and y(4) : Band pass wavelet coefficients
       y(5) to y(8)  : Finest scale wavelet coefficients (highest frequency)

    2D Example:

       load test_image        
       h = daubcqf(4,'min');
       L = 1;
       [y,L] = mdwt(test_image,h,L);

    2D Example's  output and explanation:

       The coefficients in y are arranged as follows.

              .------------------.
              |         |        |
              |    4    |   2    |
              |         |        |
              |   L,L   |   H,L  |
              |         |        |
              --------------------
              |         |        |
              |    3    |   1    |
              |         |        |
              |   L,H   |  H,H   |
              |         |        |
              `------------------'
       
       where 
            1 : High pass vertically and high pass horizontally
            2 : Low pass vertically and high pass horizontally
            3 : High pass vertically and low  pass horizontally
            4 : Low pass vertically and Low pass horizontally 
                (scaling coefficients)

    See also: midwt, mrdwt, mirdwt

    """
    
    x, L, m, n, lh = _prepareInputs(x, h, L)

    y = np.zeros_like(x)
    
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_x = x.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_h = h.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_y = y.flatten()
    
    MDWT(
        <double *>np_x.data,
        m,
        n,
        <double *>np_h.data,
        lh,
        L,
        <double *>np_y.data
        );
    
    return y, L


def midwt(y, h, L=None):
    """
    Function computes the inverse discrete wavelet transform x for a 1D or
    2D input signal y using the scaling filter h.

    Input:
	y : finite length 1D or 2D input signal (implicitly periodized)
           (see function mdwt to find the structure of y)
       h : scaling filter
       L : number of levels. In the case of a 1D signal, length(x) must be
           divisible by 2^L; in the case of a 2D signal, the row and the
           column dimension must be divisible by 2^L.  If no argument is
           specified, a full inverse DWT is returned for maximal possible
           L.

    Output:
       x : periodic reconstructed signal
       L : number of decomposition levels

    1D Example:
       xin = makesig('LinChirp',8);
       h = daubcqf(4,'min');
       L = 1;
       [y,L] = mdwt(xin,h,L);
       [x,L] = midwt(y,h,L)

    1D Example's  output:

       x = 0.0491 0.1951 0.4276 0.7071 0.9415 0.9808 0.6716 0.0000
       L = 1

    See also: mdwt, mrdwt, mirdwt
    """
    
    y, L, m, n, lh = _prepareInputs(y, h, L)
    
    x = np.zeros_like(y)
    
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_x = x.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_h = h.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_y = y.flatten()
    
    MIDWT(
        <double *>np_x.data,
        m,
        n,
        <double *>np_h.data,
        lh,
        L,
        <double *>np_y.data
        );
    
    return x, L


def mrdwt(x, h, L=None):
    """
    [yl,yh,L] = mrdwt(x,h,L);
 
    Function computes the redundant discrete wavelet transform y
    for a 1D  or 2D input signal. (Redundant means here that the
    sub-sampling after each stage is omitted.) yl contains the
    lowpass and yh the highpass components. In the case of a 2D
    signal, the ordering in yh is 
    [lh hl hh lh hl ... ] (first letter refers to row, second to
    column filtering). 

    Input:
       x : finite length 1D or 2D signal (implicitly periodized)
       h : scaling filter
       L : number of levels. In the case of a 1D 
           length(x) must be  divisible by 2^L;
           in the case of a 2D signal, the row and the
           column dimension must be divisible by 2^L.
           If no argument is
           specified, a full DWT is returned for maximal possible L.
   
    Output:
       yl : lowpass component
       yh : highpass components
       L  : number of levels

  HERE'S AN EASY WAY TO RUN THE EXAMPLES:
  Cut-and-paste the example you want to run to a new file 
  called ex.m, for example. Delete out the  at the beginning 
  of each line in ex.m (Can use search-and-replace in your editor
  to replace it with a space). Type 'ex' in matlab and hit return.


    Example 1::
    x = makesig('Leopold',8);
    h = daubcqf(4,'min');
    L = 1;
    [yl,yh,L] = mrdwt(x,h,L)
    yl =  0.8365  0.4830 0 0 0 0 -0.1294 0.2241
    yh = -0.2241 -0.1294 0 0 0 0 -0.4830 0.8365
    L = 1
    Example 2:
    load lena;
    h = daubcqf(4,'min');
    L = 2;
    [ll_lev2,yh,L] = mrdwt(lena,h,L);  lena is a 256x256 matrix
    N = 256;
    lh_lev1 = yh(:,1:N); 
    hl_lev1 = yh(:,N+1:2*N); 
    hh_lev1 = yh(:,2*N+1:3*N);
    lh_lev2 = yh(:,3*N+1:4*N); 
    hl_lev2 = yh(:,4*N+1:5*N); 
    hh_lev2 = yh(:,5*N+1:6*N);
    figure; colormap(gray); imagesc(lena); title('Original Image');
    figure; colormap(gray); imagesc(ll_lev2); title('LL Level 2');
    figure; colormap(gray); imagesc(hh_lev2); title('HH Level 2');
    figure; colormap(gray); imagesc(hl_lev2); title('HL Level 2');
    figure; colormap(gray); imagesc(lh_lev2); title('LH Level 2');
    figure; colormap(gray); imagesc(hh_lev1); title('HH Level 1');
    figure; colormap(gray); imagesc(hl_lev2); title('HL Level 1');
    figure; colormap(gray); imagesc(lh_lev2); title('LH Level 1');
           
    See also: mdwt, midwt, mirdwt

    Warning! min(size(x))/2^L should be greater than length(h)

    """

    x, L, m, n, lh = _prepareInputs(x, h, L)
    
    yl = np.zeros_like(x)
    if min(m, n) == 1:
        yh = np.zeros((m, L*n), dtype=DTYPEd)
    else:
        yh = np.zeros((m, 3*L*n), dtype=DTYPEd)

    cdef np.ndarray[DTYPEd_t, ndim=1]  np_x = x.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_h = h.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_yl = yl.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_yh = yh.flatten()
    
    MRDWT(
        <double *>np_x.data,
        m,
        n,
        <double *>np_h.data,
        lh,
        L,
        <double *>np_yl.data,
        <double *>np_yh.data
        );
    
    return yl, yh, L
    

def mirdwt(yl, yh, h, L=None):
    """
    Function computes the inverse redundant discrete wavelet
    transform x  for a 1D or 2D input signal. (Redundant means here
    that the sub-sampling after each stage of the forward transform
    has been omitted.) yl contains the lowpass and yl the highpass
    components as computed, e.g., by mrdwt. In the case of a 2D
    signal, the ordering in
    yh is [lh hl hh lh hl ... ] (first letter refers to row, second
    to column filtering).  

    Input:
       yl : lowpass component
       yh : highpass components
       h  : scaling filter
       L  : number of levels. In the case of a 1D signal, 
            length(yl) must  be divisible by 2^L;
            in the case of a 2D signal, the row and
            the column dimension must be divisible by 2^L.
   
    Output:
	     x : finite length 1D or 2D signal
	     L : number of levels

  HERE'S AN EASY WAY TO RUN THE EXAMPLES:
  Cut-and-paste the example you want to run to a new file 
  called ex.m, for example. Delete out the  at the beginning 
  of each line in ex.m (Can use search-and-replace in your editor
  to replace it with a space). Type 'ex' in matlab and hit return.


    Example 1:
    xin = makesig('Leopold',8);
    h = daubcqf(4,'min');
    L = 1;
    [yl,yh,L] = mrdwt(xin,h,L);
    [x,L] = mirdwt(yl,yh,h,L)
    x = 0.0000 1.0000 0.0000 -0.0000 0 0 0 -0.0000
    L = 1
  
    Example 2:  
    load lena;
    h = daubcqf(4,'min');
    L = 2;
    [ll_lev2,yh,L] = mrdwt(lena,h,L);  lena is a 256x256 matrix
    N = 256;
    lh_lev1 = yh(:,1:N); 
    hl_lev1 = yh(:,N+1:2*N); 
    hh_lev1 = yh(:,2*N+1:3*N);
    lh_lev2 = yh(:,3*N+1:4*N); 
    hl_lev2 = yh(:,4*N+1:5*N); 
    hh_lev2 = yh(:,5*N+1:6*N);
    figure; colormap(gray); imagesc(lena); title('Original Image');
    figure; colormap(gray); imagesc(ll_lev2); title('LL Level 2');
    figure; colormap(gray); imagesc(hh_lev2); title('HH Level 2');
    figure; colormap(gray); imagesc(hl_lev2); title('HL Level 2');
    figure; colormap(gray); imagesc(lh_lev2); title('LH Level 2');
    figure; colormap(gray); imagesc(hh_lev1); title('HH Level 1');
    figure; colormap(gray); imagesc(hl_lev2); title('HL Level 1');
    figure; colormap(gray); imagesc(lh_lev2); title('LH Level 1');
    [lena_Hat,L] = mirdwt(ll_lev2,yh,h,L);
    figure; colormap(gray); imagesc(lena_Hat); 
                            title('Reconstructed Image');

    See also: mdwt, midwt, mrdwt

    Warning! min(size(yl))/2^L should be greater than length(h)

    """

    yl, L, m, n, lh = _prepareInputs(yl, h, L)
    
    yh = np.array(yh)
    mh, nh = yh.shape
    
    #
    # check for consistency of rows and columns of yl, yh
    #
    if min(m, n) > 1:
        assert(m == mh and nh == 3*n*L, "Dimensions of first two input matrices not consistent!")
    else:
        assert(m == mh and nh == n*L, "Dimensions of first two input vectors not consistent!")

    x = np.zeros_like(yl)
    
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_x = x.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_h = h.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_yl = yl.flatten()
    cdef np.ndarray[DTYPEd_t, ndim=1]  np_yh = yh.flatten()
    
    MIRDWT(
        <double *>np_x.data,
        m,
        n,
        <double *>np_h.data,
        lh,
        L,
        <double *>np_yl.data,
        <double *>np_yh.data
        );
    
    return x, L