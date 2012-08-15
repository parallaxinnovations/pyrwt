from __future__ import division
import numpy as np
cimport numpy as np
from rwt cimport *

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t


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
    
    x = np.array(x, dtype=DTYPEd)
    if x.ndim > 1:
        m, n = x.shape
    else:
        m = 1
        n = x.size

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
    
    return (y, L)


