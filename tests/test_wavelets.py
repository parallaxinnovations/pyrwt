import numpy as np
import matplotlib.pyplot as plt
from rwt import dwt, idwt, rdwt, irdwt
from rwt.wavelets import waveletCoeffs, waveletlist
from numpy.testing import assert_allclose


def test_dwt():
    A = np.random.randn(64, 64)
    
    for wavelet in waveletlist():
        c0, c1, r0, r1 = waveletCoeffs(wavelet)
        
        A_coef, L = dwt(A, c0, c1)
        A_recon, L = idwt(A_coef, r0, r1)
        
        try:
            assert_allclose(
                A,
                A_recon,
                atol=1e-06,
                err_msg='Failed dwt/idwt for wavelet type %s' % wavelet
            )
        except (Exception, e):
            if wavelet not in ('dmey',):
                raise


def test_rdwt():
    A = np.random.randn(64, 64)
    
    for wavelet in waveletlist():
        c0, c1, r0, r1 = waveletCoeffs(wavelet)
        A_coefl, A_coefh, L = rdwt(A, c0, c1)
        A_recon, L = irdwt(A_coefl, A_coefh, r0, r1)
        
        try:
            assert_allclose(
                A,
                A_recon,
                atol=1e-06,
                err_msg='Failed rdwt/irdwt for wavelet type %s' % wavelet
            )
        except (Exception, e):
            if wavelet not in ('dmey',):
                raise


if __name__ == '__main__':
    import nose
    nose.runmodule()
    
