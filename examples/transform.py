from __future__ import division
import numpy as np
from rwt import mdwt, wt
from rwt.utilities import softThreshold, hardThreshold


def main1():
    import matplotlib.pyplot as plt
    import scipy.misc as spm
    from rwt.wavelets import daubcqf
    
    lena = spm.lena()
   
    h = daubcqf(0)[0]

    trans_lena1 = wt(lena, h, axis=0)[0]
    trans_lena1 = wt(trans_lena1, h, axis=1)[0]
    trans_lena2 = mdwt(spm.lena(), h)[0]
    
    trans_lena1 -= trans_lena1.min() - 1
    trans_lena2 -= trans_lena2.min() - 1
    
    temp = np.sort(trans_lena1.ravel())
    threshold = temp[int((len(temp)-1)*(100-5)/100)]

    trans_lena1 = hardThreshold(trans_lena1, threshold)
    trans_lena2 = hardThreshold(trans_lena2, threshold)
    
    trans_lena1 = np.log(trans_lena1+1)
    trans_lena2 = np.log(trans_lena2+1)
    
    trans_lena1 /= trans_lena1.max()/255
    trans_lena2 /= trans_lena2.max()/255
    
    plt.figure()
    plt.gray()
    plt.subplot(131)
    plt.imshow(lena)
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(trans_lena1.astype(np.uint8))
    plt.title('Transformed Image')
    plt.subplot(133)
    plt.imshow(trans_lena2.astype(np.uint8))
    plt.title('Transformed Image')
    plt.show()


def main2():
    
    from rwt.wavelets import daubcqf
    
    h = daubcqf(0)[0]
    x = np.arange(16).reshape((4, 4))
    xT = x.T.copy()
    
    print x[:, 0]
    print xT[0, :]
    
    print wt(x, h, axis=0)[0]
    print wt(xT, h, axis=1)[0]
    
    print mdwt(x[:, 0], h)[0]
    print mdwt(xT[0, :], h)[0]
    
    
if __name__ == '__main__':
    main1()