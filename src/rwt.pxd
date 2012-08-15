cdef extern from "rwt.h":

    void MDWT(
        double *x,
        int m,
        int n,
        double *h,
        int lh,
        int L,
        double *y
        )
        
    void MIDWT(
        double *x,
        int m,
        int n,
        double *h,
        int lh,
        int L,
        double *y
        )
        
