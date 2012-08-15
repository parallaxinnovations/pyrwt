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
        
    void MRDWT(
        double *x,
        int m,
        int n,
        double *h,
        int lh,
        int L,
        double *yl,
        double *yh
        )

    void MIRDWT(
        double *x,
        int m,
        int n,
        double *h,
        int lh,
        int L,
        double *yl,
        double *yh
        )

