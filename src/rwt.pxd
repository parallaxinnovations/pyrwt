cdef extern from "mdwt_r.c":

    void MDWT(
        double *x,
        int m,
        int n,
        double *h,
        int lh,
        int L,
        double *y
        )
