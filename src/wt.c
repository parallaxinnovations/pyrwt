#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "rwt.h"


void WT(
    double *x,
    int n,
    int prod_h,
    int stride,
    double *h,
    int lh,
    int L,
    double *y
    )
{
    double *h0;
    double *h1;
    double *ydummyl;
    double *ydummyh;
    double *xdummy;
    int i;
    int j;
    int actual_L;
    int actual_n;
    int c_o_a;
    int ir;
    int lhm1;
    int ind;
    int base_ind;
    
    xdummy = (double *)calloc(n+lh-1, sizeof(double));
    ydummyl = (double *)calloc(n, sizeof(double));
    ydummyh = (double *)calloc(n, sizeof(double));

    //
    // Low and high filters.
    //
    h0 = (double *)calloc(lh, sizeof(double));
    h1 = (double *)calloc(lh, sizeof(double));
    for (i=0; i<lh; i++){
        h0[i] = h[lh-i-1];
        h1[i] = h[i];
    }   
    for (i=0; i<lh; i+=2)
        h1[i] = -h1[i];

    lhm1 = lh - 1;
    actual_n = 2*n;

    //
    // Loop on all scales
    //
    for (actual_L=1; actual_L <= L; actual_L++){
        base_ind = 0;
        actual_n = actual_n/2;
        c_o_a = actual_n/2;

        //
        // Loop over all space
        //
        for (ir=0; ir<prod_h; ir++){
            //
            // store in dummy variable
            //
            for (j=0; j<stride; j++)
            {
                ind = base_ind+j;
                for (i=0; i<actual_n; i++)
                {
                    if (actual_L==1)  
                        xdummy[i] = x[ind];  
                    else 
                        xdummy[i] = y[ind];

                    ind += stride;
                }
                
                //
                // perform filtering lowpass and highpass
                //
                wtconv(xdummy, actual_n, h0, h1, lhm1, ydummyl, ydummyh); 
                
                //
                // restore dummy variables in matrices
                //
                ind = base_ind+j;
                for (i=0; i<c_o_a; i++){    
                    y[ind] = ydummyl[i];
                    y[ind+c_o_a*stride] = ydummyh[i];
                    ind += stride;
                } 
            }
            base_ind += stride * n;
        }  
    }

    free((void *)xdummy);
    free((void *)ydummyl);
    free((void *)ydummyh);
    free((void *)h0);
    free((void *)h1);
}


void wtconv(
    double *x_in,
    int lx, 
    double *h0,
    double *h1,
    int lhm1, 
    double *x_outl,
    double *x_outh
    )
{
    int i;
    int j;
    int ind;
    double x0;
    double x1;

    //
    // Replicate x_in on top.
    //
    for (i=lx; i < lx+lhm1; i++)
        x_in[i] = *(x_in+(i-lx));

    //
    // 
    //
    ind = 0;
    for (i=0; i<(lx); i+=2){
        x0 = 0;
        x1 = 0;
        for (j=0; j<=lhm1; j++){
            x0 += x_in[i+j]*h0[lhm1-j];
            x1 += x_in[i+j]*h1[lhm1-j];
        }
        x_outl[ind] = x0;
        x_outh[ind++] = x1;
    }
}
