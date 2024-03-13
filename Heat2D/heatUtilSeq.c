
#include "heatUtilSeq.h"

Region alloc_region(int Nx_0, int Nx_1) 
{
    Region r;
    r.Nx_0 = Nx_0;
    r.Nx_1 = Nx_1;

    r.mat = malloc(Nx_0*sizeof(double *));
    r.mat[0] = malloc(Nx_0*Nx_1*sizeof(double));

    for (int i = 1; i < r.Nx_0; i++) r.mat[i] = r.mat[i-1] + Nx_1;

    return r;
}

void print_region(Region r)
{
    for (int i = 0; i < r.Nx_0; i++) {
        printf("|    ");
        for (int j = 0; j < r.Nx_1; j++)
        {
            printf("%10.6lf   ", r.mat[i][j]);
        }
        printf(" |\n");
    }
    printf("\n");
}

/** @brief Initial Condition */
double phi(double x1, double x2) {return x1*x1+x2*x2;}
double bc(double t) {return 2*t*2;}

void init_region(Region * r, Domain d)
{   
    int Nx_0 = r->Nx_0;
    int Nx_1 = r->Nx_1;

    double dx1 = (d.domain_x0_e - d.domain_x0_s) / (r->Nx_0-1);
    double dx2 = (d.domain_x1_e - d.domain_x1_s) / (r->Nx_1-1);
    
    int i, j;
    for (i = 0; i < Nx_0; i++){
        for (j = 0; j < Nx_1; j++) 
            // if (i == 0 || j == 0 || i == Nx_0-1 || j == Nx_1 - 1 ) {
            //     r->mat[i][j] = 10;
            // } else {
            //     r->mat[i][j] = 0;
            // }
            r->mat[i][j] = phi(dx1*i, dx2*j);
    }
}

void free_region(Region r)
{
    free(r.mat[0]);
    free(r.mat);
}