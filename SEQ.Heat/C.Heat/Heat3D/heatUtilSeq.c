
#include "heatUtilSeq.h"

Region alloc_region(int Nx_0, int Nx_1, int Nx_2) 
{
    Region r;
    int i;
    r.Nx_0 = Nx_0;
    r.Nx_1 = Nx_1;
    r.Nx_2 = Nx_2;

    r.mat = malloc(Nx_0*sizeof(double **));
    r.mat[0] = malloc(Nx_0*Nx_1*sizeof(double *));
    r.mat[0][0] = malloc(Nx_0*Nx_1*Nx_2*sizeof(double));

    for (i = 1; i < Nx_0; i++) r.mat[i] = r.mat[i-1] + Nx_1;
    for (i = 1; i < Nx_0*Nx_1; i++) r.mat[0][i] = r.mat[0][i-1] + Nx_2;

    return r;
}


void print_region(Region r)
{
    for (int i = 0; i < r.Nx_0; i++)
    {
        for (int j = 0; j < r.Nx_1; j++)
        {
            for (int k = 0; k < r.Nx_2; k++) {printf("%12.8lf  ", r.mat[i][j][k]);}
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

// /** @brief Initial Condition */
double phi(double x1, double x2, double x3) {return x1*x1+x2*x2+x3*x3; }
double bc(double t) {return 2*t*3;}

double reference(double x1, double x2, double x3, double T, int dim) {
    return phi(x1, x2, x3) + 2*T*dim;}

void init_region(Region * r, Domain d)
{   
    int Nx_0 = r->Nx_0;
    int Nx_1 = r->Nx_1;
    int Nx_2 = r->Nx_2;

    double dx1 = (d.domain_x0_e - d.domain_x0_s) / (r->Nx_0-1);
    double dx2 = (d.domain_x1_e - d.domain_x1_s) / (r->Nx_1-1);
    double dx3 = (d.domain_x2_e - d.domain_x2_s) / (r->Nx_2-1);
    
    int i, j, k;
    for (i = 0; i < Nx_0; i++){
        for (j = 0; j < Nx_1; j++) {
            for (k = 0; k < Nx_2; k++)
            {
                if (i == 0) {
                    r->mat[i][j][k] = 10;
                } else {
                    r->mat[i][j][k] = 0;
                }
                // r->mat[i][j][k] = phi(dx1*i, dx2*j, dx3*k);
                // if (i == 0 || j == 0 || k == 0 || i+1 == Nx_0 || j+1 == Nx_1 || k+1 == Nx_2 ){
                //     r->mat[i][j][k] = 0;
                // } else {
                //     r->mat[i][j][k] = 10;
                // }
            }
                
        }
    }
}

void free_region(Region r)
{
    free(r.mat[0][0]);
    free(r.mat[0]);
    free(r.mat);
}