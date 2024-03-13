#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "heatUtilSeq.h"

#define min(a, b) ((a) <= (b) ? (a) : (b))

double compute_next(Grid now, Region future, double coff_k)
{  
    int i, j;
    double diag_0, diag_1, weight_0, weight_1;

    double local_diff;


    diag_0 = -2.0 + now.h0 * now.h0 / (2*coff_k*now.dt);
    diag_1 = -2.0 + now.h1 * now.h1 / (2*coff_k*now.dt);
    weight_0 = coff_k*now.dt / (now.h0*now.h0);
    weight_1 = coff_k*now.dt / (now.h1*now.h1);

    for (i = 0; i < future.Nx_0; i++)
    {   
        for (j = 0; j < future.Nx_1; j++)
        {
            if (i==0||j==0||i==future.Nx_0-1||j==future.Nx_1-1)
            {now.region.mat[i][j] += bc(now.dt);}
            else 
            {future.mat[i][j] = weight_0*(now.region.mat[i-1][j] + now.region.mat[i+1][j] + now.region.mat[i][j]*diag_0)
                              + weight_1*(now.region.mat[i][j-1] + now.region.mat[i][j+1] + now.region.mat[i][j]*diag_1);}
        }
    }
    
    double diff = 0.0;
    for ( i = 1; i < future.Nx_0-1; i++)
    {
        for (j = 1; j < future.Nx_1-1; j++)
        {
            local_diff = now.region.mat[i][j] - future.mat[i][j];
            now.region.mat[i][j] = future.mat[i][j];
            diff += local_diff*local_diff;
        }
    }

    return diff;
}


int main( )
{
    double coff_k = 1;
    int maxEpoch = 100000;
    double dt1 = 0.1;
    double tol = 1e-10;

    Domain domain;
    domain.domain_x0_s = 0;
    domain.domain_x0_e = 1;
    
    domain.domain_x1_s = 0;
    domain.domain_x1_e = 1.2;

    int Nx_0 = 30;
    int Nx_1 = 30;
    Region Now = alloc_region(Nx_0, Nx_1);
    Region Future = alloc_region(Nx_0, Nx_1);

    int size_0 = Now.Nx_0 - 2;
    int size_1 = Now.Nx_1 - 2;

    Grid Grid_Now;
    Grid_Now.region = Now;
    Grid_Now.h0 = (double) (domain.domain_x0_e - domain.domain_x0_s) / (double) Now.Nx_0;
    Grid_Now.h1 = (double) (domain.domain_x1_e - domain.domain_x1_s) / (double) Now.Nx_1;

    double dt2 = 0.25 * min(Grid_Now.h0, Grid_Now.h1) * min(Grid_Now.h0, Grid_Now.h1) / coff_k;
    Grid_Now.dt = (dt1 >= dt2) ? dt2 : dt1;
    maxEpoch = 1 / Grid_Now.dt + 1; 

    init_region(&Grid_Now.region, domain);


    
    int step = 0;
    double t = 0.0;
    double diff = 0;

    int convergence = 0;
    FILE * file;

    char file_name[20];
    while (!convergence)
    {
        step ++;
        t = t + Grid_Now.dt;

        // printf("?\n");
        diff = compute_next(Grid_Now, Future, coff_k);

        // printf("?\n");
        if (step % 100 == 0){
                
            printf("%d\t%lf\n", step, t);
            sprintf(file_name, "outputSeq%d.dat", step/100);
            file=fopen(file_name, "w");
            fprintf(file, "\n");
            for (int i = 0; i < Grid_Now.region.Nx_0; i++)
            {
                for (int j = 0; j < Grid_Now.region.Nx_1; j++)
                {
                    fprintf(file,"%15.11f",Grid_Now.region.mat[i][j]);
                }
                fprintf(file, "\n");
            }
        }

        if ((diff < tol) || (step >= maxEpoch)) break;

        fclose(file);
    }

    free_region(Grid_Now.region);
    free_region(Future);

    return 0;
}