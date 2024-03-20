#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "heatUtilSeq.h"

#define min(a, b) ((a) <= (b) ? (a) : (b))
// #define KOL
#define PINN
double compute_next(Grid now, Region future, double coff_k)
{  
    int i, j, k;
    double diag_0, diag_1, diag_2, weight_0, weight_1, weight_2;

    double local_diff;

    // 3 对应 维度
    diag_0 = -2.0 + now.h0 * now.h0 / (3*coff_k*now.dt);
    diag_1 = -2.0 + now.h1 * now.h1 / (3*coff_k*now.dt);
    diag_2 = -2.0 + now.h2 * now.h2 / (3*coff_k*now.dt);

    weight_0 = coff_k*now.dt / (now.h0*now.h0);
    weight_1 = coff_k*now.dt / (now.h1*now.h1);
    weight_2 = coff_k*now.dt / (now.h2*now.h2);

    for (i = 0; i < future.Nx_0; i++)
        for (j = 0; j < future.Nx_1; j++)
            for (k = 0; k < future.Nx_2; k++)
            if (i==0||j==0||k==0||i==future.Nx_0-1||j==future.Nx_1-1||k==future.Nx_1-1)
            {
                #ifdef KOL
                now.region.mat[i][j][k] += bc(now.dt);
                #endif

                #ifdef PINN
                if (i != future.Nx_0-1) {
                    now.region.mat[i][j][k] = 10.0;
                } else {
                    now.region.mat[i][j][k] = 0;
                }
                #endif
            } else
            {
                future.mat[i][j][k] = weight_0*(now.region.mat[i-1][j][k] + now.region.mat[i+1][j][k] + now.region.mat[i][j][k]*diag_0)
                                    + weight_1*(now.region.mat[i][j-1][k] + now.region.mat[i][j+1][k] + now.region.mat[i][j][k]*diag_1)
                                    + weight_2*(now.region.mat[i][j][k-1] + now.region.mat[i][j][k+1] + now.region.mat[i][j][k]*diag_2); 
            }           

    
    double diff = 0.0;
    for (i = 1; i < future.Nx_0-1; i++)
    {
        for (j = 1; j < future.Nx_1-1; j++)
        {
            for (k = 1; k < future.Nx_2-1; k++){
                local_diff = now.region.mat[i][j][k] - future.mat[i][j][k];
                now.region.mat[i][j][k] = future.mat[i][j][k];
                diff += local_diff*local_diff;
            }
        }
    }
    return diff;
}

int main( )
{
    double coff_k = 1;
    int maxEpoch = 100000;
    double dt1 = 0.1;
    double tol = 1e-9;
    double T = 1;
    int dim = 3;
    
    Domain domain;
    domain.domain_x0_s = 0.0;
    domain.domain_x0_e = 1.0;
    
    domain.domain_x1_s = 0;
    domain.domain_x1_e = 1;

    domain.domain_x2_s = 0;
    domain.domain_x2_e = 1;

    int Nx_0 = 32;
    int Nx_1 = 32;
    int Nx_2 = 32;

    Region Now = alloc_region(Nx_0, Nx_1, Nx_2);
    Region Future = alloc_region(Nx_0, Nx_1, Nx_2);

    Grid Grid_Now;
    Grid_Now.region = Now;
    Grid_Now.h0 = (double) (domain.domain_x0_e - domain.domain_x0_s) / (double) Now.Nx_0;
    Grid_Now.h1 = (double) (domain.domain_x1_e - domain.domain_x1_s) / (double) Now.Nx_1;
    Grid_Now.h2 = (double) (domain.domain_x2_e - domain.domain_x2_s) / (double) Now.Nx_2;

    
    double dt2 = 0.125 * min(min(Grid_Now.h0, Grid_Now.h1), Grid_Now.h2) * min(min(Grid_Now.h0, Grid_Now.h1), Grid_Now.h2) / coff_k;
    Grid_Now.dt = (dt1 >= dt2) ? dt2 : dt1;
    // printf("%lf %lf %lf %lf\n",  Grid_Now.h0, Grid_Now.h1, Grid_Now.h2, Grid_Now.dt);
    

    int epoch_desired = T / Grid_Now.dt;

    init_region(&Grid_Now.region, domain);
    
    int step = 0;
    int t = 0.0;
    double diff = 0;

    int convergence = 0;
    FILE * file;

    char file_name[128];
    FILE * DIFF = fopen("difference3D.dat", "w");
    fprintf(DIFF, "step diff"); 

    while (!convergence)
    {
        step ++;
        t = t + Grid_Now.dt;
        diff = compute_next(Grid_Now, Future, coff_k);
        
        if (step % 100 == 0){
            fprintf(DIFF, "\n%d %12.11lf", step, diff);

            sprintf(file_name, "outputs/outputSeq%d.dat", step / 100);
            file=fopen(file_name, "w");
            fprintf(file, "\n");
            for (int i = 0; i < Grid_Now.region.Nx_0; i++)
            {
                for (int j = 0; j < Grid_Now.region.Nx_1; j++)
                {
                    for (int k = 0; k < Grid_Now.region.Nx_2; k++) 
                    {
                        fprintf(file,"%15.11f",Grid_Now.region.mat[i][j][k]);
                    }
                    fprintf(file, "\n");    
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }

        if ((diff < tol) || step >= epoch_desired) {
            printf("Converged\n");
            fprintf(DIFF, "\n%d %12.11lf", step, diff);
            
            sprintf(file_name, "outputSeq%1.0lf.dat", T);
            file=fopen(file_name, "w");
            fprintf(file, "\n");
            for (int i = 0; i < Grid_Now.region.Nx_0; i++)
            {
                for (int j = 0; j < Grid_Now.region.Nx_1; j++)
                {
                    for (int k = 0; k < Grid_Now.region.Nx_2; k++) {
                        fprintf(file,"%15.11f",Grid_Now.region.mat[i][j][k]);
                        }
                    fprintf(file, "\n");    
                }
                fprintf(file, "\n");
            }
            fclose(file);
            break;
        }
        if (step >= maxEpoch) {
            printf("Converge Failed\n");
            break;
        }
        
    }
    fclose(DIFF);

    printf("%lf\n", step*Grid_Now.dt);

    free_region(Grid_Now.region);
    free_region(Future);

    return 0;
}