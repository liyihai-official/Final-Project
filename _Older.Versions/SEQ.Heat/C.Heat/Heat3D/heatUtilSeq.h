#include <stdlib.h>
#include <stdio.h>

typedef struct _Region 
{
    double *** mat;
    int Nx_0;
    int Nx_1;
    int Nx_2;
}   Region;

typedef struct _Domain
{
    double domain_x0_s, domain_x0_e, domain_x1_s, domain_x1_e, domain_x2_s, domain_x2_e;
} Domain;

typedef struct _Grid
{
    Region region;
    double h0, h1, h2, dt;
} Grid;

double phi(double x1, double x2, double x3);
double bc(double t);

double reference(double x1, double x2, double x3, double T, int dim);

void init_region(Region * r, Domain d);

Region alloc_region(int Nx_0, int Nx_1, int Nx_2);
void free_region(Region r);

void print_region(Region r);
