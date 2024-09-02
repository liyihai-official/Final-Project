#include <stdlib.h>
#include <stdio.h>

typedef struct _Region 
{
    double ** mat;
    int Nx_0;
    int Nx_1;
}   Region;

typedef struct _Domain
{
    double domain_x0_s, domain_x0_e, domain_x1_s, domain_x1_e;
} Domain;

typedef struct _Grid
{
    Region region;
    double h0, h1, dt;
} Grid;

double phi(double x1, double x2);
double bc(double t);

void init_region(Region * r, Domain d);

Region alloc_region(int Nx_0, int Nx_1);
void free_region(Region r);

void print_region(Region r);
