#include <iostream>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fdm/heat.hpp>
#include <fdm/evolve.hpp>

#if !defined(NX) || !defined(NY) || !defined(NZ)
#define NX 50+2
#define NY 50+2
#define NZ 50+2
#endif


typedef double value_type;

int main ( int argc, char ** argv )
{

  return 0;
}