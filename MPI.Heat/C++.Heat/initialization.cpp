#pragma once 
#include "array.cpp"

/* ---------------------- Initialize Conditions Functions ---------------------- */
template <typename T>
void init_conditions( final_project::Array<T>& init, 
                      final_project::Array<T>& init_other, 
                      final_project::Array<T>& bias)
{
  auto NX = init.get_num_rows() - 1;
  auto NY = init.get_num_cols() - 1;

  std::size_t i, j;
  for (i = 0; i <= NX; ++i)
  {
    for (j = 0; j <= NY; ++j)
    {
      bias(i, j) = 0;
      if (i == 0)
      {
        init(i, j) = 10;
        init_other(i, j) = 10;
      }

      if (i == NX)
      {
        init(i, j) = 10;
        init_other(i, j) = 10;
      }

      if (j == 0) 
      {
        init(i, j) = 10;
        init_other(i, j) = 10;
      }

      if (j == NY)
      {
        init(i, j) = 0;
        init_other(i, j) = 0;
      }
    } 
  }
}

template <typename T>
void twodinit_basic_Heat( final_project::Array<T>& init, 
                          final_project::Array<T>& init_other, 
                          final_project::Array<T>& bias, 
                          const int s[2], const int e[2])
{
  int i, j;
  int nx = init.get_num_rows() - 2;
  int ny = init.get_num_cols() - 2;

  for (i = s[0]-1; i <= e[0]+1; ++i) 
    for (j = s[1]-1; j <= e[1]+1; ++j)
    {
      init(i, j) = 0;
      init_other(i, j) = 0;
      bias(i, j) = 0;
    }

  /* Left Side */
  if (s[0] == 1) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(0,       j) = 10;
      init_other(0, j) = 10;
    }
  }

  /* Right Side */
  if (e[0] == nx) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(nx+1,        j) = 10;
      init_other(nx+1,  j) = 10;
    }
  }

  /* Bottom side */
  if (s[1] == 1) {
    for (i = s[0]; i <= e[0]; ++i) {
      init(i,       0) = 10;
      init_other(i, 0) = 10;
    }
  }
  
  /* UP side */
  if (e[1] == ny) {
    for (i = s[0]; i <= e[0]; ++i) {
      double xx = (double) i / (nx+1);
      init(i,       ny+1) = 0;
      init_other(i, ny+1) = 0;
    }
  }
}

/* Possion */
template <typename T>
void twodinit_basic_Possion(final_project::Array<T>& init, 
                            final_project::Array<T>& init_other, 
                            final_project::Array<T>& bias, 
                            const int s[2], const int e[2])
{
  int i, j;
  int nx = init.get_num_rows() - 2;
  int ny = init.get_num_cols() - 2;

  for (i = s[0]-1; i <= e[0]+1; ++i) 
    for (j = s[1]-1; j <= e[1]+1; ++j)
    {
      init(i, j) = 0;
      init_other(i, j) = 0;
      bias(i, j) = 0;
    }

  /* Left Side */
  if (s[0] == 1) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(0,       j) = yy / (1+yy*yy);
      init_other(0, j) = yy / (1+yy*yy);
    }
  }

  /* Right Side */
  if (e[0] == nx) {
    for (j = s[1]; j < e[1]+1; ++j) {
      double yy = (double) j / (ny+1);
      init(nx+1,        j) = yy / (4+yy*yy);
      init_other(nx+1,  j) = yy / (4+yy*yy);
    }
  }

  /* Bottom side */
  if (s[1] == 1) {
    for (i = s[0]; i <= e[0]; ++i) {
      init(i,       0) = 0;
      init_other(i, 0) = 0;
    }
  }
  
  /* UP side */
  if (e[1] == ny) {
    for (i = s[0]; i <= e[0]; ++i) {
      double xx = (double) i / (nx+1);
      init(i,       ny+1) = 1 / ((xx+1)*(xx+1) + 1);
      init_other(i, ny+1) = 1 / ((xx+1)*(xx+1) + 1);
    }
  }
}

template <typename T>
void twodinit_basic_Heat(final_project::Array<T>& init)
{
  
  int i, j;
  double xx, yy;

  int nx = init.get_num_rows() - 2;
  int ny = init.get_num_cols() - 2;


  for (i = 1; i <= nx; ++i)
    for (j = 1; j <= ny; ++j)
    {
      init(i, j) = 0;
    }

  for (j = 0; j <= ny; ++j)
  {
    yy = (double) j / (ny+1);
    init(0, j) = 10;
  }


  for (j = 0; j <= ny; ++j)
  {
    yy = (double) j / (ny+1);
    init(nx+1,        j) = 10;
  }


  for (i = 0; i <= nx; ++i) {
    init(i,       0) = 10;
  }


  for (i = 0; i <= nx+1; ++i) 
  {
    xx = (double) i / (nx+1);
    init(i,       ny+1) = 0;
  }
}


template <typename T>
void twodinit_basic_Heat(final_project::Array_Distribute<T>& init, 
                         final_project::Array_Distribute<T>& init_other, 
                         final_project::Array_Distribute<T> bias)
{
  
  int i, j;
  double xx, yy;

  int nx = init.get_glob_num_rows() - 2;
  int ny = init.get_glob_num_cols() - 2;

  int nx_loc = init.get_num_rows() - 2;
  int ny_loc = init.get_num_cols() - 2;

  int s[] {init.get_start(0), init.get_start(1)};
  int e[] {init.get_end(0),   init.get_end(1)  };

  for (i = 1; i <= nx_loc; ++i)
    for (j = 1; j <= ny_loc; ++j)
    {
      init(i, j) = 0;
      init_other(i, j) = 0;
      bias(i, j) = 0;
    }

  if (s[0] == 1)
    for (j = 1; j <= ny_loc; ++j)
    {
      yy = (double) j / (ny+1);
      init(0, j) = 10;
      init_other(0, j) = 10;
    }


  if (e[0] == nx)
    for (j = 0; j <= ny_loc; ++j)
    {
      yy = (double) j / (ny+1);
      init(nx_loc+1,        j) = 10;
      init_other(nx_loc+1,  j) = 10;
    }


  if (s[1] == 1)
    for (i = 1; i <= nx_loc; ++i) {
      init(i,       0) = 10;
      init_other(i, 0) = 10;
    }


  if (e[1] == ny)
    for (i = 1; i <= nx_loc; ++i) {
       xx = (double) i / (nx+1);
      init(i,       ny_loc+1) = 0;
      init_other(i, ny_loc+1) = 0;
    }
}
