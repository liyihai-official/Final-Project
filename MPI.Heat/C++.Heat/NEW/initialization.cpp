/**
 * @file initialization.cpp
 * @brief This file contains functions to initialize the boundary conditions
 *        for a 2D heat equation simulation. 
 *        
 * 
 * The functions provided in this file are designed to set up the initial
 * conditions for both regular and distributed 2D arrays used in parallel
 * processing environments.
 * 
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */
#include "multi_array/array.cpp"

/////////////////////////////////////////////////////////////////////////
//      Heat 1D
/////////////////////////////////////////////////////////////////////////

template <class T>
void init_conditions_heat1d(final_project::array1d<T> & init)
{
  std::size_t i, N {init.N - 2};
  double xx { 1.0 / (N+1)};

  init.fill(0);

  init(0) = 10;  /* Left */
  init(N+1) = 0; /* Right */
}

template <class T>
void init_conditions_heat1d(final_project::array1d_distribute<T>& ping, 
                            final_project::array1d_distribute<T>& pong)
{
  std::size_t i;
  std::size_t N {ping.glob_N - 2}, N_loc {ping.size() - 2};

  double xx { 1.0 / (N+1) };

  ping.fill(0);
  pong.fill(0);
  
  /* Left */
  if (ping.starts[0] == 1)
  {
    ping(0) = 10;
    pong(0) = 10;
  }
  
  /* Right */
  if (ping.ends[0] == N)
  {
    ping(N_loc+1) = 0;
    pong(N_loc+1) = 0;
  }

}

/////////////////////////////////////////////////////////////////////////
//      Heat 2D
/////////////////////////////////////////////////////////////////////////

/**
 * @brief Initialize the boundary and initial conditions for the Heat Equation in 2D.
 * 
 * @tparam T The type of the elements in te array.
 * @param init Reference to a 2D array where the initial conditions will be set.
 */
template <class T>
void init_conditions_heat2d(final_project::array2d<T> & init)
{

  std::size_t i, j, nx {init.Rows - 2}, ny {init.Cols - 2};
  double xx { 1.0 / (nx+1) }, yy { 1.0 / (ny+1) };

  // Initial Conditions
  init.fill(0);  

  // Boundary Conditions
  for (j = 0; j <= ny; ++j)
  {
    // yy = (double) j / (ny+1);
    init(0, j) = 10;
  }


  for (j = 0; j <= ny; ++j)
  {
    // yy = (double) j / (ny+1);
    init(nx+1, j) = 10;
  }


  for (i = 0; i <= nx; ++i)
    init(i, 0) = 10;


  for (i = 0; i <= nx+1; ++i) 
  {
    // xx = (double) i / (nx+1);
    init(i,       ny+1) = 0;
  }
}


/**
 * @brief Initialize the boundary conditions for the Heat Equation 2D for distributed 
 *        arrays. (ping-pong method)
 * 
 * @tparam T The type of the elements in the array.
 * @param ping Reference to the first distributed 2D array.
 * @param pong Reference to the second distributed 2D array.
 */
template <class T>
void init_conditions_heat2d(final_project::array2d_distribute<T>& ping, 
                            final_project::array2d_distribute<T>& pong)
{
  std::size_t i, j;
  std::size_t nx   {ping.glob_Rows - 2}, ny {ping.glob_Cols - 2}, 
              nx_loc {ping.rows() - 2}, ny_loc {ping.cols() - 2};

  double xx { 1.0 / (nx+1) }, yy { 1.0 / (ny+1) };

  ping.fill(0);
  pong.fill(0);
  
  /* Up */
  if (ping.starts[0] == 1)
    for (j = 1; j <= ny_loc; ++j)
    {
      ping(0, j) = 10;
      pong(0, j) = 10;
    }
  /* Left */
  if (ping.starts[1] == 1)
    for (i = 0; i <= nx_loc; ++i) {
      ping(i, 0) = 10;
      pong(i, 0) = 10;
    }

  /* Down */
  if (ping.ends[0] == nx)
    for (j = 0; j <= ny_loc; ++j)
    {
      ping(nx_loc+1, j) = 10;
      pong(nx_loc+1, j) = 10;
    }

  /* Right */
  if (ping.ends[1] == ny)
    for (i = 0; i <= nx_loc; ++i) {
      ping(i, ny_loc+1) = 0;
      pong(i, ny_loc+1) = 0;
    }
}

/////////////////////////////////////////////////////////////////////////
//      Heat 3D
/////////////////////////////////////////////////////////////////////////

/**
 * @brief Initialize the boundary and initial conditions for the Heat Equation in 3D.
 * 
 * @tparam T The type of the elements in te array.
 * @param init Reference to a 3D array where the initial conditions will be set.
 */
template <class T>
void init_conditions_heat3d(final_project::array3d<T> & init)
{
  std::size_t i, j, k; // axis-0, 1, 2; x, y, z;
  std::size_t nx {init.Rows - 2}, ny {init.Cols - 2}, nz {init.Height - 2};
  
  double xx { 1.0 / (nx+1) }, yy { 1.0 / (ny+1) }, zz { 1.0 / (nz+1)};

  // Initial Conditions
  init.fill(0);

  // Boundary Conditions
  // z = 0;
  for (i = 1; i <= nx; ++i) 
  {
    for (j = 1; j <= ny; ++j)  
    {
      init(i, j, 0) = 10;
    }
  }

  // z = max_z; 
  for (i = 1; i <= nx; ++i) 
  {
    for (j = 1; j <= ny; ++j)  
    {
      init(i, j, nz+1) = 10;
    }
  }

  // y = 0;
  for (i = 1; i <= nx; ++i) 
  {
    for (k = 1; k <= nz; ++k)  
    {
      init(i, 0, k) = 10;
    }
  }

  // y = max_y;
  for (i = 1; i <= nx; ++i) 
  {
    for (k = 1; k <= nz; ++k)  
    {
      init(i, ny+1, k) = 10;
    }
  }

  // x = 0;
  for (j = 1; j <= ny; ++j) 
  {
    for (k = 1; k <= nz; ++k)  
    {
      init(0, j, k) = 10;
    }
  }

  // x = max_x;
  for (j = 1; j <= ny; ++j) 
  {
    for (k = 1; k <= nz; ++k)  
    {
      init(nx+1, j, k) = 0;
    }
  }
  
  // End of initialization
}

/**
 * @brief Initialize the boundary conditions for the Heat Equation 3D for distributed 
 *        arrays. (ping-pong method)
 * 
 * @tparam T The type of the elements in the array.
 * @param ping Reference to the first distributed 3D array.
 * @param pong Reference to the second distributed 3D array.
 */
template <class T>
void init_conditions_heat3d(final_project::array3d_distribute<T>& ping, 
                            final_project::array3d_distribute<T>& pong)
{ 
  std::size_t i, j, k;
  std::size_t nx  {ping.glob_Rows - 2}, ny  {ping.glob_Cols - 2}, nz {ping.glob_Heights - 2}, 
              nx_loc {ping.rows() - 2}, ny_loc {ping.cols() - 2}, nz_loc {ping.height() - 2};

  double xx { 1.0 / (nx+1) }, yy { 1.0 / (ny+1) }, zz { 1.0 / (nz+1)};

  ping.fill(0);
  pong.fill(0);

  /* Back */
  if (ping.starts[0] == 1) 
    for (j = 1; j <= ny_loc; ++j) {
      for (k = 1; k <= nz_loc; ++k) {
        ping(0, j, k) = 10;
        pong(0, j, k) = 10;
      }
    }

  /* UP */
  if (ping.starts[1] == 1) 
    for (i = 1; i <= nx_loc; ++i) {
      for (k = 1; k <= nz_loc; ++k) {
        ping(i, 0, k) = 8;
        pong(i, 0, k) = 8;
      }
    }
  
  /* Left */
  if (ping.starts[2] == 1) 
    for (i = 1; i <= nx_loc; ++i) {
      for (j = 1; j <= ny_loc; ++j) {
        ping(i, j, 0) = 10;
        pong(i, j, 0) = 10;
      }
    }

  /* Front */
  if (ping.ends[0] == nx)
    for (j = 1; j <= ny_loc; ++j) {
      for (k = 1; k <= nz_loc; ++k) {
        ping(nx_loc+1, j, k) = 10;
        pong(nx_loc+1, j, k) = 10;
      }
    }

  /* Down */
  if (ping.ends[1] == ny)
    for (i = 1; i <= nx_loc; ++i) {
      for (k = 1; k <= nz_loc; ++k) {
        ping(i, ny_loc+1, k) = 10;
        pong(i, ny_loc+1, k) = 10;
      }
    }

  /* Right */
  if (ping.ends[2] == nz)
    for (i = 1; i <= nx_loc; ++i) {
      for (j = 1; j <= ny_loc; ++j) {
        ping(i, j, nz_loc+1) = 1;
        pong(i, j, nz_loc+1) = 1;
      }
    }


}