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
#include "array.cpp"


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

  init.fill(0);

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