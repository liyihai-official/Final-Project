#include "array.cpp"

template <class T>
void init_conditions_heat2d(final_project::array2d<T> & init)
{
  double xx, yy;
  std::size_t i, j, nx {init.Rows - 2}, ny {init.Cols - 2};

  init.fill(0);

  for (j = 0; j <= ny; ++j)
  {
    yy = (double) j / (ny+1);
    init(0, j) = 10;
  }


  for (j = 0; j <= ny; ++j)
  {
    yy = (double) j / (ny+1);
    init(nx+1, j) = 10;
  }


  for (i = 0; i <= nx; ++i)
    init(i, 0) = 10;


  for (i = 0; i <= nx+1; ++i) 
  {
    xx = (double) i / (nx+1);
    init(i,       ny+1) = 0;
  }
}

template <class T>
void init_conditions_heat2d(final_project::array2d_distribute<T>& ping, 
                            final_project::array2d_distribute<T>& pong)
{
  double xx, yy;
  std::size_t i, j, nx {ping.glob_Rows}, ny {ping.glob_Cols};

  ping.fill(0);
  pong.fill(0);
  
  /* U */
  if (ping.starts[0] == 1)
    for (j = 1; j <= ping.cols()-2; ++j)
    {
      yy = (double) j / (ny+1);
      ping(0, j) = 10;
      pong(0, j) = 10;
    }
  /* L */
  if (ping.starts[1] == 1)
    for (i = 1; i <= ping.cols()-2; ++i) {
      ping(i, 0) = 10;
      pong(i, 0) = 10;
    }

  /* D */
  if (ping.ends[0] == nx - 2)
    for (j = 0; j <= ping.cols()-2; ++j)
    {
      yy = (double) j / (ny+1);
      ping(ping.rows()-1, j) = 10;
      pong(ping.rows()-1, j) = 10;
    }


  /* R */
  if (ping.ends[1] == ny - 2)
    for (i = 1; i <= ping.cols()-2; ++i) {
      xx = (double) i / (nx+1);
      ping(i, ping.cols()-1) = 0;
      pong(i, ping.cols()-1) = 0;
    }
}