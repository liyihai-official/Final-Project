#ifndef EVOLVE_HYBRID_1_HPP
#define EVOLVE_HYBRID_1_HPP

#pragma once 
#include <cmath>
#include "array.hpp"
#include "fdm/heat.hpp"


/// @brief 
/// @tparam T 
/// @tparam NumDim 
/// @param in 
/// @param out 
/// @param hq 
/// @return 
template <typename T, std::size_t NumDim>
T update_ping_pong_omp1(  final_project::array::array_distribute<T, NumDim> & in, 
                          final_project::array::array_distribute<T, NumDim> & out,
                          final_project::heat_equation<T, NumDim>           & hq)
{
  T diff {0};

  if (NumDim == 2)
  {
    #pragma omp for
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
      {
        T current {in.get_array().__local_array(i,j)};
        out.get_array().__local_array(i,j) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j) + in.get_array().__local_array(i+1,j))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1) + in.get_array().__local_array(i,j+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1]);

        diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
      }
    }
  }

  if (NumDim == 3)
  {
    #pragma omp for
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
      {
        for (std::size_t k = 1; k < in.get_array().__local_array.__shape[2] - 1; ++k)
        {
          T current {in.get_array().__local_array(i,j,k)};
          
          out.get_array().__local_array(i,j,k) = 
              hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
            + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
            + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
            + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

          diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
        }
      }
    }
  }

  return diff;
}


template <typename T, std::size_t NumDim>
T update_ping_pong_omp_bulk(  final_project::array::array_distribute<T, NumDim> & in, 
                              final_project::array::array_distribute<T, NumDim> & out,
                              final_project::heat_equation<T, NumDim>           & hq)
{

  T diff {0.0};

  if ( NumDim == 2 )
  {
    #pragma omp for
    for (std::size_t i = 2; i < in.get_array().__local_array.__shape[0] - 2; ++i)
    {
      for (std::size_t j = 2; j < in.get_array().__local_array.__shape[1] - 2; ++j)
      {
        T current {in.get_array().__local_array(i,j)};
        out.get_array().__local_array(i,j) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j) + in.get_array().__local_array(i+1,j))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1) + in.get_array().__local_array(i,j+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1]);

        diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
      }
    }
  }



  if ( NumDim == 3 )
  {
    #pragma omp for
    for (std::size_t i = 2; i < in.get_array().__local_array.__shape[0] - 2; ++i)
    {
      for (std::size_t j = 2; j < in.get_array().__local_array.__shape[1] - 2; ++j)
      {
        for (std::size_t k = 2; k < in.get_array().__local_array.__shape[2] - 2; ++k)
        {
          T current {in.get_array().__local_array(i,j,k)};
          
          out.get_array().__local_array(i,j,k) = 
              hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
            + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
            + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
            + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

          diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
        }
      }
    }
  }

  return diff;
}




template <typename T, std::size_t NumDim>
T update_ping_pong_omp_boundary(  final_project::array::array_distribute<T, NumDim> & in, 
                                  final_project::array::array_distribute<T, NumDim> & out,
                                  final_project::heat_equation<T, NumDim>           & hq)
{
  T diff {0.0};

  if ( NumDim == 2 )
  {
    #pragma omp for 
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      std::size_t j = 1;
      T current {in.get_array().__local_array(i,j)};

      out.get_array().__local_array(i,j) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j) + in.get_array().__local_array(i+1,j))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1) + in.get_array().__local_array(i,j+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1]);

      diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
    }

    #pragma omp for 
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      std::size_t j = in.get_array().__local_array.__shape[1] - 2;
      T current {in.get_array().__local_array(i,j)};

      out.get_array().__local_array(i,j) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j) + in.get_array().__local_array(i+1,j))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1) + in.get_array().__local_array(i,j+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1]);
      
      diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
    }

    #pragma omp for 
    for (std::size_t j = 2; j < in.get_array().__local_array.__shape[1] - 2; ++j)
    {
      std::size_t i = 1;
      T current {in.get_array().__local_array(i,j)};

      out.get_array().__local_array(i,j) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j) + in.get_array().__local_array(i+1,j))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1) + in.get_array().__local_array(i,j+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1]);

      diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
    }

    #pragma omp for 
    for (std::size_t j = 2; j < in.get_array().__local_array.__shape[1] - 2; ++j)
    {
      std::size_t i = in.get_array().__local_array.__shape[0] - 2;
      T current {in.get_array().__local_array(i,j)};

      out.get_array().__local_array(i,j) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j) + in.get_array().__local_array(i+1,j))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1) + in.get_array().__local_array(i,j+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1]);

      diff += std::pow(out.get_array().__local_array(i,j) - in.get_array().__local_array(i,j), 2);
    }
  }
  

  if (NumDim == 3)
  {
    #pragma omp for
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      std::size_t j {1}, k {1};
      T current {in.get_array().__local_array(i,j,k)};
      
      out.get_array().__local_array(i,j,k) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
        + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

      diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
    }

    #pragma omp for
    for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
    {
      std::size_t i {1}, k {1};
        T current {in.get_array().__local_array(i,j,k)};
        
        out.get_array().__local_array(i,j,k) = 
            hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
          + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
          + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
          + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

        diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
    }

    #pragma omp for
    for (std::size_t k = 1; k < in.get_array().__local_array.__shape[2] - 1; ++k)
    {
      std::size_t i {1}, j {1};
      T current {in.get_array().__local_array(i,j,k)};
      
      out.get_array().__local_array(i,j,k) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
        + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

      diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
    }

    // 
    #pragma omp for
    for (std::size_t i = 1; i < in.get_array().__local_array.__shape[0] - 1; ++i)
    {
      std::size_t j {in.get_array().__local_array.__shape[1] - 2}, 
                  k {in.get_array().__local_array.__shape[2] - 2};
      T current {in.get_array().__local_array(i,j,k)};
      
      out.get_array().__local_array(i,j,k) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
        + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

      diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
    }

    #pragma omp for
    for (std::size_t j = 1; j < in.get_array().__local_array.__shape[1] - 1; ++j)
    {
      std::size_t i {in.get_array().__local_array.__shape[0] - 2}, 
                  k {in.get_array().__local_array.__shape[2] - 2};
        T current {in.get_array().__local_array(i,j,k)};
        
        out.get_array().__local_array(i,j,k) = 
            hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
          + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
          + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
          + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

        diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
    }

    #pragma omp for
    for (std::size_t k = 1; k < in.get_array().__local_array.__shape[2] - 1; ++k)
    {
      std::size_t i {in.get_array().__local_array.__shape[0] - 2}, 
                  j {in.get_array().__local_array.__shape[1] - 2};
                  
      T current {in.get_array().__local_array(i,j,k)};
      
      out.get_array().__local_array(i,j,k) = 
          hq.weights[0] * (in.get_array().__local_array(i-1,j,k) + in.get_array().__local_array(i+1,j,k))
        + hq.weights[1] * (in.get_array().__local_array(i,j-1,k) + in.get_array().__local_array(i,j+1,k))
        + hq.weights[2] * (in.get_array().__local_array(i,j,k-1) + in.get_array().__local_array(i,j,k+1))
        + current * (hq.diags[0]*hq.weights[0] + hq.diags[1]*hq.weights[1] + hq.diags[2]*hq.weights[2]);

      diff += std::pow(out.get_array().__local_array(i,j,k) - in.get_array().__local_array(i,j,k), 2);
    }
  }

  return diff;

}















#endif // end of define EVOLVE_HYBRID_1_HPP