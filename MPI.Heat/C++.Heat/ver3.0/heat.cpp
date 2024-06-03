/**
 * @file heat.cpp
 * 
 * @brief This file contains the classes of namespace for heat equations in 
 *        parallel processing arrays.
 *        
 * 
 * 
 * @author LI Yihai
 * @version 3.1
 * @date Jun 2, 2024
 */
#ifndef FINAL_PROJECT_HEAT_CPP_LIYIHAI
#define FINAL_PROJECT_HEAT_CPP_LIYIHAI

#pragma once 
#include "multi_array/array_distribute.cpp"

namespace final_project
{
  namespace heat_equation
  {

    template <class T>
    class heat1d_pure_mpi {

      public:
        array1d_distribute<T> body;

      public:
        heat1d_pure_mpi(std::size_t gN, const int dims[1], MPI_Comm comm) 
        {
          // init_conditions_heat1d(body);
          body.distribute(gN, dims, comm);

          hx = (double) (max_x - min_x) / (double) (gN+1);
          dt = 0.5 * hx * hx / coff;
          dt = std::min(dt, 0.1);

          weight = coff * dt / (hx * hx);
          diag = -2.0 + hx * hx / (body.dimension * coff * dt);
        }
        
      public:
        void sweep_heat1d(heat1d_pure_mpi<T>& out)
        {
          for (std::size_t i = 1; i <= body.size() - 2; ++i)
          {
            double current = body(i);

            if (body.rank == 0)
            {

            }

            if (body.rank == body.num_proc - 1)
            {
              
            }

            out.body(i) =  weight * (body(i-1) + body(i+1)) 
                        + current * diag * weight;
          }
        }

      private:
        double coff {1};
        double diag, weight;
        double dt, hx;
        double min_x {0}, max_x {1};
    }; // class heat1d_pure_mpi




    template <class T>
    class heat2d_pure_mpi {

      public:
        array2d_distribute<T> body;

      public:
        heat2d_pure_mpi(std::size_t gRows, std::size_t gCols, const int dims[2], MPI_Comm comm)
        {
          body.distribute(gRows, gCols, dims, comm);

          hx = (double) (max_x - min_x) / (double) (gRows+1);
          hy = (double) (max_y - min_y) / (double) (gCols+1);
          
          dt = 0.25 * std::min({hx, hy}) * std::min({hx, hy}) / coff;
          dt = std::min(dt, 0.1);

          weight_x = coff * dt / (hx * hx);
          weight_y = coff * dt / (hy * hy);

          diag_x = -2.0 + hx * hx / (body.dimension * coff * dt);
          diag_y = -2.0 + hy * hy / (body.dimension * coff * dt);

        }

      public:
        void sweep_heat2d(heat2d_pure_mpi<T>& out)
        {
          std::size_t i,j, Nx {body.rows() - 2}, Ny{body.cols() - 2};

          // Neumann boundary condition
          /* Up */
          if (body.starts[0] == 1) 
            for (j = 1; j <= Ny; ++j) out.body(0, j) = body(0, j);

          /* Left */
          if (body.starts[1] == 1)
            for (i = 1; i <= Nx; ++i) out.body(i, 0) = body(i, 0); 
            
          /* Down */
          if (body.ends[0] == body.glob_Rows - 2)
            for (j = 1; j <= Ny; ++j) out.body(Nx+1, j) = body(Nx+1, j);

          /* Right */
          if (body.ends[1] == body.glob_Cols - 2)
            for (i = 1; i <= Nx; ++i) out.body(i, Ny+1) = body(i, Ny+1);

          /* Inside */
          for (i = 1; i <= Nx; ++i)
            for (j = 1; j <= Ny; ++j) 
            {
              double current = body(i, j);
              out.body(i,j) = weight_x * (body(i-1, j) + body(i+1, j))
                       + weight_y * (body(i, j-1) + body(i, j+1))
                       + current  * (diag_x*weight_x + diag_y*weight_y);
            }
        }

      // Update data
      private:
        double coff {1};
        double diag_x, diag_y, weight_x, weight_y;
        double dt, hx, hy;
        double min_x {0}, max_x {1};
        double min_y {0}, max_y {1};
    }; // class heat2d_pure_mpi


    template <class T>
    class heat3d_pure_mpi {
      public:
        array3d_distribute<T> body;
      
      public:
        heat3d_pure_mpi(std::size_t gRows, std::size_t gCols, std::size_t gHeights, 
                        const int dims[3], MPI_Comm comm)
        {
           body.distribute(gRows, gCols, gHeights, dims, comm);

            hx = (double) (max_x - min_x) / (double) (gRows+1);
            hy = (double) (max_y - min_y) / (double) (gCols+1);
            hz = (double) (max_z - min_z) / (double) (gHeights+1);

            dt = 0.125 * std::min({hx, hy, hz}) * std::min({hx, hy, hz}) / coff;
            dt = std::min(dt, 0.1);

            weight_x = coff * dt / (hx * hx);
            weight_y = coff * dt / (hy * hy);
            weight_z = coff * dt / (hz * hz);

            diag_x = -2.0 + hx * hx / (body.dimension * coff * dt);
            diag_y = -2.0 + hy * hy / (body.dimension * coff * dt);
            diag_z = -2.0 + hz * hz / (body.dimension * coff * dt);
        }
      
      public:
        void sweep_heat3d(heat3d_pure_mpi<T>& out)
        {
          std::size_t i,j,k, Nx {body.rows() - 2}, Ny{body.cols() - 2}, Nz{body.height() - 2};

          for (i = 1; i <= Nx; ++i) 
            for (j = 1; j <= Ny; ++j) 
              for (k = 1; k <= Nz; ++k) 
              {
                double current = body(i, j, k);
                out.body(i, j, k) = weight_x * (body(i-1, j, k) + body(i+1, j, k))
                                  + weight_y * (body(i, j-1, k) + body(i, j+1, k))
                                  + weight_z * (body(i, j, k-1) + body(i, j, k+1))
                                  + current  * (diag_x*weight_x + diag_y*weight_y + diag_z*weight_z);
              }
        } 

      // Update data
      private:
        double coff {1};
        double diag_x, diag_y, diag_z, weight_x, weight_y, weight_z;
        double dt, hx, hy, hz;
        double min_x {0}, max_x {1};
        double min_y {0}, max_y {1};
        double min_z {0}, max_z {1};
        
    }; // class heat3d_pure_mpi

  } // namespace heat_equation



} // namespace final_project


#endif // FINAL_PROJECT_HEAT_CPP_LIYIHAI