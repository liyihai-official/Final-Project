/// 
/// @file Heat_2D.hpp
/// @brief An object of Heat Equation 2D, with intergrade features.
///
/// @author LI Yihai
/// @version 6.0
///
#ifndef FINAL_PROJECT_HEAT_2D_HPP
#define FINAL_PROJECT_HEAT_2D_HPP

#pragma once
#include <pde/Heat.hpp>


namespace final_project {  namespace pde {

/// @class Heat_2D<T>
/// @brief A 2D Heat Equation object with internal functions for solving this system.
/// @tparam T The Value type.
template <typename T>
  class Heat_2D : protected Heat_Base<T, 2> 
  {
    using ICFunction = std::function<T(T, T)>;
    using BCFunction = std::function<T(T, T, T)>;
    
    public:
    Heat_2D() = delete;
    Heat_2D(mpi::environment &, size_type, size_type);

    void SetHeatBC(BoundaryConditions_2D<T> &, BCFunction, BCFunction, BCFunction, BCFunction);

    void SetHeatInitC(InitialConditions::Init_2D<T> &     );

    Integer solve_pure_mpi(       const T, const Integer=100, const Integer=0);
    Integer solve_hybrid_mpi_omp( const T, const Integer=100, const Integer=0);
    Integer solve_hybrid2_mpi_omp(const T, const Integer=100, const Integer=0);

    void SaveToBinary(const String);

    private:
    void exchange_ping_pong_SR()      override;
    void exchange_ping_pong_I()       override;

    T update_ping_pong(const T)       override;
    T update_ping_pong_omp(const T)   override;
    T update_ping_pong_bulk()         override;
    T update_ping_pong_edge(const T)  override;
    void switch_in_out();

    private:
    std::unique_ptr<BoundaryConditions_2D<T>>       BC_2D;
    std::unique_ptr<InitialConditions::Init_2D<T>>  IC_2D;
    
    mpi::array_Cart<T, 2>         in, out;
    multi_array::array_base<T, 2> gather;

    Bool converge;

    friend InitialConditions::Init_2D<T>;
    friend BoundaryConditions_2D<T>;

  }; // class Heat_2D

}}


///
/// Inline Function Definitions
///

// Standard Library
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h> // OpenMP 
#endif // end define OpenMP


namespace final_project { namespace pde {

/// @brief Constructs the Heat Equation system in 2D space, with grid size [nx] by [ny].
/// @tparam T Value type of this system.
/// @param env An @c MPI_Comm environment of this system.
/// @param nx Grid size in dimension [0].
/// @param ny Grid size in dimension [1].
template <typename T>
  inline
  Heat_2D<T>::Heat_2D(mpi::environment & env, size_type nx, size_type ny)
: Heat_Base<T, 2>(env, nx, ny), BC_2D {nullptr}, IC_2D {nullptr}, converge {false}
  {
    in  = mpi::array_Cart<T, 2>(env, nx, ny);
    out = mpi::array_Cart<T, 2>(env, nx, ny);

    gather = multi_array::array_base<T, 2>(nx, ny);

    in.array().__loc_array.fill(0);
    out.array().__loc_array.fill(0);    
  }

/// @brief Save gather.data to given filename.
/// @tparam T Value type
/// @param filename String, filename 
template <typename T>
  inline void 
  Heat_2D<T>::SaveToBinary(const String filename)
  {
      if (in.topology().rank == 0)
        gather.saveToBinary(filename);
  }


/// @brief Setup Boundary Conditions to this Object
/// @tparam T Value type
/// @param BC Reference of the objet BoundaryConditions_2D<T>
/// @param FuncDim00 function in Dimension 0, source site.
/// @param FuncDim01 function in Dimension 0, dest site.
/// @param FuncDim10 function in Dimension 1, source site.
/// @param FuncDim11 function in Dimension 1, dest site.
template <typename T> 
  inline void 
  Heat_2D<T>::SetHeatBC(
    BoundaryConditions_2D<T> & BC, 
    BCFunction FuncDim00, BCFunction FuncDim01, 
    BCFunction FuncDim10, BCFunction FuncDim11)
  {
    BC_2D = std::make_unique<BoundaryConditions_2D<T>>(BC);

    BC_2D->SetBC(*this, 
      FuncDim00, FuncDim01, 
      FuncDim10, FuncDim11);
  }



/// @brief Setup the Initial Conditions of this system.
/// @tparam T Value type
/// @param IC A set InitialConditions::Init_2D<T> object that 
///             includes the function of initial condition.
template <typename T>
  inline void 
  Heat_2D<T>::SetHeatInitC(InitialConditions::Init_2D<T> & IC)
  {
    IC_2D = std::make_unique<InitialConditions::Init_2D<T>>(IC);
    IC_2D->SetUpInit(*this);
  }

/// @brief Switch private members [in] and [out]
/// @tparam T Value type.
template <typename T>
  inline void 
  Heat_2D<T>::switch_in_out() 
{ in.swap(out); }

/// @brief A ping pong update strategy based update function. The updated values are stored in 
///       private object mpi::array_Cart<T, 2> [out].
/// @tparam T Value Type.
/// @return Return the difference of each iteration.
template <typename T>
  inline T 
  Heat_2D<T>::update_ping_pong(const T time)
  {
    T diff {0.0};
    size_type i {1}, j {1};
    for (i=1; i < in.topology().__local_shape[0] - 1; ++i)
    {
      for (j=1; j < in.topology().__local_shape[1] - 1; ++j)
      {
        T current {in(i,j)};
        out(i,j) =
            this->weights[0] * (in(i-1,j) + in(i+1,j))
          + this->weights[1] * (in(i,j-1) + in(i,j+1))
          + current * (
              this->diags[0]*this->weights[0] 
            + this->diags[1]*this->weights[1]
            );

        diff += std::pow(current - out(i,j), 2);
      }
    }
      
    return diff;
  }


/// @brief Using OpenMP, 
///         A ping pong update strategy based update function. The updated values are stored in 
///         private object mpi::array_Cart<T, 2> [out].
/// @tparam T Value Type.
/// @return Return the difference of each iteration.
template <typename T>
  inline T 
  Heat_2D<T>::update_ping_pong_omp(const T time)
  {
    T diff {0.0};

    #pragma omp for
    for (size_type i=1; i < in.topology().__local_shape[0] - 1; ++i)
    {
      for (size_type j=1; j < in.topology().__local_shape[1] - 1; ++j)
      {
        T current {in(i,j)};
        out(i,j) =
            this->weights[0] * (in(i-1,j) + in(i+1,j))
          + this->weights[1] * (in(i,j-1) + in(i,j+1))
          + current * (
              this->diags[0]*this->weights[0] 
            + this->diags[1]*this->weights[1]
            );

        diff += std::pow(current - out(i,j), 2);
      }
    }

    return diff;
  }


/// @brief Update bulk of grid surrounded by Edges
/// @tparam T Value type
/// @return Return the difference
template <typename T>
  inline T
  Heat_2D<T>::update_ping_pong_bulk()
  {
    T diff {0.0};

    #pragma omp for
    for (size_type i=2; i < in.topology().__local_shape[0]-2; ++i)
    {
      for (size_type j=2; j < in.topology().__local_shape[1]-2; ++j)
      {
        T current {in(i,j)};
        out(i,j) =
            this->weights[0] * (in(i-1,j) + in(i+1,j))
          + this->weights[1] * (in(i,j-1) + in(i,j+1))
          + current * (
              this->diags[0]*this->weights[0] 
            + this->diags[1]*this->weights[1]
            );

        diff += std::pow(current - out(i,j), 2);
      }
    }

    return diff;
  }


/// @brief Update the Edges surrounding Bulk
/// @tparam T Value type
/// @return Return the difference
template <typename T>
  inline T
  Heat_2D<T>::update_ping_pong_edge(const T time)
  { 
    T diff {0.0};
    // auto local_shape {in.topology().__local_shape};
    // size_type i, j;
    #pragma omp single
    {
      size_type i = 1; size_type j = 1;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2); 
    }

    #pragma omp single
    {
      size_type i = 1; size_type j = in.topology().__local_shape[1] - 2;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2); 
    }

    #pragma omp single
    {
    size_type i = in.topology().__local_shape[0] - 2; size_type j = 1;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2); 
    }

    #pragma omp single
    {
      size_type i = in.topology().__local_shape[0] - 2; size_type j = in.topology().__local_shape[1] - 2;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2); 
    }



    
    #pragma omp for
    for (size_type i = 2; i <in.topology().__local_shape[0]-2; ++i)
    {
      size_type j = 1;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2); 
    }

    
    #pragma omp for
    for (size_type i = 2; i < in.topology().__local_shape[0]-2; ++i)
    {
      size_type j = in.topology().__local_shape[1] - 2;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2);
    }

    
    #pragma omp for
    for (size_type j = 2; j < in.topology().__local_shape[1]-2; ++j)
    {
      size_type i = 1;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2);
    }

    #pragma omp for
    for (size_type j = 2; j < in.topology().__local_shape[1]-2; ++j)
    {
      size_type i = in.topology().__local_shape[0] - 2;
      T current {in(i,j)};
      out(i,j) =
          this->weights[0] * (in(i-1,j) + in(i+1,j))
        + this->weights[1] * (in(i,j-1) + in(i,j+1))
        + current * (
            this->diags[0]*this->weights[0] 
          + this->diags[1]*this->weights[1]
          );
      diff += std::pow(current - out(i,j), 2);
    }

    return diff;
  }

/// @brief A communication function using @c MPI_Sendrecv , the blocking @c MPI_Send and @c MPI_Recv  .
/// @tparam T Value type.
template <typename T>
  inline void 
  Heat_2D<T>::exchange_ping_pong_SR()
  {
    Integer flag {0}; size_type dim {0};
    auto n_size {in.topology().__local_shape[dim]};
    MPI_Sendrecv( &in(1 ,1), 1, in.topology().halos[dim], in.topology().nbr_src[dim], flag,
                  &in(n_size-1 ,1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &in(n_size-2 ,1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag,
                  &in(0 ,1)       , 1, in.topology().halos[dim], in.topology().nbr_src[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    flag = 1; dim = 1;
    n_size = in.topology().__local_shape[dim];
    MPI_Sendrecv( &in(1,        1), 1, in.topology().halos[dim], in.topology().nbr_src[dim], flag,
                  &in(1, n_size-1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &in(1, n_size-2), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag,
                  &in(1, 0)       , 1, in.topology().halos[dim], in.topology().nbr_src[dim] , flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);
  }


/// @brief A none-blocking communication function using @c MPI_ISend and @c MPI_IRecv .
/// @tparam T value type.
template <typename T>
  inline void
  Heat_2D<T>::exchange_ping_pong_I()
  {
    MPI_Request reqs[8];
    Integer req_cnt {0}, flag {0}; size_type dim {0};

    auto n_size {in.topology().__local_shape[dim]};
    // Send and receive for the first dimension
    MPI_Irecv(&in(n_size-1, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(1, 1),        1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);

    MPI_Irecv(&in(0, 1),        1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(n_size-2, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);


    flag = 1; dim = 1;
    n_size = in.topology().__local_shape[dim];
    // Send and receive for the second dimension
    MPI_Irecv(&in(1, n_size-1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(1,        1), 1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);

    MPI_Irecv(&in(1,        0), 1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(1, n_size-2), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);



    MPI_Waitall(req_cnt, reqs, MPI_STATUS_IGNORE);
  }


/// @brief A integrated function of solve this system, purely based on @c MPI .
/// @tparam T Value type
/// @param tol Tolerance
/// @param nsteps Maximum number of iterations.
/// @param root The root process.
/// @return The number of convergence of the steps it takes to converge.
template <typename T>
  Integer 
  Heat_2D<T>::solve_pure_mpi(const T tol, const Integer nsteps, const Integer root)
    {
      T ldiff {0.0}, gdiff {0.0}, time {0}; MPI_Datatype DiffType {mpi::get_mpi_type<T>()};
      Integer iter {1};

#ifndef NDEBUG
FINAL_PROJECT_MPI_ASSERT_GLOBAL((BC_2D != nullptr && IC_2D != nullptr));
FINAL_PROJECT_ASSERT(BC_2D->isSetUpBC && IC_2D->isSetUpInit);
#endif 

#ifndef NDEBUG
{
  if (root == in.topology().rank)
  {
    std::cout << 2 << " Dimension Simulation Parameters: " << std::endl;
    std::cout << "\tRows: "       << in.topology().__local_shape[0]-2 
              << "\n\tColumns: "  << in.topology().__local_shape[1]-2       << std::endl;
    std::cout << "\tTime steps: " << nsteps << std::endl;
    std::cout << "\tTolerance: "  << tol    << std::endl;

    std::cout << "MPI Parameters: "             << std::endl;
    std::cout << "\tNumber of MPI Processes: "  << in.topology().num_procs  << std::endl;
    std::cout << "\tRoot Process: "             << root                     << std::endl;

    std::cout << "Heat Parameters: "    << std::endl;
    std::cout << "\tCoefficient: "      << this->coff       << "\n"
              << "\tTime resolution: "  << this->dt         << "\n"
              << "\tWeights: "          << this->weights[0] << ", " 
                                        << this->weights[1] << "\n"
              << "\tdxs: "              << this->dxs[0]     << ", " 
                                        << this->dxs[1]     << std::endl;
  }
}
#endif // end NDEBUG


      Double t0 {MPI_Wtime()};
      for (iter = 1; iter < nsteps; ++iter)
      {
        time = iter*this->dt; 
        exchange_ping_pong_SR();
        ldiff = update_ping_pong(time);
        BC_2D->UpdateBC(*this, time);
        MPI_Allreduce(&ldiff, &gdiff, 1, DiffType, MPI_SUM, in.topology().comm_cart);

#ifndef NDEBUG
mpi::Gather(gather, in, root);
if (in.topology().rank == root) 
{
  std::cout << std::fixed << std::setprecision(13) << std::setw(15) << gdiff 
            << std::endl;
}
#endif 

        switch_in_out();

        if (gdiff  <= tol) {
          converge = true;
          break;
        }
      }
      Double t1 {MPI_Wtime() - t0};

      if (converge)
      {
        Double total {0};
        mpi::Gather(gather, in, root);
        MPI_Reduce(&t1, &total, 1, MPI_DOUBLE, MPI_MAX, root, in.topology().comm_cart);
        if (root == in.topology().rank)
          std::cout << "Total Converge time: " << total << "\n" 
                    << "Iterations: " << iter << std::endl;
      } else {
        if (in.topology().rank == root) std::cout << "Fail to converge" << std::endl;
      }
    
#ifndef NDEBUG // Gather 
mpi::Gather(gather, in, root);
#endif

      return iter;
    }  



/// @brief Solve this object using MPI/OpenMP Hybrid strategy, version 1
/// @tparam T Value type
/// @param tol Tolerance
/// @param nsteps Maximum Iterating Steps
/// @param root Root Processor's ID.
/// @return number of iterations
template <typename T>
  Integer
  Heat_2D<T>::solve_hybrid_mpi_omp(const T tol, const Integer nsteps, const Integer root)
  {
    T ldiff {0.0}, gdiff {0.0}; MPI_Datatype DiffType {mpi::get_mpi_type<T>()};
    Integer num_threads {1};

#ifndef NDEBUG
FINAL_PROJECT_MPI_ASSERT_GLOBAL((BC_2D != nullptr && IC_2D != nullptr));
FINAL_PROJECT_ASSERT(BC_2D->isSetUpBC == true && IC_2D->isSetUpInit == true);
#endif 


    #pragma omp parallel num_threads(2)
    {
      T omp_ldiff {0.0}, time {0};

#ifdef _OPENMP
  #pragma omp master
  num_threads = omp_get_num_threads();
#endif // end _OPENMP
      
#ifndef NDEBUG
{
  #pragma omp single
  {
    if (root == in.topology().rank)
    {
      std::cout << 2 << " Dimension Simulation Parameters: " << std::endl;
      std::cout << "\tRows: "       << in.topology().__local_shape[0]-2 
                << "\n\tColumns: "  << in.topology().__local_shape[1]-2       << std::endl;
      std::cout << "\tTime steps: " << nsteps << std::endl;
      std::cout << "\tTolerance: "  << tol    << std::endl;

      std::cout << "MPI Parameters: "             << std::endl;
      std::cout << "\tNumber of MPI Processes: "  << in.topology().num_procs  << std::endl;
      std::cout << "\tRoot Process: "             << root                     << std::endl;
      std::cout << "\tNumber of Threads: "        << num_threads              << std::endl;

      std::cout << "Heat Parameters: "    << std::endl;
      std::cout << "\tCoefficient: "      << this->coff       << "\n"
                << "\tTime resolution: "  << this->dt         << "\n"
                << "\tWeights: "          << this->weights[0] << ", " 
                                          << this->weights[1] << "\n"
                << "\tdxs: "              << this->dxs[0]     << ", " 
                                          << this->dxs[1]     << std::endl;
    }
  }
}
#endif // end NDEBUG

    Integer iter = 1;
    Double t0 {MPI_Wtime()};
    for (iter = 1; iter <= nsteps; ++iter)
    {
      #pragma omp single
      exchange_ping_pong_I();
      #pragma omp barrier

      ldiff = 0; 
      time = iter*this->dt;

      omp_ldiff = update_ping_pong_omp(time);
      
      #pragma omp critical 
      ldiff += omp_ldiff;
      #pragma omp barrier

      #pragma omp single
      {
        BC_2D->UpdateBC(*this, time);
        MPI_Allreduce(&ldiff, &gdiff, 1, DiffType, MPI_SUM, in.topology().comm_cart);

#ifndef NDEBUG
mpi::Gather(gather, in, root);
if (in.topology().rank == root) 
{
  std::cout << std::fixed << std::setprecision(13) << std::setw(15) << gdiff 
            << std::endl;
}
#endif

        switch_in_out();

        if (gdiff  <= tol) { converge = true; }
      } // end single

      if ( converge ) { break; }
    } 
    Double t1 {MPI_Wtime() - t0};


#pragma omp master
{
    if (converge)
    {
      Double total {0};
      mpi::Gather(gather, in, root);
      MPI_Reduce(&t1, &total, 1, MPI_DOUBLE, MPI_MAX, root, in.topology().comm_cart);
      if (root == in.topology().rank)
        std::cout << "Total Converge time: " << total << "\n" 
                  << "Iterations: " << iter << std::endl;
    } else {
      if (in.topology().rank == root) std::cout << "Fail to converge" << std::endl;
    }
  
#ifndef NDEBUG // Gather 
mpi::Gather(gather, in, root);
#endif
}

    } // end omp parallel 

    return 0;

  }



/// @brief Solve this object using MPI/OpenMP Hybrid strategy, version 1
/// @tparam T Value type
/// @param tol Tolerance
/// @param nsteps Maximum Iterating Steps
/// @param root Root Processor's ID.
/// @return number of iterations
template <typename T>
  Integer
  Heat_2D<T>::solve_hybrid2_mpi_omp(const T tol, const Integer nsteps, const Integer root)
  {
    T ldiff {0.0}, gdiff {0.0}; MPI_Datatype DiffType {mpi::get_mpi_type<T>()};

    Integer num_threads {1}; // omp_threads

#ifndef NDEBUG
FINAL_PROJECT_MPI_ASSERT_GLOBAL((BC_2D != nullptr && IC_2D != nullptr));
FINAL_PROJECT_ASSERT(BC_2D->isSetUpBC == true && IC_2D->isSetUpInit == true);
#endif 


  MPI_Request reqs[8];
    #pragma omp parallel num_threads(4)
    {
      Integer omp_id { omp_get_thread_num() };
      T omp_ldiff_bulk {0.0}, omp_ldiff_edge {0.0}, time {0.0};

    #ifdef _OPENMP
      #pragma omp master
      num_threads = omp_get_num_threads();
    #endif // end _OPENMP
      
#ifndef NDEBUG
{
  #pragma omp single
  {
    if (root == in.topology().rank)
    {
      std::cout << 2 << " Dimension Simulation Parameters: " << std::endl;
      std::cout << "\tRows: "       << in.topology().__local_shape[0]-2 
                << "\n\tColumns: "  << in.topology().__local_shape[1]-2       << std::endl;
      std::cout << "\tTime steps: " << nsteps << std::endl;
      std::cout << "\tTolerance: "  << tol    << std::endl;

      std::cout << "MPI Parameters: "             << std::endl;
      std::cout << "\tNumber of MPI Processes: "  << in.topology().num_procs  << std::endl;
      std::cout << "\tRoot Process: "             << root                     << std::endl;
      std::cout << "\tNumber of Threads: "        << num_threads              << std::endl;

      std::cout << "Heat Parameters: "    << std::endl;
      std::cout << "\tCoefficient: "      << this->coff       << "\n"
                << "\tTime resolution: "  << this->dt         << "\n"
                << "\tWeights: "          << this->weights[0] << ", " 
                                          << this->weights[1] << "\n"
                << "\tdxs: "              << this->dxs[0]     << ", " 
                                          << this->dxs[1]     << std::endl;
    }
  }
}
#endif // end NDEBUG

      Integer iter {1};
      Double t0 { MPI_Wtime() };
      for (; iter < nsteps; ++iter )
      {
        ldiff = 0; time = iter*this->dt;

        // omp_ldiff_bulk = update_ping_pong_bulk();

        // #pragma omp single 
        // exchange_ping_pong_I();
        // #pragma omp barrier

// ////////////////////////////////////////////////////////////////////////////////////////////////////////

  #pragma omp single
  {
    Integer req_cnt {0}, flag {0}; size_type dim {0};
    auto n_size {in.topology().__local_shape[dim]};
    // Send and receive for the first dimension
    MPI_Irecv(&in(n_size-1, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(1, 1),        1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);

    MPI_Irecv(&in(0, 1),        1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(n_size-2, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);


    flag = 1; dim = 1;
    n_size = in.topology().__local_shape[dim];
    // Send and receive for the second dimension
    MPI_Irecv(&in(1, n_size-1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(1,        1), 1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);

    MPI_Irecv(&in(1,        0), 1, in.topology().halos[dim], in.topology().nbr_src[dim],  flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);
    MPI_Isend(&in(1, n_size-2), 1, in.topology().halos[dim], in.topology().nbr_dest[dim], flag, 
                in.topology().comm_cart, &reqs[req_cnt++]);

  }

  omp_ldiff_bulk = update_ping_pong_bulk();

  #pragma omp single
  MPI_Waitall(8, reqs, MPI_STATUS_IGNORE); 

// ////////////////////////////////////////////////////////////////////////////////////////////////////////



#pragma omp barrier

        omp_ldiff_edge = update_ping_pong_edge(time);

        #pragma omp critical
        {
          ldiff += (omp_ldiff_bulk + omp_ldiff_edge);
        }
        #pragma omp barrier

        #pragma omp single
        {
          MPI_Allreduce(&ldiff, &gdiff, 1, DiffType, MPI_SUM, in.topology().comm_cart);

#ifndef NDEBUG
mpi::Gather(gather, in, root);
if (in.topology().rank == root) 
{
  std::cout << std::fixed << std::setprecision(13) << std::setw(15) << gdiff 
            << std::endl;
}
#endif
          BC_2D->UpdateBC(*this, time);
          switch_in_out();

          if (gdiff  <= tol) { converge = true; }
        } // end single

        if ( converge ) { break; }
      }
      Double t1 { MPI_Wtime() - t0 };

#pragma omp master
{
  if (converge)
    {
      Double total {0};
      mpi::Gather(gather, in, root);
      MPI_Reduce(&t1, &total, 1, MPI_DOUBLE, MPI_MAX, root, in.topology().comm_cart);
      if (root == in.topology().rank)
        std::cout << "Total Converge time: " << total << "\n" 
                  << "Iterations: " << iter << std::endl;
    } else {
      if (in.topology().rank == root) std::cout << "Fail to converge" << std::endl;
    }
#ifndef NDEBUG // Gather 
mpi::Gather(gather, in, root);
#endif
}


    } // end of omp parallel

    return 0;
  }


} // namespace pde
} // namespace final_project



#endif // end of define FINAL_PROJECT_HEAT_2D_HPP