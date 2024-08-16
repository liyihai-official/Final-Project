///
/// @file Heat_3D.hpp
/// @brief An object of heat equation 3D, with integrated features.
///   
/// @author LI Yihai 
/// @version 6.0


#ifndef FINAL_PROJECT_HEAT_3D_HPP
#define FINAL_PROJECT_HEAT_3D_HPP

#pragma once 
#include <pde/Heat.hpp>


namespace final_project { namespace pde {


/// @class Heat_3D<T>
/// @brief A 3D Heat Equation object with internal functions for solving this system.
/// @tparam T The Value type.
template <typename T>
  class Heat_3D : protected Heat_Base<T, 3>
  {
    using ICFunction = std::function<T(T, T, T)>;
    using BCFunction = std::function<T(T, T, T, T)>;

    public:
    Heat_3D() = default;
    Heat_3D(mpi::environment &, size_type, size_type, size_type);

    void SetHeatBC(BoundaryConditions_3D<T> &, 
      BCFunction, BCFunction, 
      BCFunction, BCFunction, 
      BCFunction, BCFunction);

    void SetHeatInitC(InitialConditions::Init_3D<T> & );

    Integer solve_pure_mpi(       const T, const Integer=100, const Integer=0);
    Integer solve_hybrid_mpi_omp( const T, const Integer=100, const Integer=0);
    Integer solve_hybrid2_mpi_omp(const T, const Integer=100, const Integer=0);

    void SaveToBinary( const String );

    private:
    void exchange_ping_pong_SR()  override;
    void exchange_ping_pong_I()   override;

    T update_ping_pong(const T)       override;
    T update_ping_pong_omp(const T)   override;
    T update_ping_pong_bulk()         override;
    T update_ping_pong_edge(const T)  override;
    void switch_in_out();

    private: // RAII design
    std::unique_ptr<BoundaryConditions_3D<T>>       BC_3D;
    std::unique_ptr<InitialConditions::Init_3D<T>>  IC_3D;

    mpi::array_Cart<T, 3>       in, out;
    multi_array::array_base<T, 3> gather;
    Bool converge;

    friend InitialConditions::Init_3D<T>;
    friend BoundaryConditions_3D<T>;

  }; // class Heat_3D




} // namespace pde
} // namespace final_project


///
///
/// --------------------------- Inline Function Definitions  ---------------------------  ///
///
///


// Standard Library
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h> // OpenMP 
#endif // end define OpenMP



namespace final_project { namespace pde {


/// @brief Constructs the Heat Equation system in 2D space, with grid size [nx] by [ny] by [nz].
/// @tparam T Value type of this system.
/// @param env An @c MPI_Comm environment of this system.
/// @param nx Grid size in dimension [0].
/// @param ny Grid size in dimension [1].
/// @param nz Grid size in dimension [2].
template <typename T>
  inline 
  Heat_3D<T>::Heat_3D(mpi::environment & env, size_type nx, size_type ny, size_type nz)
: Heat_Base<T, 3>(env, nx, ny, nz), BC_3D {nullptr}, IC_3D {nullptr}, converge {false}
  {
    in  = mpi::array_Cart<T, 3>(env, nx, ny, nz);
    out = mpi::array_Cart<T, 3>(env, nx, ny, nz);

    gather = multi_array::array_base<T, 3>(nx, ny, nz);

    in.array().__loc_array.fill(0);
    out.array().__loc_array.fill(0);    
  }


/// @brief Save gather.data to given filename.
/// @tparam T Value type
/// @param filename String, filename 
template <typename T>
  inline void 
  Heat_3D<T>::SaveToBinary(const String filename)
  {
    if (in.topology().rank == 0)
      gather.saveToBinary(filename);
  }




template <typename T>
  inline void 
  Heat_3D<T>::SetHeatBC(
    BoundaryConditions_3D<T> & BC, 
    BCFunction FuncDim00, BCFunction FuncDim01,
    BCFunction FuncDim10, BCFunction FuncDim11,
    BCFunction FuncDim20, BCFunction FuncDim21)
  {
    BC_3D = std::make_unique<BoundaryConditions_3D<T>>(BC);

    BC_3D->SetBC(*this, 
      FuncDim00, FuncDim01,
      FuncDim10, FuncDim11,
      FuncDim20, FuncDim21);
  }


/// @brief 
/// @tparam T 
/// @param IC 
template <typename T>
  inline void 
  Heat_3D<T>::SetHeatInitC(InitialConditions::Init_3D<T> & IC)
  {
    IC_3D = std::make_unique<InitialConditions::Init_3D<T>>(IC);
    IC_3D->SetUpInit(*this);
  }

/// @brief 
/// @tparam T 
template <typename T>
  inline void
  Heat_3D<T>::switch_in_out()
{ in.swap(out); }

/// @brief 
/// @tparam T 
template <typename T>
  inline void 
  Heat_3D<T>::exchange_ping_pong_SR()
  {

    Integer flag {0}; size_type dim {0};
    auto n_size {in.topology().__local_shape[dim]};
    MPI_Sendrecv( &in(1       , 1, 1), 1, in.topology().halos[dim], in.topology().nbr_src[dim],   flag,
                  &in(n_size-1, 1, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim],  flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &in(n_size-2, 1, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim],  flag,
                  &in(0       , 1, 1), 1, in.topology().halos[dim], in.topology().nbr_src[dim],   flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    flag = 1; dim = 1;
    n_size = in.topology().__local_shape[dim];
    MPI_Sendrecv( &in(1,        1, 1), 1, in.topology().halos[dim], in.topology().nbr_src[dim],   flag,
                  &in(1, n_size-1, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim],  flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &in(1, n_size-2, 1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim],  flag,
                  &in(1, 0,        1), 1, in.topology().halos[dim], in.topology().nbr_src[dim],   flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    flag = 2; dim = 2;
    n_size = in.topology().__local_shape[dim];
    MPI_Sendrecv( &in(1,        1, 1), 1, in.topology().halos[dim], in.topology().nbr_src[dim],   flag,
                  &in(1, 1, n_size-1), 1, in.topology().halos[dim], in.topology().nbr_dest[dim],  flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &in(1, 1, n_size-2), 1, in.topology().halos[dim], in.topology().nbr_dest[dim],  flag,
                  &in(1, 1,        0), 1, in.topology().halos[dim], in.topology().nbr_src[dim],   flag,
                  in.topology().comm_cart, MPI_STATUS_IGNORE);
  }
  
/// @brief 
/// @tparam T 
template <typename T>
  inline void 
  Heat_3D<T>::exchange_ping_pong_I()
  {




  }

/// @brief 
/// @tparam T 
/// @param time 
/// @return 
template <typename T>
  inline T 
  Heat_3D<T>::update_ping_pong(const T time)
  {
    T diff {0.0};
    size_type i {1}, j {1}, k {1};
    for ( i = 1; i < in.topology().__local_shape[0] - 1; ++i )
    {
      for ( j = 1; j < in.topology().__local_shape[1] - 1; ++j )
      {
        for ( k = 1; k < in.topology().__local_shape[2] - 1; ++k )
        {
          T current { in(i,j,k) };
          out(i,j,k) = 
            this->weights[0] * ( in(i-1, j, k) + in(i+1, j, k) )
          + this->weights[1] * ( in(i, j-1, k) + in(i, j+1, k) )
          + this->weights[2] * ( in(i, j, k-1) + in(i, j, k+1) )
          + current * (
              this->diags[0] * this->weights[0]
            + this->diags[1] * this->weights[1]
            + this->diags[2] * this->weights[2]
          );

          diff += std::pow(current - out(i,j,k), 2);
        }
      }
    }

    return diff;
  }

/// @brief 
/// @tparam T 
/// @param time 
/// @return 
template <typename T>
  inline T 
  Heat_3D<T>::update_ping_pong_omp(const T time)
  {
    return 0;
  }


/// @brief 
/// @tparam T 
/// @return 
template <typename T>
  inline T
  Heat_3D<T>::update_ping_pong_bulk( )
  {
    T diff {0.0};
    return diff;
  }

/// @brief 
/// @tparam T 
/// @param time 
/// @return 
template <typename T>
  inline T 
  Heat_3D<T>::update_ping_pong_edge(const T time)
  {

    return 0; 
  }



template <typename T>
  Integer
  Heat_3D<T>::solve_pure_mpi(const T tol, const Integer nsteps, const Integer root)
  {
    T ldiff {0.0}, gdiff {0.0}, time {0.0}; MPI_Datatype DiffType {mpi::get_mpi_type<T>()};
    Integer iter {1};

#ifndef DEBUG
FINAL_PROJECT_MPI_ASSERT((BC_3D != nullptr && IC_3D != nullptr));
FINAL_PROJECT_ASSERT(BC_3D->isSetUpBC && IC_3D->isSetUpInit);
#endif

#ifndef NDEBUG
{
  if (root == in.topology().rank)
  {
    std::cout << 3 << " Dimension Simulation Parameters: " << std::endl;
    std::cout << "\tDepth: "        << in.topology().__local_shape[0]-2 
              << "\n\tRow: "        << in.topology().__local_shape[1]-2 
              << "\n\tColumn: "     << in.topology().__local_shape[2]-2     << std::endl;
    std::cout << "\tTime steps: " << nsteps << std::endl;
    std::cout << "\tTolerance: "  << tol    << std::endl;

    std::cout << "MPI Parameters: "             << std::endl;
    std::cout << "\tNumber of MPI Processes: "  << in.topology().num_procs  << std::endl;
    std::cout << "\tRoot Process: "             << root                     << std::endl;

    std::cout << "Heat Parameters: "    << std::endl;
    std::cout << "\tCoefficient: "      << this->coff       << "\n"
              << "\tTime resolution: "  << this->dt         << "\n"
              << "\tWeights: "          << this->weights[0] << ", " 
                                        << this->weights[1] << ", "
                                        << this->weights[2] << "\n"
              << "\tdxs: "              << this->dxs[0]     << ", " 
                                        << this->dxs[1]     << ", "
                                        << this->dxs[2]     << std::endl;
  }
} 
#endif // end NDEBUG


    Double t0 { MPI_Wtime() };
    for (iter = 1; iter < nsteps; ++iter)
    {
      time = iter*this->dt;

      exchange_ping_pong_SR();
      ldiff = update_ping_pong(time);
      BC_3D->UpdateBC(*this, time);
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

      if (gdiff <= tol) {
        converge = true;
        break;
      }
    }
    Double t1 { MPI_Wtime() - t0 };

    if (converge)
    {
      Double total {0.0};
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










} // namespace pde
} // namespace final_project


#endif // end define FINAL_PROJECT_HEAT_3D_HPP
