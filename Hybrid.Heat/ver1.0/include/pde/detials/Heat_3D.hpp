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

    Integer solve_pure_mpi(T, Integer=100, Integer=0);
    Integer solve_hybrid_mpi_omp(T, Integer=100, Integer=0);
    Integer solve_hybrid2_mpi_omp(T, Integer=100, Integer=0);

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
    BCFunction FuncDim000, BCFunction FuncDim001,
    BCFunction FuncDim010, BCFunction FuncDim011,
    BCFunction FuncDim100, BCFunction FuncDim101)
  {

  }


template <typename T>
  inline void 
  Heat_3D<T>::SetHeatInitC(InitialConditions::Init_3D<T> & IC)
  {

  }

template <typename T>
  inline void
  Heat_3D<T>::switch_in_out()
  {

  }


template <typename T>
  inline void 
  Heat_3D<T>::exchange_ping_pong_SR()
  {

  }

template <typename T>
  inline void 
  Heat_3D<T>::exchange_ping_pong_I()
  {

  }


template <typename T>
  inline T 
  Heat_3D<T>::update_ping_pong(const T time)
  {
    return 0;
  }


template <typename T>
  inline T 
  Heat_3D<T>::update_ping_pong_omp(const T time)
  {
    return 0;
  }


template <typename T>
  inline T
  Heat_3D<T>::update_ping_pong_bulk( )
  {
    T diff {0.0};
    return diff;
  }


template <typename T>
  inline T 
  Heat_3D<T>::update_ping_pong_edge(const T time)
  {
    
    return 0;
  }

} // namespace pde
} // namespace final_project


#endif // end define FINAL_PROJECT_HEAT_3D_HPP
