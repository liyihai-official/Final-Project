/**
 * @file heat.cpp
 * 
 * @brief This file contains the classes of namespace for heat equations in 
 *        parallel processing arrays.
 *        
 * 
 * 
 * @author LI Yihai
 * @version 4.1
 * @date Jun 8, 2024
 */
#ifndef FINAL_PROJECT_HEAT_CPP_LIYIHAI
#define FINAL_PROJECT_HEAT_CPP_LIYIHAI

#pragma once
#include "final_project.cpp"
#include <cmath>

namespace final_project {
namespace heat_equation {

typedef _detail::_types::_size_type _size_type;

template <class _T, _size_type _NumDim>
class _heat_pure_mpi {
  public:
  typedef array::array_distribute<_T, _NumDim> _grid_type;
  typedef mpi::env    mpi_env;

  public:
  _grid_type _grid_world;

  public:
  template <typename ... Args>
  _heat_pure_mpi(mpi_env& env, Args ... args);
  virtual void _sweep(_heat_pure_mpi<_T, _NumDim> & out) {}

  public:
    _T _coff {1}, _dt {0.1};
    std::unique_ptr<_T[]> _hx, _min_x, _max_x, _weight, _diag;
};

template <class _T>
  class _heat_2d 
  : public _heat_pure_mpi<_T, 2> 
  {
    public:
    using _heat_pure_mpi<_T, 2>::_heat_pure_mpi;
    void _sweep(_heat_pure_mpi<_T, 2> & out) override;

  };



} // namespace heat_equation
} // namespace final_project

namespace final_project {
namespace heat_equation {

template <class _T, _size_type _NumDim>
template <typename ... Args>
  _heat_pure_mpi<_T, _NumDim>::_heat_pure_mpi(mpi_env& env, Args ... args)
  : _grid_world(env, args...)
  {
    _hx     = std::make_unique<_T[]>(_NumDim);
    _min_x  = std::make_unique<_T[]>(_NumDim);
    _max_x  = std::make_unique<_T[]>(_NumDim);
    _diag   = std::make_unique<_T[]>(_NumDim);
    _weight = std::make_unique<_T[]>(_NumDim);
    

    for (_size_type i = 0; i < _NumDim; ++i)
    {
      _min_x[i] = 0.0;
      _max_x[i] = 1.0;

      _hx[i] = _max_x[i] - _min_x[i];

      auto _temp = std::pow(2, _NumDim) * _hx[i] * _hx[i] / _coff;
      
      _dt = (_dt > _temp) ? _temp : _dt ;
    }

    for (_size_type i = 0; i < _NumDim; ++i)
    {
      _weight[i] = _coff * _dt  / (_hx[i] * _hx[i]);
      _diag[i]   = -2.0 + _hx[i] * _hx[i] / (_NumDim * _coff * _dt);
    }
  }

template <class _T>
  void _heat_2d<_T>::_sweep(_heat_pure_mpi<_T, 2> & out)
  {
    auto shape {out._grid_world.shape()};
    for (_size_type i = 1; i < shape.size(0)-1; ++i)
    {
      for (_size_type j = 1; j < shape.size(1)-1; ++j)
      {
        out._grid_world(i,j) = -1;
      }
    }
  }

  
} // namespace heat_equation
} // namespace final_project






#endif