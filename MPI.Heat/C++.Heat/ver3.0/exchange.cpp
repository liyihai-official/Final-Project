/**
 * @file exchange.cpp
 * @brief This file contains the implementation of the exchange functions 
 *        for exchanging halo regions in a distributed 2D array in an MPI 
 *        environment.
 * 
 * This header file defines the functions SR_exchange2d and I_exchange2d 
 * for the array2d_distribute class, which are used to exchange halo 
 * regions between neighboring MPI processes.
 * 
 * @author LI Yihai
 * @version 3.0
 * @date May 25, 2024
 */

#ifndef FINAL_PROJECT_EXCHANGE_HPP_LIYIHAI
#define FINAL_PROJECT_EXCHANGE_HPP_LIYIHAI
#include "multi_array/array.cpp"
#include <cstring>

namespace final_project
{  

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          Heat 1D
////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Perform synchronous (blocking) halo exchange in a 1D 
   *        distributed array.
   * 
   * This function exchanges the halo regions of a 1D array with 
   * neighboring processes in a synchronous (blocking) manner.
   * 
   * @tparam T The type of the elements in the array.
   */
  template <class T>
  void array1d_distribute<T>::SR_exchange1d()
  {
    int flag, scnt = 1;
    std::size_t N {this->size() - 2};

    flag = 0;
    MPI_Sendrecv( &(*this)(1)   , 1, type, nbr_left,  flag,
                  &(*this)(N+1) , 1, type, nbr_right, flag,
                  communicator, MPI_STATUS_IGNORE);

    MPI_Sendrecv( &(*this)(N)   , 1, type, nbr_right,  flag,
                  &(*this)(0)   , 1, type, nbr_left,   flag,
                  communicator, MPI_STATUS_IGNORE);
  }

  /**
   * @brief Perform asynchronous (non-blocking) halo exchange in a 
   *        1D distributed array.
   * 
   * This function exchanges the halo regions of a 1D array with 
   * neighboring processes in an asynchronous (non-blocking) manner.
   * 
   * @tparam T The type of the elements in the array.
   */
  template <class T>
  void array1d_distribute<T>::I_exchange1d()
  {
      int flag = 0;
      std::size_t N {this->size() - 2};

      MPI_Request reqs[4];

      // Post non-blocking receives
      MPI_Irecv(&(*this)(N+1), 1, type, nbr_right, flag, communicator, &reqs[0]);
      MPI_Irecv(&(*this)(0),   1, type, nbr_left,  flag, communicator, &reqs[1]);

      // Post non-blocking sends
      MPI_Isend(&(*this)(1),   1, type, nbr_left,  flag, communicator, &reqs[2]);
      MPI_Isend(&(*this)(N),   1, type, nbr_right, flag, communicator, &reqs[3]);

      // Wait for all requests to complete
      MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
  }


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          Heat 2D
////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Perform synchronous (blocking) halo exchange in a 2D 
   *        distributed array.
   * 
   * This function exchanges the halo regions of a 2D array with 
   * neighboring processes in a synchronous (blocking) manner.
   * 
   * @tparam T The type of the elements in the array.
   */
  template <class T>
  void array2d_distribute<T>::SR_exchange2d()
    {

      int flag, scnt = 1;
      
      std::size_t nx {this->rows() - 2};
      std::size_t ny {this->cols() - 2};

      flag = 0;

      MPI_Sendrecv( &(*this)(1,    1   ), 1, vecs[0], nbr_up,    flag,
                    &(*this)(nx+1, 1   ), 1, vecs[0], nbr_down,  flag, 
                    communicator, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &(*this)(nx,   1   ), 1, vecs[0], nbr_down,  flag,
                    &(*this)(0,    1   ), 1, vecs[0], nbr_up,    flag, 
                    communicator, MPI_STATUS_IGNORE);

      flag = 1;
      MPI_Sendrecv(&(*this)(1    , ny  ), 1, vecs[1], nbr_right, flag,
                   &(*this)(1     , 0  ), 1, vecs[1], nbr_left,  flag, 
                   communicator, MPI_STATUS_IGNORE);

      MPI_Sendrecv(&(*this)(1    , 1   ), 1, vecs[1], nbr_left,  flag,
                   &(*this)(1    , ny+1), 1, vecs[1], nbr_right, flag, 
                   communicator, MPI_STATUS_IGNORE);

    }

  // template <class T>
  // void array2d_distribute<T>::SR_OMP_exchange2d(const int & omp_pid, const MPI_Datatype vecs_omp[2])
  //   {
  //     int flag, scnt = 1;

  //     std::size_t nx {this->rows() - 2};
  //     std::size_t ny {this->cols() - 2};

  //     flag = 0;
  //     MPI_Sendrecv( &(*this)(1,    1+omp_pid   ), 1, vecs_omp[0], nbr_up,    flag,
  //                   &(*this)(nx+1, 1+omp_pid   ), 1, vecs_omp[0], nbr_down,  flag, 
  //                   communicator, MPI_STATUS_IGNORE);

  //     MPI_Sendrecv( &(*this)(nx,   1+omp_pid   ), 1, vecs_omp[0], nbr_down,  flag,
  //                   &(*this)(0,    1+omp_pid   ), 1, vecs_omp[0], nbr_up,    flag, 
  //                   communicator, MPI_STATUS_IGNORE);

  //     flag = 1;
  //     MPI_Sendrecv(&(*this)(1+omp_pid     , ny ), 1, vecs_omp[1], nbr_right, flag,
  //                  &(*this)(1+omp_pid     , 0  ), 1, vecs_omp[1], nbr_left,  flag, 
  //                  communicator, MPI_STATUS_IGNORE);

  //     MPI_Sendrecv(&(*this)(1+omp_pid    , 1   ), 1, vecs_omp[1], nbr_left,  flag,
  //                  &(*this)(1+omp_pid    , ny+1), 1, vecs_omp[1], nbr_right, flag, 
  //                  communicator, MPI_STATUS_IGNORE);
  //   }

  /**
   * @brief Perform asynchronous (non-blocking) halo exchange in a 
   *        2D distributed array.
   * 
   * This function exchanges the halo regions of a 2D array with 
   * neighboring processes in an asynchronous (non-blocking) manner.
   * 
   * @tparam T The type of the elements in the array.
   */
  template <class T>
  void array2d_distribute<T>::I_exchange2d()
    {
      MPI_Request reqs[8];
      int flag, scnt {1};

      std::size_t nx {this->rows() - 2}, ny {this->cols() - 2};

      flag = 0;
      MPI_Irecv(&(*this)(nx+1, 1), 1, vecs[0], nbr_down, flag, communicator, &reqs[5]);
      MPI_Isend(&(*this)(1,    1), 1, vecs[0], nbr_up,   flag, communicator, &reqs[4]);

      MPI_Irecv(&(*this)(0,  1), 1, vecs[0], nbr_up,   flag, communicator, &reqs[7]);
      MPI_Isend(&(*this)(nx, 1), 1, vecs[0], nbr_down, flag, communicator, &reqs[6]);

      flag = 1;
      MPI_Irecv(&(*this)(1,  0), 1, vecs[1], nbr_left,  flag, communicator, &reqs[1]);
      MPI_Isend(&(*this)(1, ny), 1, vecs[1], nbr_right, flag, communicator, &reqs[0]);

      MPI_Irecv(&(*this)(1, ny+1), 1, vecs[1], nbr_right, flag, communicator, &reqs[3]);
      MPI_Isend(&(*this)(1,    1), 1, vecs[1], nbr_left,  flag, communicator, &reqs[2]);

      MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
    }


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//          Heat 3D
////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Perform synchronous (blocking) halo exchange in a 3D 
   *        distributed array.
   * 
   * This function exchanges the halo regions of a 3D array with 
   * neighboring processes in a synchronous (blocking) manner.
   * 
   * @tparam T The type of the elements in the array.
   */
  template <class T>
  void array3d_distribute<T>::SR_exchange3d()
    {
      int flag;

      std::size_t nx {this->rows() - 2}, ny {this->cols() - 2}, nz{this->height() - 2};

      // Back / Front 
      flag = 0;
      MPI_Sendrecv( &(*this)(1   , 1, 1), 1, vecs[0], nbr_back,  flag, 
                    &(*this)(nx+1, 1, 1), 1, vecs[0], nbr_front, flag,
                    communicator, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &(*this)(nx  , 1, 1), 1, vecs[0], nbr_front, flag, 
                    &(*this)(0   , 1, 1), 1, vecs[0], nbr_back,  flag,
                    communicator, MPI_STATUS_IGNORE);

      // Up / Down
      flag = 1;
      MPI_Sendrecv( &(*this)(1   , 1, 1), 1, vecs[1], nbr_up,    flag, 
                    &(*this)(1, ny+1, 1), 1, vecs[1], nbr_down,  flag,
                    communicator, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &(*this)(1,   ny, 1), 1, vecs[1], nbr_down,  flag, 
                    &(*this)(1,    0, 1), 1, vecs[1], nbr_up,    flag,
                    communicator, MPI_STATUS_IGNORE);
      
      // Left / Right
      flag = 2;
      MPI_Sendrecv( &(*this)(1   , 1, 1), 1, vecs[2], nbr_left,  flag,
                    &(*this)(1, 1, nz+1), 1, vecs[2], nbr_right, flag,
                    communicator, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &(*this)(1, 1, nz  ), 1, vecs[2], nbr_right, flag,
                    &(*this)(1, 1, 0   ), 1, vecs[2], nbr_left,  flag,
                    communicator, MPI_STATUS_IGNORE);

    }

  /**
   * @brief Perform synchronous (none-blocking) halo exchange in a 3D 
   *        distributed array.
   * 
   * This function exchanges the halo regions of a 3D array with 
   * neighboring processes in a synchronous (none-blocking) manner.
   * 
   * @tparam T The type of the elements in the array.
   */
  template <class T>
  void array3d_distribute<T>::I_exchange3d() 
  {
      MPI_Request requests[12];
      int flag, request_count {0};
      
      std::size_t nx {this->rows() - 2}, ny {this->cols() - 2}, nz {this->height() - 2};

      // Back / Front
      flag = 0;
      MPI_Irecv(&(*this)(nx+1, 1, 1), 1, vecs[0], nbr_front, flag, 
                                      communicator, &requests[request_count++]);
      MPI_Isend(&(*this)(1,    1, 1), 1, vecs[0], nbr_back,  flag, 
                                      communicator, &requests[request_count++]);
      MPI_Irecv(&(*this)(0,    1, 1), 1, vecs[0], nbr_back,  flag, 
                                      communicator, &requests[request_count++]);
      MPI_Isend(&(*this)(nx,   1, 1), 1, vecs[0], nbr_front, flag, 
                                      communicator, &requests[request_count++]);

      // Up / Down
      flag = 1;
      MPI_Irecv(&(*this)(1, ny+1, 1), 1, vecs[1], nbr_down, flag, 
                                      communicator, &requests[request_count++]);
      MPI_Isend(&(*this)(1,    1, 1), 1, vecs[1], nbr_up,   flag, 
                                      communicator, &requests[request_count++]);
      MPI_Irecv(&(*this)(1,    0, 1), 1, vecs[1], nbr_up,   flag, 
                                      communicator, &requests[request_count++]);
      MPI_Isend(&(*this)(1,   ny, 1), 1, vecs[1], nbr_down, flag, 
                                      communicator, &requests[request_count++]);

      // Left / Right
      flag = 2;
      MPI_Irecv(&(*this)(1, 1, nz+1), 1, vecs[2], nbr_right, flag, 
                                      communicator, &requests[request_count++]);
      MPI_Isend(&(*this)(1, 1,    1), 1, vecs[2], nbr_left,  flag, 
                                      communicator, &requests[request_count++]);
      MPI_Irecv(&(*this)(1, 1,    0), 1, vecs[2], nbr_left,  flag, 
                                      communicator, &requests[request_count++]);
      MPI_Isend(&(*this)(1, 1,   nz), 1, vecs[2], nbr_right, flag, 
                                      communicator, &requests[request_count++]);

      // Wait All
      MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
  }


} // namespace final_project















#endif