/**
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * May 25, 2024
 */

#ifndef FINAL_PROJECT_EXCHANGE_HPP_LIYIHAI
#define FINAL_PROJECT_EXCHANGE_HPP_LIYIHAI
#include "array.cpp"
#include <cstring>

namespace final_project
{  
  template <class T>
  void array2d_distribute<T>::SR_exchange2d()
    {
      int flag, scnt = 1;

      
      std::size_t nx {this->rows() - 2};
      std::size_t ny {this->cols() - 2};

      flag = 0;
      MPI_Sendrecv(&(*this)(1    , ny  ), 1, vecs[1], nbr_right, flag,
                   &(*this)(1     , 0   ), 1, vecs[1], nbr_left,  flag, communicator, MPI_STATUS_IGNORE);

      MPI_Sendrecv(&(*this)(1    , 1   ), 1, vecs[1], nbr_left,  flag,
                   &(*this)(1     , ny+1), 1, vecs[1], nbr_right, flag, communicator, MPI_STATUS_IGNORE);


      flag = 1;

      MPI_Sendrecv( &(*this)(1,    1   ), 1, vecs[0], nbr_up,    flag,
                    &(*this)(nx+1, 1   ), 1, vecs[0], nbr_down,  flag, communicator, MPI_STATUS_IGNORE);

      MPI_Sendrecv( &(*this)(nx,   1   ), 1, vecs[0], nbr_down,  flag,
                    &(*this)(0,    1   ), 1, vecs[0], nbr_up,    flag, communicator, MPI_STATUS_IGNORE);
    }

  template <class T>
  void array2d_distribute<T>::I_exchange2d()
    {
      MPI_Request reqs[8];
      int flag, scnt = 1;

      std::size_t nx {this->rows() - 2};
      std::size_t ny {this->cols() - 2};

      flag = 0;

      MPI_Irecv(&(*this)(1, 0), 1, vecs[1], nbr_left, flag, communicator, &reqs[1]);
      MPI_Isend(&(*this)(1, ny), 1, vecs[1], nbr_right, flag, communicator, &reqs[0]);

      MPI_Irecv(&(*this)(1, ny+1), 1, vecs[1], nbr_right, flag, communicator, &reqs[3]);
      MPI_Isend(&(*this)(1, 1), 1, vecs[1], nbr_left, flag, communicator, &reqs[2]);

      flag = 1;
      MPI_Irecv(&(*this)(nx+1, 1), 1, vecs[0], nbr_down, flag, communicator, &reqs[5]);
      MPI_Isend(&(*this)(1, 1), 1, vecs[0], nbr_up, flag, communicator, &reqs[4]);

      MPI_Irecv(&(*this)(0, 1), 1, vecs[0], nbr_up, flag, communicator, &reqs[7]);
      MPI_Isend(&(*this)(nx, 1), 1, vecs[0], nbr_down, flag, communicator, &reqs[6]);

      MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
    }

} // namespace final_project















#endif