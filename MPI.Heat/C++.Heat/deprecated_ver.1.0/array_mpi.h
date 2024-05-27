#pragma once 
#include <mpi.h>
#include "array.h"

template<typename T>
MPI_Datatype get_mpi_type();

template<>
MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template<>
MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }

template<>
MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }

template <typename T>
class Array_Distribute : public Array<T> {
  public:
  Array_Distribute() = delete;

  Array_Distribute(
    std::size_t const rows, std::size_t const cols, 
    int const dims[2], MPI_Comm comm_cart)
  : Array<T>( get_loc_dim(rows - 2, dims, 0, comm_cart), 
              get_loc_dim(cols - 2, dims, 1, comm_cart)),
    nx_glob {static_cast<int>(rows)}, 
    ny_glob {static_cast<int>(cols)}, 
    comm {comm_cart}
  {

    MPI_Cart_shift(comm_cart, 0, 1, &nbr_up,   &nbr_down );
    MPI_Cart_shift(comm_cart, 1, 1, &nbr_left, &nbr_right);

    /* Setup vector types of halo */ 
    MPI_Type_contiguous(ends[1] - starts[1] + 1,         get_mpi_type<T>(), &vecs[0]);
    MPI_Type_commit(&vecs[0]);

    int ny = ends[1] - starts[1] + 1 + 2;
    MPI_Type_vector(    ends[0] - starts[0] + 1, 1, ny , get_mpi_type<T>(), &vecs[1]);
    MPI_Type_commit(&vecs[1]);

  }

  void sweep(Array_Distribute<T>& out, Array_Distribute<T> const bias);   /* Possion Equation */
  void sweep(Array_Distribute<T>& out);                                      /* Heat Equation */

  void Iexchange();
  void SRexchange();

  void Array_Gather(Array<T>& gather, const int root);


  int get_start(const int idx) {
    if (idx == 0 || idx == 1) 
      return starts[idx];
    else 
    {
      std::string msg {"Array subscript out of index.\n"};
      throw std::out_of_range(msg);
    }
  }

  int get_end(const int idx) {
    if (idx == 0 || idx == 1) 
      return ends[idx];
    else 
    {
      std::string msg {"Array subscript out of index.\n"};
      throw std::out_of_range(msg);
    }
  }

  int get_glob_num_rows() {return nx_glob; }
  int get_glob_num_cols() {return ny_glob; }

  protected:
  constexpr static int dimension {2};

  private:
  int rank, num_proc;
  int nx_glob, ny_glob;
  int nbr_up, nbr_down, nbr_right, nbr_left;

  int starts[dimension], ends[dimension], coordinates[dimension];

  MPI_Comm comm;

  MPI_Datatype vecs[dimension];

  std::size_t get_loc_dim(auto glob_dim, const int dims[], const int coord_idx, MPI_Comm comm)
  {
    int start, end;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_proc);
    MPI_Cart_coords(comm, rank, dimension, coordinates);

    Decomp1d(glob_dim, dims[coord_idx], coordinates[coord_idx], starts[coord_idx], ends[coord_idx]);
    
    return ends[coord_idx] - starts[coord_idx] + 1 + 2;
  };

};

template <typename T>
void Array_Distribute<T>::sweep(Array_Distribute<T>& out)
{

  int i, j;

  int nx = this->get_num_rows() - 2;
  int ny = this->get_num_cols() - 2;

  double coff = 1;
  double dt, hx, hy;
  double diag_x, diag_y;
  double weight_x, weight_y;

  auto min = [](auto const a, auto const b){return (a <= b) ? a : b;};

  hx = (double) 1 / (double) (nx_glob+1);
  hy = (double) 1 / (double) (ny_glob+1);

  dt = 0.25 * (double) (min(hx, hy) * min(hx, hy)) / coff;
  dt = min(dt, 0.1);

  weight_x = coff * dt / (hx * hx);
  weight_y = coff * dt / (hy * hy);

  diag_x = -2.0 + hx * hx / (2 * coff * dt);
  diag_y = -2.0 + hy * hy / (2 * coff * dt);

  for (i = 1; i <= nx; ++i)
    for (j = 1; j <= ny; ++j)
    {
      out(i,j) = weight_x * ((*this)(i-1, j) + (*this)(i+1, j) + (*this)(i,j) * diag_x)
               + weight_y * ((*this)(i, j-1) + (*this)(i, j+1) + (*this)(i,j) * diag_y);
    }
    
}

template <typename T>
void Array_Distribute<T>::Iexchange()
{
    MPI_Request reqs[8];
    int flag, scnt = 1;

    int nx = this->get_num_rows() - 2;
    int ny = this->get_num_cols() - 2;

    flag = 0;

    MPI_Irecv(&(*this)(1, 0), 1, vecs[1], nbr_left, flag, comm, &reqs[1]);
    MPI_Isend(&(*this)(1, ny), 1, vecs[1], nbr_right, flag, comm, &reqs[0]);
    
    MPI_Irecv(&(*this)(1, ny+1), 1, vecs[1], nbr_right, flag, comm, &reqs[3]);
    MPI_Isend(&(*this)(1, 1), 1, vecs[1], nbr_left, flag, comm, &reqs[2]);

    flag = 1;
    MPI_Irecv(&(*this)(nx+1, 1), 1, vecs[0], nbr_down, flag, comm, &reqs[5]);
    MPI_Isend(&(*this)(1, 1), 1, vecs[0], nbr_up, flag, comm, &reqs[4]);

    MPI_Irecv(&(*this)(0, 1), 1, vecs[0], nbr_up, flag, comm, &reqs[7]);
    MPI_Isend(&(*this)(nx, 1), 1, vecs[0], nbr_down, flag, comm, &reqs[6]);
    
    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
}

template <typename T>
void Array_Distribute<T>::SRexchange()
{
  int flag, scnt = 1;

  int nx = this->get_num_rows() - 2;
  int ny = this->get_num_cols() - 2;

  flag = 0;
  MPI_Sendrecv(&(*this)(1    , ny  ), 1, vecs[1], nbr_right, flag,
               &(*this)(1    , 0   ), 1, vecs[1], nbr_left,  flag, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&(*this)(1    , 1   ), 1, vecs[1], nbr_left,  flag,
               &(*this)(1    , ny+1), 1, vecs[1], nbr_right, flag, comm, MPI_STATUS_IGNORE);


  flag = 1;

  MPI_Sendrecv( &(*this)(1,    1   ), 1, vecs[0], nbr_up,    flag,
                &(*this)(nx+1, 1   ), 1, vecs[0], nbr_down,  flag, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv( &(*this)(nx,   1   ), 1, vecs[0], nbr_down,  flag,
                &(*this)(0,    1   ), 1, vecs[0], nbr_up,    flag, comm, MPI_STATUS_IGNORE);
}

template <typename T>
void Array_Distribute<T>::Array_Gather(Array<T>& gather, const int root)
{
  MPI_Datatype temp, Block, mpi_T {get_mpi_type<T>()};

  int nx = this->get_num_rows() - 2;
  int ny = this->get_num_cols() - 2;

  int size, pid, i;
  MPI_Comm_size(comm, &size);
  int s0_list[size], s1_list[size], nx_list[size], ny_list[size];

  MPI_Type_vector(nx, ny, ny+2, mpi_T, &Block);
  MPI_Type_commit(&Block);

  MPI_Gather(&starts[0], 1, MPI_INT, s0_list, 1, MPI_INT, root, comm);
  MPI_Gather(&starts[1], 1, MPI_INT, s1_list, 1, MPI_INT, root, comm);
  MPI_Gather(&nx       , 1, MPI_INT, nx_list, 1, MPI_INT, root, comm);
  MPI_Gather(&ny       , 1, MPI_INT, ny_list, 1, MPI_INT, root, comm);


  if (rank != root)
  {
    MPI_Send(&(*this)(1,1), 1, Block, root, rank, comm);
  }
  MPI_Type_free(&Block);

  if (rank == root)
  {
    for (pid = 0; pid < size; ++pid)
    {
      if (pid == root)
      {
        for (i = starts[0]; i <= ends[0]; ++i)
        {
          memcpy(&gather(i, starts[1]), &(*this)(i, starts[1]), ny_list[pid]*sizeof(T));
        }
      }

      if (pid != root)
      {
        MPI_Type_vector(nx_list[pid], ny_list[pid], gather.get_num_cols(), mpi_T, &temp);
        MPI_Type_commit(&temp);

        MPI_Recv(&gather(s0_list[pid], s1_list[pid]), 1,
                          temp, pid, pid, comm, MPI_STATUS_IGNORE);

        MPI_Type_free(&temp);  
      }
    }
  }
}
