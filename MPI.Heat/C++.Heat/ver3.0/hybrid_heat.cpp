
// #include "multi_array/array_distribute.cpp"

// namespace final_project {
// namespace heat_equation {
//   template <class T>
//   class heat1d_hybrid 
//   {
//   private:
//   double coff {1};
//   double diag, weight;
//   double dt, hx;
//   double min_x {0}, max_x {1};

//     public:
//     array1d_distribute<T> body;


//     public:
//     heat1d_hybrid(std::size_t gN, const int dims[1], MPI_Comm comm)
//     {
//       body.distribute(gN, dims, comm);

//       hx = (double) (max_x - min_x) / (double) (gN + 1);
//       dt = 0.5 * hx * hx / coff;

//       weight = coff * dt / ( hx * hx );
//       diag = -2.0 + hx * hd / ( body.dimension * coff * dt );
//     }

//   }; // class heat1d_hybrid
// } // namespace heat_equation
// } // namespace final_project
