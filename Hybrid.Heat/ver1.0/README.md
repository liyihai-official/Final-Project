```
.
├── CMakeLists.txt
├── README.md
├── include
│   ├── assert.hpp
│   ├── mpi
│   │   ├── assert.hpp
│   │   ├── environment.hpp
│   │   ├── topology.hpp
│   │   └── types.hpp
│   ├── multiarray
│   │   ├── base.hpp
│   │   └── types.hpp
│   ├── solver
│   │   ├── boundaryconditions
│   │   │   ├── DirchletBC.hpp
│   │   │   └── NeumannBC.hpp
│   │   ├── detials
│   │   │   ├── evolve_hybrid.hpp
│   │   │   └── evolve_pure_mpi.hpp
│   │   ├── evolve.hpp
│   │   ├── gather.hpp
│   │   └── mpi_distribute_array.hpp
│   └── types.hpp
├── src
│   ├── assert.cpp
│   ├── mpi
│   │   ├── environment.cpp
│   │   └── types.cpp
│   ├── multiarray
│   │   └── types.cpp
│   ├── solver
│   └── types.cpp
└── test
    └── test.cc

12 directories, 23 files
```