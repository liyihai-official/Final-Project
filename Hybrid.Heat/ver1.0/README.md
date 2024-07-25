```
.
├── CMakeLists.txt
├── README.md
├── include
│   ├── assert.hpp
│   ├── mpi
│   │   ├── asserts.hpp
│   │   ├── environment.hpp
│   │   ├── topology.hpp
│   │   ├── types.hpp
│   │   └── util
│   ├── multiarray
│   │   ├── base.hpp
│   │   ├── types.hpp
│   │   └── util
│   │       ├── base.cpp
│   │       └── types.cpp
│   ├── solver
│   │   ├── boundaryconditions
│   │   │   ├── DirchletBC.hpp
│   │   │   ├── NeumannBC.hpp
│   │   │   └── detials
│   │   │       ├── DirchletBC.cpp
│   │   │       └── NeumannBC.cpp
│   │   ├── detials
│   │   │   ├── evolve_hybrid.hpp
│   │   │   ├── evolve_pure_mpi.hpp
│   │   │   └── mpi_distribute_array.cpp
│   │   ├── evolve.hpp
│   │   ├── gather.hpp
│   │   └── mpi_distribute_array.hpp
│   └── types.hpp
└── test
    └── test.cc

11 directories, 23 files
```