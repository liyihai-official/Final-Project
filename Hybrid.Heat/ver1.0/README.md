```shell
.
├── CMakeLists.txt
├── Doxyfile
├── README.md
├── config.h.in
├── include
│   ├── assert.hpp
│   ├── helper.hpp
│   ├── mpi
│   │   ├── assert.hpp
│   │   ├── environment.hpp
│   │   ├── multiarray.hpp
│   │   ├── topology.hpp
│   │   └── types.hpp
│   ├── multiarray
│   │   ├── base.hpp
│   │   └── types.hpp
│   ├── multiarray.hpp
│   ├── pde
│   │   ├── Heat.hpp
│   │   ├── detials
│   │   │   ├── BoundaryConditions
│   │   │   │   ├── BoundaryConditions_2D.hpp
│   │   │   │   └── BoundaryConditions_3D.hpp
│   │   │   ├── Heat_2D.hpp
│   │   │   ├── Heat_3D.hpp
│   │   │   └── InitializationsC.hpp
│   │   └── pde.hpp
│   ├── pinn
│   │   ├── dataset.hpp
│   │   ├── helper.hpp
│   │   ├── pinn.hpp
│   │   └── types.hpp
│   └── types.hpp
├── main1
│   ├── main1.cpp
│   └── main1_3d.cpp
├── main2
│   ├── datagen.cpp
│   ├── main2.cpp
│   └── pred.cpp
├── out
│   ├── Strong_main1.png
│   ├── main1.png
│   ├── main2.png
│   ├── main2_3d.png
│   ├── main2_dataset.png
│   ├── main3.png
│   ├── model_3d.pt
│   ├── output2d.bin
│   ├── output3d.bin
│   ├── pred_2d.bin
│   ├── pred_3d.bin
│   ├── test.bin
│   └── test_3d.bin
└── src
    ├── boundary.m
    ├── helper.cpp
    ├── loadFromBinary.m
    ├── pinn
    │   ├── dataset.cpp
    │   └── helper.cpp
    ├── saveToBinary.m
    ├── vis2d.m
    ├── vis3d.m
    └── visdataset.m

13 directories, 53 files
```