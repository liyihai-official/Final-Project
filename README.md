# Final-Project

<!-- ![Animated GIF](Heat3D/Heat_3D.gif) -->

This is the final project of M.Sc in High Performance Computing program.

<img src="Heat3D/Heat_3D.gif" alt="Animated GIF" height="200">

### Development Note
---
#### May, 16, 2024
- Working Directory: 
```MPI.Heat/C++Heat```.

- File Management. 
    - Create ```namespace final_project```.
      ```sweep.cpp```, ```environment.cpp```, ```final_project.cpp```, ```initialization.cpp``` and ```array.cpp```.

    - Files ```array.h```, ```array_mpi.h```, ```lib2d.h``` are planning to deprecate.

- Add New Features
    - [x] Add None Blocking Communications
        - See ```void final_project::Array_Distribute<T>::Iexchange()```.
    - [ ] Add ```benchmark.sh```
        Add benchmark shell script of 'weak scaling' and 'strong scaling' tests. The basic grid size is `16x16`.


---
#### May, 17, 2024
- Working Directory: 
```MPI.Heat/C++Heat```.

- Add New Features
    - [x] Performance Tests
        - Including in```gperftools\profiler.h``` main file。
        - run script ```gperftools.sh```.
    - [x] ```benchmark.sh```
        - It runs on cluster ```callan``` by SLURM successfully.
    - [ ] OpenMP
        - OpenMP has poor performance when it is inited many times.
    - [ ] Report.pdf


<img src="MPI.Heat/C++.Heat/Strong.png" alt="Animated GIF" height="200">
<img src="MPI.Heat/C++.Heat/Weak.png" alt="Animated GIF" height="200">