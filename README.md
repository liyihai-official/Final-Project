# Final-Project

<!-- ![Animated GIF](Heat3D/Heat_3D.gif) -->
This is the final project of M.Sc in High Performance Computing program.
Currently, research is focusing on finding numerical solutions of Partial
Differential Equations (PDEs), including heat equations.
<!-- $$
\frac{\partial u}{\partial t} 
(t,\textbf{x})
= 
\kappa\nabla^2_{\textbf{x}}
u(\textbf{x}, t)
, \:\:\:\: \textbf{x} \in \Omega \sub \mathbb{R}^d, d\in \{1, 2, 3\}.
$$

Dirchlet Boundary Condition
$$
u(t,\textbf{x}) = f(\textbf{x}), , \:\:\:\: \textbf{x} \in \bar{\Omega}
$$

Neumann Boundary Condition
$$
\frac{\partial u}{\partial \vec{n}} 
(t,\textbf{x})
= g(\textbf{x}), \:\:\:\: \textbf{x} \in \bar{\Omega}
$$ -->


<!-- <img src="Heat3D/Heat_3D.gif" alt="Animated GIF" height="200"> -->

### Development Note

<img src="MPI.Heat/C++.Heat/ver3.0/visualize/Heat_1D.gif" alt="Animated GIF" height="300">
<img src="MPI.Heat/C++.Heat/ver3.0/visualize/Heat_2D.gif" alt="Animated GIF" height="300">
<img src="MPI.Heat/C++.Heat/ver3.0/visualize/Heat_3D.gif" alt="Animated GIF" height="300">

---
#### Jun, 1-3, 2024
- Working Directory: 
```MPI.Heat/C++Heat/ver3.0```.

- Add 1 dimension version ```main_1d.cc```.
- Optimizations
    - Structure of array, arrays now are isolated from numerical methods' features.
    - New namespace ```heat_equation``` includes objects aiming for solving heat equations in different vector space by various strategies.
    - Gathering, now is capable of collecting boundary values.
    - Sweep, now is capable of updating boundary values while iterating.

- Visualization
    - Add I/O features and visualization programs.

<!-- <img src="MPI.Heat/C++.Heat/ver3.0/Heat_2D.gif" alt="Animated GIF" height="300"> -->

---
#### May, 29-30, 2024
- Working Directory: 
```MPI.Heat/C++Heat/NEW```.

- New Version
    - [x] Complete 3D main function.
    - [x] Add visualization of 3D version.

- A demo of 3D Result
<img src="MPI.Heat/C++.Heat/ver3.0/vis3d.png" alt="Animated GIF" height="500">
<img src="MPI.Heat/C++.Heat/Strong_3d.png" alt="Animated GIF" height="500">

---
#### May, 27-28, 2024
- Working Directory: 
```MPI.Heat/C++Heat/NEW```.
- New Version
    - [ ] Run Strong-Weak Scaling test on multiple nodes.
    - [x] Add basic class for 3d problem.  
    - [x] Add Storing features for 2d and 3d problems which is compatible for ```MATLAB```.

- A demo of Result
<img src="MPI.Heat/C++.Heat/ver3.0/vis.png" alt="Animated GIF" height="300">

---
#### May, 24-26, 2024
- Working Directory: 
```MPI.Heat/C++Heat/NEW```.
- New Version
    - [x] Improve file management.
    - [x] Add Doxygen file and comments.
    - [x] Improve generality of classes design.
    - [x] Prepare 3D problem.
    - [x] Run Strong-Weak Scaling tests on Cluster

- Results of Scaling Tests
<img src="MPI.Heat/C++.Heat/ver3.0/benchmark.results.25.5.first/StrongNew1.png" alt="Animated GIF" height="500">
<img src="MPI.Heat/C++.Heat/ver3.0/benchmark.results.25.5.first/WeakNew1.png" alt="Animated GIF" height="500">
<img src="MPI.Heat/C++.Heat/ver3.0/benchmark.results.25.5.first/WeakNew2.png" alt="Animated GIF" height="500">

---
#### May, 23, 2024
- Working Directory: 
```MPI.Heat/C++Heat```.
- Results of Scaling Tests
Add more tests,
<img src="MPI.Heat/C++.Heat/Strong2.png" alt="Animated GIF" height="500">

---
#### May, 22, 2024
- Working Directory: 
```MPI.Heat/C++Heat```.

- Results of Scaling Tests
<img src="MPI.Heat/C++.Heat/Strong.png" alt="Animated GIF" height="500">
<img src="MPI.Heat/C++.Heat/Weak.png" alt="Animated GIF" height="500">

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





