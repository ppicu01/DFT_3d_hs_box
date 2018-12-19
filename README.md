# 3D Density Functional Theory on Graphics Cards
The program calculates the density profile of hard spheres in a hard box enploying the original Rosenfeld functional. The numerical calculations are performed on a GPU making use of C/C++ and the CUDA development environment. The code accompanies the paper [Massively parallel GPU-accelerated minimization of classical density functional theory](https://aip.scitation.org/doi/full/10.1063/1.4997636), J. Chem. Phys. **147**, 064508 (2017). 
The code should serve as an educational example for those who aim to perform 2D or 3D DFT calculations on graphics cards. 

# Compilation
The program requires having a Nvidia Graphics Card available in your workstation and a working CUDA environment. Compilation should be straight forward on all platforms using the simple makefile. The latter produces an executable file named 'DFT_3d_box'.
Compilation has been tested on Ubuntu 16.04 LTS with CUDA versions 8, 9, 9.1, and 10 and GPUs GeForce GTX1080, Tesla K80, and Tesla V100.   

# Usage
The program takes six mandatory arguments: 1) the fluid packing fraction, 2) Lx, 3) Ly, 4) Lz, 5) precision, 6) full_profile.
We recommend to choose the fluid packing fraction <= 25% as the Rosenfeld functional quickly becomes unreliable in highly confining geometries. 'Lx', 'Ly', 'Lz' are the side lengths of the system in units of the hard-sphere radius R. 'precision' is a boolean with values 0 or 1, determining the precision of the calculations (float or double). 'full_profile' is also a boolean which determines whether the full 3D density should be printed as a binary file (could be viewed e.g. using ImageJ). If full_profile = 0, the density along the z-axis is printed as well as the density contained in the plane for x = 0. 

For example, using the program under Linux or Mac OS with a packing fraction of 20% and a box of size Lx = Ly = Lz = 20R in single-floating point precision reads

`./DFT_3d_box 0.2 20 20 20 0 0`

The program automatically chooses the GPU that has the highest amount of graphics memory available. During minimization, the program prints serveral informations to the console. 

Per default, the number of grid points is N x M x L = 128 x 128 x 128. The spatial resolution along each axis is then specified by dx = Lx/N, dy = Ly/M, and dz = Lz/L. The user can change the number of grid points in the file 'main.cu' by editing the defined integers N, M, L at the beginning of the file. However note that the necessary amount of memory scales with the systems volume. For example, a system of size 256 x 256 x 256 grid points in double precision already occupies roughly 4 Gigabytes of graphics memory. 





