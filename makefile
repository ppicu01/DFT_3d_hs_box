main: main.cu
	nvcc -g -G -O3 -lcufft -o DFT_3d_box main.cu 
clean:
	rm DFT_3d_box
