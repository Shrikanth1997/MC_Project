all:
	nvcc -rdc=true -gencode=arch=compute_35,code=\"sm_35,compute_35\" -I"pub/cuda-10.2/include/" -I"../libs/glfw/deps/" -I"../libs/glfw/include/" -I"../MarchingCubes/" -I"../libs/glfw/src/"  ../MarchingCubes/*.cu ../MarchingCubes/*.cpp ../libs/glfw/deps/*.c ../libs/glfw/build/src/libglfw3.a -lnvToolsExt test.cu -o ts

