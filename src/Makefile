all:
	nvcc -rdc=true -gencode=arch=compute_35,code=\"sm_35,compute_35\" -I"pub/cuda-10.2/include/" -I"../include/libs/glfw/deps/" -I"../include/libs/glfw/include/" -I"../include/MarchingCubes/" -I"../include/libs/glfw/src/"  ../include/MarchingCubes/*.cu ../include/MarchingCubes/*.cpp ../include/libs/glfw/deps/*.c ../include/libs/glfw/build/src/libglfw3.a -lnvToolsExt test.cu -o ts

