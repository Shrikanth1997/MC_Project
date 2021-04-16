#define USE_NVTOOLS_EXT

#ifdef USE_NVTOOLS_EXT
#include <nvToolsExt.h> 
#endif
#include <cuda_runtime_api.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <typeinfo>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <fstream>
#include <chrono>

#include <MC.h>

enum LogLevels {
    ALWAYS = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5
  };
uint32_t loglevel = 4;

#define CHECK_CUDA do { cudaError_t error = cudaGetLastError(); if(error != cudaSuccess) handleCudaError(error, __FILE__, __LINE__); } while(0)
#define LOG_ERROR(msg, ...) do { if(ERROR <= loglevel) {  fputs("[E] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr);} } while (0)
#define LOG_INFO(msg, ...) do { if(INFO <= loglevel) {  fputs("[I] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr);} } while (0)
#define CHECKED_CUDA(a) do { cudaError_t error = (a); if(error != cudaSuccess) handleCudaError(error, __FILE__, __LINE__); } while(0)
#define LOG_ALWAYS(msg, ...) do { fputs("[A] ", stderr); fprintf(stderr, msg, ##__VA_ARGS__); fputc('\n', stderr); } while (0)

[[noreturn]]
  void handleCudaError(cudaError_t error, const std::string file, int line)
  {
    LOG_ERROR("%s@%d: CUDA: %s", file.c_str(), line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }


uint3 field_size = make_uint3(60, 60, 60);

std::vector<char> scalarField_host;
std::vector<char> scalarField_blurred;

std::vector<float> blurred;
std::vector<float> blurredBinaryMask;

void readFile();
bool readDATFile(const char* path);


void createCharVector(){
	
	auto* bytes = reinterpret_cast<char*>(&blurredBinaryMask[0]);
	std::vector<char> byteVec(bytes, bytes + sizeof(float) * (blurred.size()-1));
	scalarField_blurred = byteVec;

}

// This function takes a linearized matrix in the form of a vector and
// calculates elements according to the 3D Gaussian distribution
void generateGaussian(std::vector<float>& K, int dim, int radius) {
	double stdev = 1.0;
	double pi = 355.0 / 113.0;
	double constant = 1.0 / (pow((2.0 * pi),1.5) * pow(stdev, 3));

	for (int i = -radius; i < radius + 1; ++i)
		for (int j = -radius; j < radius + 1; ++j)
			for (int k = -radius; k < radius + 1; ++k)
				K[((k+radius)*dim*dim) + ((i + radius) * dim) + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2) + pow(k, 2)) / (2 * pow(stdev, 2))));
}


#define inBounds(x, y, z) \
  ((0 <= (x) && (x) < x_size) && \
   (0 <= (y) && (y) < y_size) && \
   (0 <= (z) && (z) < z_size))

//*** Program-wide constants ***//
#define KERNEL_SIZE   7
#define KERNEL_RADIUS 3 

#define TILE_SIZE     KERNEL_SIZE
#define CACHE_SIZE    (KERNEL_SIZE + (KERNEL_RADIUS * 2))

//*** Device constant memory ***//
__constant__ float deviceKernel[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE];

//*** 3D convolution kernel ***//
__global__ void conv3d(float *input, float *output,
                        const int z_size, const int y_size, const int x_size) {

  // General Thread Info
  int bx = blockIdx.x * TILE_SIZE; int tx = threadIdx.x;
  int by = blockIdx.y * TILE_SIZE; int ty = threadIdx.y;
  int bz = blockIdx.z * TILE_SIZE; int tz = threadIdx.z;

  //*** Generate tileCache ***//
  __shared__ float tileCache[CACHE_SIZE][CACHE_SIZE][CACHE_SIZE];

  // map each thread to a position in the kernel
  int tid = tz * (KERNEL_SIZE * KERNEL_SIZE) + ty * (KERNEL_SIZE) + tx;
  if (tid < CACHE_SIZE * CACHE_SIZE) {

    // map each kernel position to location in tile cache
    int tileX =  tid % CACHE_SIZE;
    int tileY = (tid / CACHE_SIZE) % CACHE_SIZE;

    int inputX = bx + tileX - 1;
    int inputY = by + tileY - 1;
    int inputZPartial = bz - 1;
    int inputZ;

    // load part of the tile cache
    for (int i = 0; i < CACHE_SIZE; i += 1) {
      inputZ = inputZPartial + i;

      if (inBounds(inputX, inputY, inputZ)) {
        tileCache[tileX][tileY][i] = input[inputZ * (y_size * x_size) + inputY * (x_size) + inputX];
      } else {
        tileCache[tileX][tileY][i] = 0;
      }
    }
  }

  __syncthreads();

  //*** Perform block convolution ***//
  // Exit threads outside of matrix boundry
  int xPos = bx + tx;
  int yPos = by + ty;
  int zPos = bz + tz;

  if (inBounds(xPos, yPos, zPos)) {
    float outputValue = 0;
    for (int x = 0; x < KERNEL_SIZE; x += 1) {
      for (int y = 0; y < KERNEL_SIZE; y += 1) {
        for (int z = 0; z < KERNEL_SIZE; z += 1) {
            outputValue +=
              tileCache[tx + x][ty + y][tz + z] *
              deviceKernel[z * (KERNEL_SIZE * KERNEL_SIZE) + y * (KERNEL_SIZE) + x];
        }
      }
    }
    output[zPos * (y_size * x_size) + yPos * (x_size) + xPos] = outputValue;
  }
}


void performGaussian(std::vector<float>& blurred){
	std::vector<float> hKernel;
	
	int z_size;
	int y_size;
	int x_size;
	int kDim, kRadius;
	
	float *deviceInput;
	float *deviceOutput;
	
	kDim = KERNEL_SIZE; // Kernel is square and odd in dimension, should be variable at some point
	kRadius = floor(kDim / 2.0); // Radius of odd kernel doesn't consider middle index
	hKernel.resize(pow(kDim, 3), 0);
	generateGaussian(hKernel, kDim, kRadius);
	
	x_size = field_size.x;
	y_size = field_size.y;
	z_size = field_size.z;
	
	cudaMalloc((void**) &deviceInput,  z_size * y_size * x_size * sizeof(float));
    cudaMalloc((void**) &deviceOutput, z_size * y_size * x_size * sizeof(float));
	
	cudaMemcpy(deviceInput, blurred.data(),  z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceKernel, hKernel.data(), hKernel.size() * sizeof(float) , 0, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(x_size/double(TILE_SIZE)), ceil(y_size/double(TILE_SIZE)), ceil(z_size/double(TILE_SIZE)));
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    conv3d<<<dimGrid, dimBlock>>>(
      deviceInput, deviceOutput,
      z_size, y_size, x_size
    );
    cudaDeviceSynchronize();
	
	cudaMemcpy(blurred.data(), deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);

	float max=0;
	for (auto& value : blurred)
		max = (value > max) ? value : max;
	for (auto& value : blurred)
		value = (value * 255) / max;

	
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
}


void setupScalarField(float*& scalar_field_d, const uint3& field_size, cudaStream_t stream)
{
	
	performGaussian(blurredBinaryMask);
	
	createCharVector();

	std::cout<<"DAT file size: "<<scalarField_host.size()<<'\n';

    LOG_INFO("Scalar field is [%d x %d x %d] (%d cells total)", field_size.x, field_size.y, field_size.z, field_size.x * field_size.y * field_size.z);

	CHECKED_CUDA(cudaMalloc(&scalar_field_d, scalarField_host.size()));
	CHECKED_CUDA(cudaMemcpyAsync(scalar_field_d, scalarField_blurred.data(), scalarField_host.size(), cudaMemcpyHostToDevice, stream));
}

bool readDATFile(const char* path)
  {
    assert(path);
    LOG_INFO("Reading %s...", path);

    FILE* fp = fopen(path, "rb");
    if (!fp) {
      LOG_ERROR("Error opening file \"%s\" for reading.", path);
      return false;
    }
    if (fseek(fp, 0L, SEEK_END) == 0) {
      uint8_t header[6];
      long size = ftell(fp);
      if (sizeof(header) <= size) {
        if (fseek(fp, 0L, SEEK_SET) == 0) {
          if (fread(header, sizeof(header), 1, fp) == 1) {
            field_size.x = header[0] | header[1] << 8;
            field_size.y = header[2] | header[3] << 8;
            field_size.z = header[4] | header[5] << 8;
            size_t N = static_cast<size_t>(field_size.x) * field_size.y * field_size.z;
            if ((N + 3) * 2 != size) {
              LOG_ERROR("Unexpected file size.");
            }
            else {
              std::vector<uint8_t> tmp(2 * N);
              if (fread(tmp.data(), 2, N, fp) == N) {
                
                  scalarField_host.resize(sizeof(float) * N);
                  auto* dst = reinterpret_cast<float*>(scalarField_host.data());
                  for (size_t i = 0; i < N; i++) {
                    const uint32_t v = tmp[2 * i + 0] | tmp[2 * i + 1] << 8;
                    dst[i] = static_cast<float>(v);
                  }
                  
                LOG_INFO("Successfully loaded %s", path);
                fclose(fp);
                return true;
              }
            }
          }
        }
      }
    }
    LOG_ERROR("Error loading \"%s\"", path);
    fclose(fp);
    return false;
  }




void readFile(std::string fileName){
	std::ifstream file;
	file.open(fileName);

	size_t N = static_cast<size_t>(field_size.x) * field_size.y * field_size.z;
	scalarField_host.resize(sizeof(float)*N);
    auto* dst = reinterpret_cast<float*>(scalarField_host.data());

	double val=0;
	long count=0;
	while(!file.eof()){
		file >> val;
		dst[count] = static_cast<float>(val);
		blurred.push_back(static_cast<float>(val));
		count++;
	}

	std::cout<<"SIZE: "<<scalarField_host.size()<<" ACTUAL: "<<count<<'\n';
	

	file.close();
	
}


void writeFile(float* vertex, int vertexCount, uint32_t* index, int indexCount, std::string fileName){
	std::ofstream fileV, fileI;
	fileV.open("../data/vertices" + fileName + ".dat");
	fileI.open("../data/indices" + fileName + ".dat");


	for(int i=0;i<vertexCount;i++)
		fileV << vertex[i] << std::endl;

	for(int i=0;i<indexCount;i++)
		fileI << index[i] << std::endl;


	fileV.close();
	fileI.close();
}


__global__ void createBinaryMask(float *input, float *output, const int label, const int z_size, const int y_size, const int x_size) {

  // General Thread Info
  int bx = blockIdx.x * blockDim.x; int tx = threadIdx.x;
  int by = blockIdx.y * blockDim.y; int ty = threadIdx.y;
  int bz = blockIdx.z * blockDim.z; int tz = threadIdx.z;

  int xPos = bx + tx;
  int yPos = by + ty;
  int zPos = bz + tz;

	float ones = 1.000;
	float zero = 0.000;

  if ((xPos < (z_size)) && (yPos < (y_size)) && (zPos < (z_size))) {
  	if(input[zPos * (y_size * x_size) + yPos * (x_size) + xPos] == label)
    	output[zPos * (y_size * x_size) + yPos * (x_size) + xPos] = ones;
	else
		output[zPos * (y_size * x_size) + yPos * (x_size) + xPos] = zero;
  }
  __syncthreads();

}

void getBinaryMask(std::vector<float>& blurred, float label){

	int x_size = field_size.x;
	int y_size = field_size.y;
	int z_size = field_size.z;

	float *labeledInput;
	float *maskedOutput;

	cudaMalloc((void**) &labeledInput,  z_size * y_size * x_size * sizeof(float));
    cudaMalloc((void**) &maskedOutput, z_size * y_size * x_size * sizeof(float));


	cudaMemcpy(labeledInput, blurred.data(),  z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);

	//dim3 dimGrid(ceil(x_size/double(TILE_SIZE)), ceil(y_size/double(TILE_SIZE)), ceil(z_size/double(TILE_SIZE)));
    //dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);

	dim3 dimGrid(x_size, y_size, z_size);
    dim3 dimBlock(1, 1, 1);
    createBinaryMask<<<dimGrid, dimBlock>>>(labeledInput, maskedOutput, label , z_size, y_size, x_size);
    cudaDeviceSynchronize();
	
	cudaMemcpy(blurred.data(), maskedOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(labeledInput);
	cudaFree(maskedOutput);
}




int main(int argc, char** argv)
{


	cudaStream_t stream;

	// Setting up CUDA device
	int deviceCount = 0;
	int deviceIndex = 0;
    CHECKED_CUDA(cudaGetDeviceCount(&deviceCount));

    bool found = false;
    for (int i = 0; i < deviceCount; i++) {
      cudaDeviceProp dev_prop;
      cudaGetDeviceProperties(&dev_prop, i);
      LOG_INFO("%c[%i] %s cap=%d.%d", i == deviceIndex ? '*' : ' ', i, dev_prop.name, dev_prop.major, dev_prop.minor);
      if (i == deviceIndex) {
        found = true;
      }
    }
    if (!found) {
      LOG_ERROR("Illegal CUDA device index %d", deviceIndex);
      return EXIT_FAILURE;
    }
    cudaSetDevice(deviceIndex);
    CHECKED_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	

	float threshold = 0.5f;

	// Create events for timing
    static const unsigned eventNum = 32;
    cudaEvent_t events[2 * eventNum];
    for (size_t i = 0; i < 2 * eventNum; i++) {
      CHECKED_CUDA(cudaEventCreate(&events[i]));
      CHECKED_CUDA(cudaEventRecord(events[i], stream));
    }

    size_t free, total;
    CHECKED_CUDA(cudaMemGetInfo(&free, &total));
    LOG_INFO("CUDA memory free=%zumb total=%zumb", (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));


	// Read the input file	
	readFile("../data/labelsAll.dat");

	// Get all labels inside the input file
	std::set<float, std::less<float>> labels(blurred.begin(), blurred.end());

// Loop across the labels
for(auto it = labels.begin();it!=labels.end();it++){
	std::cout<<"For Labels: "<<*it<<" ----------"<<'\n';


	float label=*it;
	// Binary mask the input based on the label
	std::vector<float> temp(blurred);
	getBinaryMask(temp, label);
	blurredBinaryMask = temp;

	//blurredBinaryMask = blurred;

	// Set up scalar field
	float* scalar_field_d = nullptr;
  	setupScalarField(scalar_field_d, field_size, stream);

	
	// Checking if data is present in device
	/*float* check_data=(float*)malloc(scalarField_host.size());	
	cudaError_t e = cudaMemcpyAsync(check_data, scalar_field_d, scalarField_host.size(), cudaMemcpyDeviceToHost, stream);
	std::cout<<"CHECK DATA memcpy error: "<<cudaGetErrorString(e)<<'\n';
	std::ofstream fileC;
	fileC.open("check.dat");
	for(long int i=0;i<scalarField_host.size()/4;i++)
		fileC << check_data[i] << std::endl;
	fileC.close();*/
	LOG_INFO("Built scalar field");

    CHECKED_CUDA(cudaMemGetInfo(&free, &total));
    LOG_INFO("CUDA memory free=%zumb total=%zumb", (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));



	auto* tables = ComputeStuff::MC::createTables(stream);

	float* vertexData = nullptr;
	int vertexCount=0;

	uint32_t* indexData = nullptr;
	int indexCount=0;

struct {
      const char* name;
      bool indexed;
      bool sync;
    }
    benchmark_cases[] = {
      {"ix sync", true, true}
    };

    for (auto& bc : benchmark_cases) {
#ifdef USE_NVTOOLS_EXT
      nvtxRangePush(bc.name);
#endif
      auto* ctx = createContext(tables, field_size, true, stream);
      LOG_INFO("%12s: Created context.", bc.name);
      LOG_INFO("Grid size [%u x %u x %u]", ctx->grid_size.x, ctx->grid_size.y, ctx->grid_size.z);
      LOG_INFO("Chunks [%u x %u x %u] (= %u) cover=[%u x %u x %u]",
              ctx->chunks.x, ctx->chunks.y, ctx->chunks.z, ctx->chunk_total,
              31 * ctx->chunks.x, 5 * ctx->chunks.y, 5 * ctx->chunks.z);
      LOG_INFO("Level vec4-offset  vec4-size  (    size)");
      for (unsigned l = 0; l < ctx->levels; l++) {
        LOG_INFO("[%2d] %12d %10d  (%8d)", l, ctx->level_offsets[l], ctx->level_sizes[l], 4 * ctx->level_sizes[l]);
      }
      LOG_INFO("Total %d, levels %d", ctx->total_size, ctx->levels);

      // Run with no output buffers to get size of output.
      ComputeStuff::MC::buildPN(ctx,
                                nullptr,
                                nullptr,
                                0,
                                0,
                                field_size.x,
                                field_size.x* field_size.y,
                                make_uint3(0, 0, 0),
                                field_size,
                                scalar_field_d,
                                threshold,
                                stream,
                                true,
                                true);
      uint32_t vertex_count = 0;
      uint32_t index_count = 0;
      ComputeStuff::MC::getCounts(ctx, &vertex_count, &index_count, stream);
	std::cout<<"index count: "<<index_count<<'\n';
	   	  std::cout<<"vertex count: "<<vertex_count<<'\n';

	 float* vertexDataH = (float*)calloc(0,sizeof(float)*6 * vertex_count);
	 uint32_t* indexDataH = (uint32_t*)calloc(0,sizeof(float)* index_count);
      
	  float* vertex_data_d = nullptr;
      CHECKED_CUDA(cudaMalloc(&vertex_data_d, 6 * sizeof(float) * vertex_count));
	  //CHECKED_CUDA(cudaMemcpy(vertex_data_d, vertexDataH, sizeof(float)*6 * vertex_count, cudaMemcpyHostToDevice));
      
	  uint32_t* index_data_d = nullptr;
      CHECKED_CUDA(cudaMalloc(&index_data_d, sizeof(uint32_t)* index_count));
	  //CHECKED_CUDA(cudaMemcpy(index_data_d, indexDataH, sizeof(uint32_t)* index_count, cudaMemcpyHostToDevice));
      
	  LOG_INFO("%12s: Allocated output buffers.", bc.name);

      LOG_INFO("%12s: Warming up", bc.name);
        ComputeStuff::MC::buildPN(ctx,
                                  vertex_data_d,
                                  index_data_d,
                                  6 * sizeof(float) * vertex_count,
                                  sizeof(uint32_t) * index_count,
                                  field_size.x,
                                  field_size.x * field_size.y,
                                  make_uint3(0, 0, 0),
                                  field_size,
                                  scalar_field_d,
                                  threshold,
                                  stream,
                                  true,
                                  true);
        if (bc.sync) {
          ComputeStuff::MC::getCounts(ctx, &vertex_count, &index_count, stream);
		  std::cout<<"index count: "<<index_count<<'\n';
	   	  std::cout<<"vertex count: "<<vertex_count<<'\n';

        }

      LOG_INFO("%12s: Benchmarking", bc.name);
      auto start = std::chrono::high_resolution_clock::now();
      double elapsed = 0.f;
      float cuda_ms = 0.f;
      unsigned iterations = 0;
      unsigned cuda_ms_n = 0;
      
	  CHECKED_CUDA(cudaMemGetInfo(&free, &total));
      LOG_ALWAYS("%12s: %.2f FPS (%.0fMVPS) cuda: %.2fms (%.0f MVPS) %ux%ux%u Nv=%u Ni=%u memfree=%zumb/%zumb",
              bc.name,
              iterations / elapsed, (float(iterations) * field_size.x * field_size.y * field_size.z) / (1000000.f * elapsed),
              cuda_ms / cuda_ms_n, (float(cuda_ms_n) * field_size.x * field_size.y * field_size.z) / (1000.f * cuda_ms),
              field_size.x, field_size.y, field_size.z,
              vertex_count,
              index_count,
              (free + 1024 * 1024 - 1) / (1024 * 1024),
              (total + 1024 * 1024 - 1) / (1024 * 1024));


	 vertexCount = 6*vertex_count;
	 vertexData = (float*)malloc(sizeof(float) * vertexCount);
	 cudaError_t e = cudaMemcpy((void*)vertexData, vertex_data_d, sizeof(float) * vertexCount, cudaMemcpyDeviceToHost);
	 std::cout<<"Vertex data memcpy error: "<<cudaGetErrorString(e)<<'\n';


	indexData = (uint32_t*)malloc(sizeof(uint32_t)*index_count);
	indexCount = index_count;
	 e = cudaMemcpy((void*)indexData, index_data_d, sizeof(uint32_t)*indexCount, cudaMemcpyDeviceToHost);
	 std::cout<<"Index Data memcpy error: "<<cudaGetErrorString(e)<<'\n';


	  freeContext(ctx, stream);
      CHECKED_CUDA(cudaStreamSynchronize(stream));

	  CHECKED_CUDA(cudaFree(vertex_data_d));
      CHECKED_CUDA(cudaFree(index_data_d));

      CHECKED_CUDA(cudaMemGetInfo(&free, &total));
      LOG_INFO("%12s: Released resources free=%zumb total=%zumb", bc.name, (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));
#ifdef USE_NVTOOLS_EXT
      nvtxRangePop();
#endif
    }

    LOG_ALWAYS("Exiting...");
    CHECKED_CUDA(cudaMemGetInfo(&free, &total));
    LOG_INFO("CUDA memory free=%zumb total=%zumb", (free + 1024 * 1024 - 1) / (1024 * 1024), (total + 1024 * 1024 - 1) / (1024 * 1024));
    

	std::cout<<"index count: "<<indexCount<<'\n';
	std::cout<<"vertex count: "<<vertexCount<<'\n';

	LOG_ALWAYS("Writing into files....");
	writeFile(vertexData, vertexCount, indexData, indexCount, "labels_" + std::to_string(label));

	std::cout<<"---------------------------------------------------\n";

	}

	return 0;

}
