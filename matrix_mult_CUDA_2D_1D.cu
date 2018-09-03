//Seung Lee - A01021720
//Matrix Mult con CUDA

#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;

#define NSize 1000; //Definimos el tamano de nuestra matriz N x N

void fillMat(float * ip, const int size) { //Funcion para llenar nuestras matrices (hecho como el ejemplo en clase matrix_sum_1d)
    for(int i = 0; i < size; i++) {
        ip[i] = i;
    }
}


// grid 1D block 1D
__global__ void multMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny)
{
    //Codigo de clase
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    // unsigned int idx = iy * nx + ix;

    float temp = 0;
    if (ix < nx && iy < ny){
        for (int i = 0; i < ny; i++) {
            C[ix*ny+i] += A[ix*ny+i] * B[i*ny+iy];
        }
    }
}

void multMat(float *A, float *B, float *C, const int nx, const int ny) { //Funcion para multiplicar matriz (como ejemplo)
    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < ny; k++) { //Regla del karatazo pu pi pao
                C[i * nx + j] += (A[i * nx + k] * B[k + nx * j]);
                // printf("G"); //Debug
            }
        }
    }
}

//Checar resultado
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N*N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Matrix multiplications from host and GPU match!.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = NSize
    int ny = NSize
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // Inicializar nuestros datos
    fillMat(h_A, nxy);
    fillMat(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMat(h_A, h_B, hostRef, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multMat elapsed %f ms\n", duration_ms.count());

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = 256;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, ny);

    //Multiplicar matrices con cantidad de repeticiones
    int timeAverage = 0;
    for(int i = 0; i < 1; i++) {
        // add matrix at host side for result SAFE_CALLs
        //Lo sacamos del ejemplo de clase
        auto start_cpu =  chrono::high_resolution_clock::now();
        multMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
        SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
        auto end_cpu =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
        timeAverage += duration_ms.count();
    }

    int performanceTime = timeAverage/1;
    printf("La cantidad de tiempo que se tarda cada ejecucion es alrededor de: %d ms\n", performanceTime);
    printf("Cantidad de repeticiones hechas: 1\n");
    printf("Tamano de matriz: %d x %d\n", nx, ny);
    printf("Tamano de bloque en x: %d\n", dimx);

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}