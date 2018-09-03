//Seung Lee - A01021720
//Matrix Mult CPU (No threads)
//g++ -o MatrixMultOMP matrix_mult_cpu_omp.cpp -std=c++11
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <iomanip> 
#include <string>
#include <iostream>

using namespace std;

#define NSize 4000; //Definimos el tamano de nuestra matriz N x N

void fillMat(float * ip, const int size) { //Funcion para llenar nuestras matrices (hecho como el ejemplo en clase matrix_sum_1d)
    for(int i = 0; i < size; i++) {
        ip[i] = i;
    }
}

void multMat(float *A, float *B, float *C, const int nx, const int ny) { //Funcion para multiplicar matriz (como ejemplo)
    for(int i = 0; i < ny; i++) {
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < ny; k++) { //Regla del karatazo pu pi pao
                C[i * nx + j] += (A[i * nx + k] * B[k *nx + i]);
                // printf("G"); //Debug
            }
        }
    }
}

int main() {
    //Definicion de tamanos y sus valores correspondientes
    int nx, ny, matSize;
    nx = NSize;
    ny = NSize;
    matSize = nx * ny;

    //Utilizar malloc para sacar espacio para matriz
    float *h_A, *h_B, *h_C;
    int nBytes = matSize * sizeof(int);
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);

    // printf("MatrixSize: nx %d, ny %d\n", nx, ny);
    //Llenar las matrices con nuestra funcion
    fillMat(h_A, matSize);
    fillMat(h_B, matSize);

    //Multiplicar matrices con cantidad de repeticiones
    int timeAverage = 0;
    for(int i = 0; i < 10; i++) {
        // add matrix at host side for result SAFE_CALLs
        //Lo sacamos del ejemplo de clase
        auto start_cpu =  chrono::high_resolution_clock::now();
        multMat(h_A, h_B, h_C, nx, ny);
        auto end_cpu =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
        timeAverage += duration_ms.count();
    }

    int performanceTime = timeAverage/10;
    printf("La cantidad de tiempo que se tarda cada ejecucion es alrededor de: %d ms\n", performanceTime);
    printf("Cantidad de repeticiones hechas: 10\n");
    printf("Tamano de matriz: %d x %d\n", nx, ny);

    //Freeamos la memoria
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
