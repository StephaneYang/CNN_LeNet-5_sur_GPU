#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void cuda_hello(){
    printf("Hello World\n");
}

void MatrixInit(float *M, int n, int p){
    for(int i=0;i<n*p;i++){
        M[i] = (float)rand()/(float)(RAND_MAX)*2-1; //value between -1 and 1
    }
}

void MatrixPrint(float *M, int n, int p){
    for(int i=0;i<n*p;i++){
        if(i%p == 0){
            printf("\n");
        }
        printf("%.3f\t",M[i]);
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i=0; i<n*p; i++){
        Mout[i] = M1[i] + M2[i];
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i=0; i<n*p; i++){
        Mout[i] = M1[i] + M2[i];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    for(int a=0; a<n; a++){
        for(int b=0; b<n; b++){
            Mout[a*n+b] = 0.0;
            for(int i = 0; i<n; i++){
                Mout[a*n+b] = Mout[a*n+b] + M1[a*n+i]*M2[i*n+b];
            }
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    for(int a=0; a<n; a++){
        for(int b=0; b<n; b++){
            Mout[a*n+b] = 0.0;
            for(int i = 0; i<n; i++){
                Mout[a*n+b] = Mout[a*n+b] + M1[a*n+i]*M2[i*n+b];
            }
        }
    }
}

int main(int argc, char* argv[]){
    clock_t start_CPU, end_CPU;
    clock_t start_GPU, end_GPU;
    double elapsed_CPU, elapsed_GPU;
    int n=atoi(argv[1]), p=atoi(argv[2]);
    float* M1 = (float*)malloc(sizeof(float)*n*p);
    float* M2 = (float*)malloc(sizeof(float)*n*p);
    float* Mout = (float*)malloc(sizeof(float)*n*p);
    float* M1_cuda = NULL;
    float* M2_cuda = NULL;
    float* Mout_cuda = NULL;
    cudaMalloc((void**)&M1_cuda, sizeof(float)*n*p);
    cudaMalloc((void**)&M2_cuda, sizeof(float)*n*p);
    cudaMalloc((void**)&Mout_cuda, sizeof(float)*n*p);
    srand((unsigned int) time(NULL));
    /*Matrix random initialisation*/
    //CPU
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    //GPU
    cudaMemcpy(M1_cuda, M1, sizeof(float)*n*p, cudaMemcpyHostToDevice);
    cudaMemcpy(M2_cuda, M2, sizeof(float)*n*p, cudaMemcpyHostToDevice);
    /*Add of matrix*/
    //CPU
    //MatrixAdd(M1,M2,Mout,n,p);
    //GPU
    cudaMatrixAdd<<<n,n>>>(M1_cuda,M2_cuda,Mout_cuda,n,p);
    cudaMemcpy(Mout, Mout_cuda, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
    /*Mult of matrix*/
    //CPU
    start_CPU = clock();
    //MatrixMult(M1, M2, Mout, n);
    end_CPU = clock();
    elapsed_CPU = ((double)end_CPU - start_CPU);
    printf("%.2f micro secondes pour multiplication CPU\n", elapsed_CPU);
    //GPU
    start_GPU = clock();
    cudaMatrixMult<<<n,n>>>(M1_cuda,M2_cuda,Mout_cuda,n);
    end_GPU = clock();
    cudaMemcpy(Mout, Mout_cuda, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
    elapsed_GPU = ((double)end_GPU - start_GPU);
    printf("%.2f micro secondes pour multiplication GPU\n", elapsed_GPU);
    /*Matrix print*/
    //CPU
    //MatrixPrint(M1, n, p);
    //MatrixPrint(M2, n, p);
    //MatrixPrint(Mout, n, p);
    /*free memory*/
    free(M1);
    free(M2);
    free(Mout);
    cudaFree(M1_cuda);
    cudaFree(M2_cuda);
    cudaFree(Mout_cuda);
    /*Sychronisation of cuda*/
    cudaDeviceSynchronize();
    return 0;
}