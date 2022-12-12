#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

void VectInit(float *M, int n, int etat){
    if(etat==1){
        for(int i=0;i<n;i++){
            M[i] = (float)rand()/(float)(RAND_MAX); //value between 0 and 1
        }
    }
    else{
        for(int i=0;i<n;i++){
            M[i] = 0; 
        }
    }
}

void MatrixPrint(float *M, int x, int y, int z){
    for(int i=0;i<x*y*z;i++){
        if(i%y == 0){
            printf("\n");
        }
        if(i%(y*x) == 0){
            printf("\n");
        }
        printf("%1.2f ",M[i]);
    }
    printf("\n");
}

__global__ void Conv2D(float* M, float* kernel, float* out, int size_matrix, int size_kernel, int size_out){
    // les sizes correspondent à la longueur d'un côté et non la taille totale
    int i = threadIdx.x; // ligne
    int j = threadIdx.y; // col
    int k = blockIdx.x; // feature
    out[k*size_out*size_out+i*size_out+j] = 0;
    for(int position_i = 0; position_i<size_kernel; position_i++){
        for(int position_j = 0; position_j<size_kernel; position_j++){
            out[k*size_out*size_out+i*size_out+j] += M[size_matrix*(i+position_i)+j+position_j]*kernel[k*size_kernel*size_kernel+size_kernel*position_i+position_j];
        }
    }
                    
}

int main(){
    /*initailisation random*/
    srand((unsigned int) time(NULL));

    /*declaration matrix*/
    float* raw_data = (float*)malloc(sizeof(float)*32*32);
    float* C1_data = (float*)malloc(sizeof(float)*6*28*28);
    float* S1_data = (float*)malloc(sizeof(float)*6*14*14);
    float* C1_kernel= (float*)malloc(sizeof(float)*6*5*5);
    /********************/

    /*declaration matrix GPU*/
    float* raw_data_cuda = NULL;
    float* C1_data_cuda = NULL;
    float* S1_data_cuda = NULL;
    float* C1_kernel_cuda = NULL;
    cudaMalloc((void**)&raw_data_cuda, sizeof(float)*32*32);
    cudaMalloc((void**)&C1_data_cuda, sizeof(float)*6*28*28);
    cudaMalloc((void**)&S1_data_cuda, sizeof(float)*6*14*14);
    cudaMalloc((void**)&C1_kernel_cuda, sizeof(float)*6*5*5);
    /************************/

    /*Matrix CPU Initialisation*/
    VectInit(raw_data, 32*32, 1);
    VectInit(C1_data, 6*28*28, 0);
    VectInit(S1_data, 6*14*14, 0);
    VectInit(C1_kernel, 6*5*5, 1);
    /***********************/
    
    /*Matrix GPU Initialisation*/
    cudaMemcpy(raw_data_cuda, raw_data, sizeof(float)*32*32, cudaMemcpyHostToDevice);
    cudaMemcpy(C1_data_cuda, C1_data, sizeof(float)*6*28*28, cudaMemcpyHostToDevice);
    cudaMemcpy(S1_data_cuda, S1_data, sizeof(float)*6*14*14, cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernel_cuda, C1_kernel, sizeof(float)*6*5*5, cudaMemcpyHostToDevice);
    /***************************/

    /*Matrix Convolution*/
    dim3 gridDim2(6,1);
    dim3 blockDim2(32,32);
    Conv2D<<<gridDim2,blockDim2>>>(raw_data_cuda, C1_kernel_cuda, C1_data_cuda, 32, 5, 28);
    cudaMemcpy(C1_data, C1_data_cuda, sizeof(float)*6*28*28, cudaMemcpyDeviceToHost);
    /********************/

    /*Matrix print*/
    printf("\nRaw data \n");
    MatrixPrint(raw_data, 32, 32, 1);
    printf("\nC1_kernel \n");
    MatrixPrint(C1_kernel, 5, 5, 6);
    printf("\nC1_data \n");
    MatrixPrint(C1_data, 28, 28, 6);
    /**************/

    /*free memory*/
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    cudaFree(raw_data_cuda);
    cudaFree(C1_data_cuda);
    cudaFree(S1_data_cuda);
    cudaFree(C1_kernel_cuda);
    /*************/

    /*Sychronisation of cuda*/
    cudaDeviceSynchronize();
    return 0;
}