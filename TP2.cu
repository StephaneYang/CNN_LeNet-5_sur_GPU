#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>

#define WIDTH 28
#define HEIGHT 28

void VectInit(float *M, int n, int etat){
    if(etat==1){
        for(int i=0;i<n;i++){
            M[i] = (float)rand()/(float)(RAND_MAX)*2-1; //value between -1 and 1
        }
    }
    else{
        for(int i=0;i<n;i++){
            M[i] = 0.0; 
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

/*********Activation function**********/

__device__ float activation_tanh(float M){
    return (1-exp(-2*M))/(1+exp(-2*M));
}

__device__ float activation_softmax(float* M, int size, int output){
    int x = 0;
    for(int i=0; i<size; i++){
        x = x + exp(M[i]); 
    }
    return exp(M[output])/x;
}

/**************************************/

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
    out[k*size_out*size_out+i*size_out+j] =  activation_tanh(out[k*size_out*size_out+i*size_out+j]);         
}

__global__ void Padding(float* M, float* out, int size_M,int size_out){
    int i = threadIdx.x; // ligne
    int j = threadIdx.y; // col
    out[(i+2)*size_out+j+2] = M[size_M*i + j];
}

__global__ void AveragePooling(float* C1_data, float* S1_data, int size_C1, int size_S1){
    int i = threadIdx.x;// ligne
    int j = threadIdx.y;// col
    int k = blockIdx.x;// feature
    int n = size_C1;
    S1_data[k*size_S1*size_S1+i*size_S1+j] = (C1_data[n*n*k+2*i*n+2*j]+C1_data[n*n*k+(2*i+1)*n+2*j]+C1_data[n*n*k+2*i*n+2*j+1]+C1_data[n*n*k+(2*i+1)*n+2*j+1])*0.25;
}

__global__ void Dense(int output_layer,float* Weight, float* x, int size_weight, int size_x, int b){
    float output = 0.0;
    for(int i=0; i<size_weight; i++){
        for(int j=0; j<size_weight; j++){
            output = output + Weight[i*size_weight+j]*x[j];
        }
    }
    output = output+b;
}

void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
  }
}

int main(){
    /*initailisation random*/
    srand((unsigned int) time(NULL));

    /********load MNIST*******/
    int i, j;
    int ***img;
    int color[3]={255,0,0};
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;

    // Malloc image
    img = (int ***)malloc(HEIGHT*sizeof(int **));
    for(i=0; i<HEIGHT; i++){
        img[i]= (int **)malloc(WIDTH*sizeof(int *));
        for(j=0; j<WIDTH; j++){
        img[i][j] = (int *)malloc(sizeof(int)*3);
        }
    }
    if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
        printf("Can't open file");
        exit(1);
    }
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    for(i=0; i<HEIGHT; i++){
    for(j=0; j<WIDTH; j++){ 
            fread(&val, sizeof(unsigned char), 1, fptr);  
            img[i][j][0]=(int)val*color[0]/255;
            img[i][j][1]=(int)val*color[1]/255;
            img[i][j][2]=(int)val*color[2]/255;
        }
    }

    imgColorPrint(HEIGHT, WIDTH, img);

    // setup image grayscale
    for(i=0; i<HEIGHT; i++){
        for(j=0; j<WIDTH; j++){
            img[i][j][0] = ((i+j)*4)%255;
            img[i][j][1] = ((i+j)*4)%255;
            img[i][j][2] = ((i+j)*4)%255;
        }
    }

    // print image
    imgColorPrint(HEIGHT, WIDTH, img);

    /*****************************************************/

    /******************Weights load************************/
    
    FILE *f;
    f=fopen("layer_0.bin","rb");

    // déterminer la taille du fichier    
    fseek(f,0L, SEEK_END);
    size_t file_size = ftell(f);

    // nombre de float32 dans le fichier en supposant qu'ils le sont tous
    size_t float_count = file_size / sizeof(float);

    //allocation pour contenir les float
    float* weights_layer0 = (float*)malloc(float_count * sizeof(float));

    rewind(f);

    fread(weights_layer0, sizeof(float), float_count, f);
    fclose(f);

    f=fopen("layer_1.bin","rb");

    // déterminer la taille du fichier    
    fseek(f,0L, SEEK_END);
    file_size = ftell(f);

    // nombre de float32 dans le fichier en supposant qu'ils le sont tous
    float_count = file_size / sizeof(float);

    //allocation pour contenir les float
    float* weights_layer1 = (float*)malloc(float_count * sizeof(float));

    rewind(f);

    fread(weights_layer1, sizeof(float), float_count, f);
    fclose(f);

    f=fopen("layer_2.bin","rb");

    // déterminer la taille du fichier    
    fseek(f,0L, SEEK_END);
    file_size = ftell(f);

    // nombre de float32 dans le fichier en supposant qu'ils le sont tous
    float_count = file_size / sizeof(float);

    //allocation pour contenir les float
    float* weights_layer2 = (float*)malloc(float_count * sizeof(float));

    rewind(f);

    fread(weights_layer2, sizeof(float), float_count, f);
    fclose(f);

    f=fopen("layer_3.bin","rb");

    // déterminer la taille du fichier    
    fseek(f,0L, SEEK_END);
    file_size = ftell(f);

    // nombre de float32 dans le fichier en supposant qu'ils le sont tous
    float_count = file_size / sizeof(float);

    //allocation pour contenir les float
    float* weights_layer3 = (float*)malloc(float_count * sizeof(float));

    rewind(f);

    fread(weights_layer3, sizeof(float), float_count, f);
    fclose(f);

    f=fopen("layer_4.bin","rb");

    // déterminer la taille du fichier    
    fseek(f,0L, SEEK_END);
    file_size = ftell(f);

    // nombre de float32 dans le fichier en supposant qu'ils le sont tous
    float_count = file_size / sizeof(float);

    //allocation pour contenir les float
    float* weights_layer4 = (float*)malloc(float_count * sizeof(float));

    rewind(f);

    fread(weights_layer4, sizeof(float), float_count, f);
    fclose(f);

    /******************************************************/

    /*declaration matrix*/
    float* raw_data = (float*)malloc(sizeof(float)*32*32);
    float* raw_data_padding = (float*)malloc(sizeof(float)*36*36);
    float* C1_data = (float*)malloc(sizeof(float)*6*28*28);
    float* S1_data = (float*)malloc(sizeof(float)*6*14*14);
    float* C1_kernel= (float*)malloc(sizeof(float)*6*5*5);
    /********************/

    /*declaration matrix GPU*/
    float* raw_data_cuda = NULL;
    float* C1_data_cuda = NULL;
    float* S1_data_cuda = NULL;
    float* C1_kernel_cuda = NULL;
    float* raw_data_padding_cuda = NULL;
    cudaMalloc((void**)&raw_data_cuda, sizeof(float)*32*32);
    cudaMalloc((void**)&raw_data_padding_cuda, sizeof(float)*36*36);
    cudaMalloc((void**)&C1_data_cuda, sizeof(float)*6*28*28);
    cudaMalloc((void**)&S1_data_cuda, sizeof(float)*6*14*14);
    cudaMalloc((void**)&C1_kernel_cuda, sizeof(float)*6*5*5);
    /************************/

    /*Matrix CPU Initialisation*/
    VectInit(raw_data, 32*32, 1);
    VectInit(raw_data_padding, 36*36, 0);
    VectInit(C1_data, 6*28*28, 0);
    VectInit(S1_data, 6*14*14, 0);
    VectInit(C1_kernel, 6*5*5, 1);
    /***********************/
    
    /*Matrix GPU Initialisation*/
    cudaMemcpy(raw_data_cuda, raw_data, sizeof(float)*32*32, cudaMemcpyHostToDevice);
    cudaMemcpy(raw_data_padding_cuda, raw_data_padding, sizeof(float)*36*36, cudaMemcpyHostToDevice);
    cudaMemcpy(C1_data_cuda, C1_data, sizeof(float)*6*28*28, cudaMemcpyHostToDevice);
    cudaMemcpy(S1_data_cuda, S1_data, sizeof(float)*6*14*14, cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernel_cuda, C1_kernel, sizeof(float)*6*5*5, cudaMemcpyHostToDevice);
    /***************************/

    /*apply padding*/
    dim3 gridDim3_padd(1,1,1);
    dim3 blockDim3_padd(32,32,1);
    Padding<<<gridDim3_padd,blockDim3_padd>>>(raw_data_cuda, raw_data_padding_cuda, 32,36);
    cudaMemcpy(raw_data_padding, raw_data_padding_cuda, sizeof(float)*36*36, cudaMemcpyDeviceToHost);
    /***************/

    /*Matrix Convolution*/
    dim3 gridDim3_conv(6,1,1);
    dim3 blockDim3_conv(28,28,1);
    Conv2D<<<gridDim3_conv,blockDim3_conv>>>(raw_data_cuda, C1_kernel_cuda, C1_data_cuda, 32, 5, 28);
    cudaMemcpy(C1_data, C1_data_cuda, sizeof(float)*6*28*28, cudaMemcpyDeviceToHost);
    /********************/

    /*Matrix average pooling*/
    dim3 gridDim3_pool(6,1,1);
    dim3 blockDim3_pool(14,14,1);
    AveragePooling<<<gridDim3_pool,blockDim3_pool>>>(C1_data_cuda, S1_data_cuda, 28, 14);
    cudaMemcpy(S1_data, S1_data_cuda, sizeof(float)*6*14*14, cudaMemcpyDeviceToHost);
    /************************/

    /*Matrix print*/
    printf("\nRaw data \n");
    MatrixPrint(raw_data, 32, 32, 1);
    printf("\nRaw data padding \n");
    MatrixPrint(raw_data_padding, 36, 36, 1);
    printf("\nC1_kernel \n");
    MatrixPrint(C1_kernel, 5, 5, 6);
    printf("\nC1_data \n");
    MatrixPrint(C1_data, 28, 28, 6);
    printf("\nS1_data \n");
    MatrixPrint(S1_data, 14, 14, 6);
    /**************/

    /*free memory*/
    free(raw_data);
    free(raw_data_padding);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    cudaFree(raw_data_cuda);
    cudaFree(raw_data_padding_cuda);
    cudaFree(C1_data_cuda);
    cudaFree(S1_data_cuda);
    cudaFree(C1_kernel_cuda);
    /*************/

    /*Sychronisation of cuda*/
    cudaDeviceSynchronize();
    return 0;
}