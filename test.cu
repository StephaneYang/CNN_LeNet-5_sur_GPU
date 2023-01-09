#include <stdio.h>
#include <stdlib.h>


int main(void){
    FILE *f;
    f=fopen("layer_0.bin","rb");
    if (!f)
    { 
        printf("Unable to open file!");     
    }

    // déterminer la taille du fichier    
    fseek(f,0L, SEEK_END);
    size_t file_size = ftell(f);

    // nombre de float32 dans le fichier en supposant qu'ils le sont tous
    size_t float_count = file_size / sizeof(float);

    //allocation pour contenir les float
    float* weights = (float*)malloc(float_count * sizeof(float));

    rewind(f);

    fread(weights, sizeof(float), float_count, f);
    fclose(f);

    for(int i=0;i<float_count;i++){
        printf("%1.3f\n",weights[i]);
    }
    return 0;
} 