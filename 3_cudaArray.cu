#include "stdio.h"
#include "stdlib.h"

#define ROW 8
#define COL 8

__global__ void reduce(int *data1, int *data2);
void randomArray(int *a, int n);
   

int main(void)
{	
	int array[ROW][COL];
        int c[ROW];
        int *dev_a;
        int *dev_c;

        int size_2D = ROW*COL*sizeof(int);
        int size_c = ROW*sizeof(int);

        cudaMalloc((void**)&dev_a, size_2D);
        cudaMalloc((void**)&dev_c, size_c);

        for (int i = 0; i < ROW; i++)
        {
                if(i == ROW -1)
                {
                        for(int j = 0; j < COL; j++)
                        {
                                array[i][j] = (j*2);
                                printf("%i ", array[i][j]);
                        }
                }
                else
                {
                        for(int j = 0; j < COL; j++)
                        {
                                array[i][j] = j;
                                printf("%i ", array[i][j]);
                        }
                }
                printf("\n");
        }

         cudaMemcpy(dev_a, array, size_2D, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, c, size_c,     cudaMemcpyHostToDevice);

        reduce<<< ROW, COL >>> (dev_a, dev_c);

        cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost);
        
        printf("\n");
        for(int i = 0; i < ROW; i++)
        {
            printf("%i\n", c[i]);
        }

        //Release the memory
        cudaFree (dev_a);
        cudaFree (dev_c);

}
    
__global__ void reduce(int *data1, int *data2)
{
        __shared__ int xdata[ROW];

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        xdata[threadIdx.x] = data1[i];

        __syncthreads();

        for(int s = 1; s < blockDim.x; s *= 2)
        {
                int index = 2 * s * threadIdx.x;;
                if(index < blockDim.x)
                {
                        xdata[index] += xdata[index + s];
                }
                __syncthreads();
        }
        if(threadIdx.x == 0) data2[blockIdx.x] = xdata[0];
}

void randomArray(int *a, int n)
{
        for (int i = 0; i < n; i++)
                a[i] = rand() % 10;  //random 0 to 9
}

