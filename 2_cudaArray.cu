#include "stdio.h"
#include "stdlib.h"

#define ROW 3
#define COL 4

__global__ void collapse(int *a, int *c) {

	int total = 0;
	for(int i = 0; i < COL; i++)
	{
		total = total + a[blockIdx.x*COL + i];
	}
	c[blockIdx.x] = total;
}

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
	cudaMemcpy(dev_c, c, size_c,	 cudaMemcpyHostToDevice);
	
	collapse<<< ROW, COL >>> (dev_a, dev_c);	

	cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost);	

	printf("\n");
	for(int i = 0; i < ROW; i++)
	{
	    printf("%i\n", c[i]);
	}	

	
	cudaFree (dev_a);
	cudaFree (dev_c);

}
	
