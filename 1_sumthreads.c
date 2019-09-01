#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define M 10
#define N 10

int rows,
    columns,
    a[M][N],
    s[M];

void *calculatesum(void *ptr)
{
	int k = *((int*) ptr);
	int i;
	for (i = 0; i < columns; i++)
	{
	   s[k] += a[k][i];
	}
	return NULL;
}

int main()
{
	int i, j, *ptr, rc;
	int sum = 0;
	pthread_t thread[M];

	
	printf("No of Columns: ");
	scanf("%d", &rows);
	printf("No of Rows: ");
	scanf("%d", &columns);

	for(i = 0; i < rows; i++)
	{
	   for(j = 0; j < columns; j++)
	   {
	      printf("a[%d][%d] = ", i, j);
	      scanf("%d", &a[i][j]);
	   }
	}

	for(i = 0; i < rows; i++)
	{
	   for(j = 0; j < columns; j++)
		printf("%d ", a[i][j]);
		printf("\n"); 
	}

	for (i = 0; i < rows; i++)
	{
	   ptr = malloc(sizeof(int));
	   *ptr = i;
	   rc = pthread_create(&thread[i], NULL, calculatesum, ptr);
	   if (rc != 0)
	   {
	 	printf("Thread failed!");
		exit(-1);
	   }
	}

	for (i = 0; i < rows; i++)
	{
	    pthread_join(thread[i], NULL);
	}

	for (i = 0; i < rows; i++)
	{
	   sum += s[i];
	}
	
	printf("Total sum : %d\n", sum);

	return 0;
}
