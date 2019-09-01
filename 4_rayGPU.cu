// This is a simple ray tracer that shoots rays top down toward randomly
// generates spheres and draws the sphere in a random color based on where
// the ray hits it.

#include "FreeImage.h"
#include "stdio.h"

#define DIM 1028
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    // Tells us if a ray hits the sphere; return the
    // depth of the hit, or -infinity if the ray misses the sphere
    __device__ float hit( float ox, float oy, float *n ) 
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius)
        {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

#define SPHERES 40

// Loops through each pixel in the image (represented by arrays of
// red, green, and blue) and then for each pixel checks if a ray from
// top down hits one of the randomly generated spheres.
// If so, calculate a shade of color based on where the ray hits it.
__global__ void drawSpheres(Sphere *spheres, char *red, char *green, char *blue)
{
    
 	int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.x + blockIdx.y * blockDim.y;
	
	float   ox = (x - DIM/2);
	float   oy = (y - DIM/2);

	float   r=0, g=0, b=0;
	float   maxz = -INF;
	for(int i=0; i<SPHERES; i++)
 	{
        	float   n;
        	float   t = spheres[i].hit( ox, oy, &n );
        	if (t > maxz)
		{
			// Scale RGB color based on z depth of sphere
            		float fscale = n;
            		r = spheres[i].r * fscale;
            		g = spheres[i].g * fscale;
			b = spheres[i].b * fscale;
            		maxz = t;
        	} 
        }
	
    	int offset = x + y * blockDim.x * gridDim.x;
    	red[offset] = (char) (r * 255);
    	green[offset] = (char) (g * 255);
    	blue[offset] = (char) (b * 255);
    
}

int main()
{
  FreeImage_Initialise();
  atexit(FreeImage_DeInitialise);
  FIBITMAP * bitmap = FreeImage_Allocate(DIM, DIM, 24);
  srand(time(NULL));

 
  char red[DIM][DIM];
  char green[DIM][DIM];
  char blue[DIM][DIM];
  char *dev_red; 
  char *dev_green; 
  char *dev_blue;
  Sphere *dev_spheres;
  
 
 
  // Dynamically create enough memory for DIM * DIM array of char.
  // By making these dynamic rather than auto (e.g. char red[DIM][DIM])
  // we can make them much bigger since they are allocated off the heap
  cudaMalloc((void**)&dev_red, DIM*DIM*sizeof(char));
  cudaMalloc((void**)&dev_green, DIM*DIM*sizeof(char));
  cudaMalloc((void**)&dev_blue, DIM*DIM*sizeof(char));
  cudaMalloc((void**)&dev_spheres, sizeof(Sphere)*SPHERES);

  

  //Create random spheres at different coordinates, colors, radius
  //Sphere spheres[SPHERES];
 
  Sphere *spheres =(Sphere*)malloc(sizeof(Sphere)*SPHERES); 
  for (int i = 0; i<SPHERES; i++)
  {
        spheres[i].r = rnd( 1.0f );
        spheres[i].g = rnd( 1.0f );
        spheres[i].b = rnd( 1.0f );
        spheres[i].x = rnd( (float) DIM ) - (DIM/2.0);
        spheres[i].y = rnd( (float) DIM ) - (DIM/2.0);
        spheres[i].z = rnd( (float) DIM ) - (DIM/2.0);
        spheres[i].radius = rnd( 100.0f ) + 20;
  } 

  cudaMemcpy(dev_spheres, spheres, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice);
  free(spheres);

  dim3 grid(DIM/16,DIM/16);
  dim3 threads(16,16);
  drawSpheres<<<grid,threads>>>(dev_spheres, dev_red, dev_green, dev_blue);
 
  cudaMemcpy(dev_red, red, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
  cudaMemcpy(dev_green, green, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
  cudaMemcpy(dev_blue, blue, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
 

  RGBQUAD color;
  for (int i = 0; i < DIM; i++)
  {
    for (int j = 0; j < DIM; j++)
    {
      //int index = j*DIM + i;
      color.rgbRed = red[i][j];
      color.rgbGreen = green[i][j];
      color.rgbBlue = blue[i][j];
      FreeImage_SetPixelColor(bitmap, i, j, &color);
    }
  }
  

  FreeImage_Save(FIF_PNG, bitmap, "ray.png", 0);
  FreeImage_Unload(bitmap);
  cudaFree(dev_red);
  cudaFree(dev_green);
  cudaFree(dev_blue);
  cudaFree(dev_spheres);

  return 0;
}
