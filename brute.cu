#include "stdio.h"
#include "string.h"
#include "math.h"


__global__ void crypt(char *password, char *salt, char *encryptedPassword)
{
 int i;
 long hash;
 char hashmap;
 char *p,*s;
 encryptedPassword[15]='\0';


 p = password;
 hash=199; // prime numbers are our friend
 while (*p != '\0')
 {
    hash = hash*3 + (*p);
    p++;
 }


 p = password;
 s = salt;
 for (i=0; i<15; i++)
 {
    //int i = blockIdx.x;
    hash = hash*7 + (*p) + (*s);
    hashmap = (hash % 94) + 33; // range of printable ASCII chars
    encryptedPassword[i] = hashmap;
    p++;
    if (*p == '\0') 
      p=password;
    s++;
    if (*s == '\0') 
      s=salt;
 }
}


__global__ void intToString(int num, char *s)
{
  int ones = (num) % 26;
  int twentySix = (num / 26) % 26;
  int twentySixSquared = (num / 26 / 26) % 26;
  int twentySixCubed = (num / 26 / 26 / 26) % 26;
  int twentySixFourth = (num / 26 / 26 / 26 / 26) % 26;
  int twentySixFifth = (num / 26 / 26 / 26 / 26 / 26) % 26;

  int i = 0;
  s[i++] = twentySixFifth + 'A';
  s[i++] = twentySixFourth + 'A';
  s[i++] = twentySixCubed + 'A';
  s[i++] = twentySixSquared + 'A';
  s[i++] = twentySix + 'A';
  s[i++] = ones + 'A';
  s[i] = '\0';
}

__device__ int stringToInt(char *s)
{
  int length = strlen(s);
  int sum = 0;
  int power = 0;

  for (int i = length-1; i >= 0; i--)
  {
	int digit = s[i] - 'A';
	sum += digit * pow(26,power);	
	power++;
  } 
  return sum;
}


int main()
{
 char salt[3];
 char password[9];
 char encryptedPassword[16];
 char encryptedGuess[16];
 char *dev_password;
 char *dev_encryptedPassword;
 char *dev_salt;
 char *dev_encryptedGuess;

 strcpy(salt,"XY");	// Our salt is just the fixed string XY

 printf("Welcome to the brute force password cracker.\n");
 printf("Enter a six-letter password in all upper-case letters: ");
 int n = scanf("%s",password);

 cudaMalloc((void**)&dev_password, sizeof(char));
 cudaMalloc((void**)&dev_encryptedPassword, sizeof(char));
 cudaMalloc((void**)&dev_salt, sizeof(char));
 cudaMalloc((void**)&dev_encryptedGuess, sizeof(char));

 crypt<<<1,1>>>(dev_password, dev_encryptedPassword, dev_salt);
 int num = stringToInt(password);
 printf("Using salt %s, your pasword maps to the encrypted password of: %s\n", salt, encryptedPassword);
 printf("Mapped to a number, %s is %d\n",password,num);

 printf("Working on cracking the encryption by trying all password combinations...\n");

 cudaMemcpy(password, dev_password, sizeof(char), cudaMemcpyDeviceToHost);
 cudaMemcpy(encryptedPassword, dev_encryptedPassword, sizeof(char), cudaMemcpyDeviceToHost);
 cudaMemcpy(salt, dev_salt, sizeof(char), cudaMemcpyDeviceToHost);
 cudaMemcpy(salt, dev_encryptedGuess, sizeof(char), cudaMemcpyDeviceToHost);

 for (int i = 0; i < 26*26*26*26*26*26; i++)
 {
      
        intToString(i, password); 
	crypt<<<1,1>>>(dev_password, dev_salt, dev_encryptedGuess);	
	if (!strcmp(encryptedPassword, encryptedGuess))
	{
		printf("CRACKED! The password is: %s\n", password);
		break;
	}	
	
 } 
}
