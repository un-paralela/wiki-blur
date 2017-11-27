#include <stdio.h>
#include <cuda_runtime.h>

__global__ void blur(const int *image, int *newImage,int ROWS, int COLS, int kernel, int numElements){
    // Calculates the index of the, this index will go from 0 to the number of rows * number of cols
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    // Validates that the calculated index will not be bigger than the size of the image
    if (index < numElements){
        int ch;
        // Iterates over the three channels
		for(ch = 0; ch<3; ch++){
			int sum = 0;
		    int count = 0;
            // Calculates the down limit row
			int limitDownRow = ROWS * COLS * (ch + 1);
            // Calculates the up limit row
		    int limitUpRow = ROWS * COLS * ch - 1;
            // Calculates the 'real' row
		    int currentRow = index / COLS;
            // Calculates the left limit column
		    int limitLeftCol = (currentRow * COLS - 1) + (ROWS * COLS * ch);
            // Calculates the right limit column
		    int limitRightCol = (currentRow + 1) * COLS+ (ROWS * COLS * ch);
            // Calculates the real index of the image vector
			int realIndex = index + (ROWS * COLS * ch);
            int j;
            // The sum of rows or cols will go from -kernel to +kernel
		    for(j=-kernel; j<=kernel; j++){
                // Calculates the new row adding (or subtracting) jth number of cols
		        int newRow = realIndex + j * COLS;
                // The pixel value will be added if it is not the same row, bigger than the lower row limit or lower than the upper row limit
		        if(newRow!=realIndex && newRow>limitUpRow && newRow<limitDownRow){
		            sum+=image[newRow];
		            count++;
		        }
                // Calculates the new column adding (or subtracting) 'j'
		        int newCol = realIndex + j;
                // The pixel value will be added if it is not the same column, bigger than the right columnt limit or lower than the lower row limit
		        if(newCol!=realIndex && newCol>limitLeftCol && newCol<limitRightCol){
		            sum+=image[newCol];
		            count++;
		        }
		    }
            // The blured pixel value will be the division between the added pixel values and the number of added pixel values
		    newImage[realIndex]=sum / count;
		}
    }
}

int readFile(char *fileName, int *array, int indexVec){
    FILE *file;
    int character;
    // Opens the given file
    file = fopen(fileName,"r");
    if (file == NULL){
        fprintf(stderr,"Error de apertura del file");
        exit(EXIT_FAILURE);
    } else {
        // Char array to store a pixel value, it could go from 0 to 255, i.e. 000 to 255
        char num[3] = {};
        int d;
        int index = 0;
        // Reads the file character per character
        while((character = fgetc(file)) != EOF){
            // When the character is different from a tab or a new line, the given value is stored in the char array in the next available position
            if (character != '\t' && character != '\n'){
                num[index++] = character;
            } else {
                // When the character is a tab or a new line it means that it will come a new pixel value
                index = 0;
                // Converts the given char array to an integer value and stores it in the variable 'd'
                sscanf(num, "%d", &d);
                // Stores the las value in the array and increases the indexVec
                array[indexVec++] = d;
                // Empties the char array 'num'
                memset(num,0,sizeof(num));
            }
        }
        // Converts the last given char array to an integer value and stores it in the variable 'd'
        sscanf(num, "%d", &d);
        // Stores the las value in the array
        array[indexVec] = d;
    }
    fclose(file);
    // Returns the last stored position of the array, indicating that the next vector should start at this position
    return indexVec;
}

int main(int argc, char** argv){
	if(argc!=8){
        fprintf(stderr, "The format should be: fileRed fileGreen fileBlue #_Rows #_Cols kernel num_threads");
        exit(EXIT_FAILURE);
    }
    // Reads the number of rows and cols of an image and the given files
    int NUM_ROWS = atoi(argv[4]);
    int NUM_COLS = atoi(argv[5]);
    int NUM_CHS = 3;
    // Checks if the given kernel is odd and bigger than 1
    if(atoi(argv[6])%2==0 && atoi(argv[6])<=1){
        fprintf(stderr, "The number of kernels should be odd and bigger than 1");
        exit(EXIT_FAILURE);
    }
    int kernel = atoi(argv[6])/2;
    // The size of the image vector depends on the rows, cols, channels and the vector will have only int values
    int SIZE = NUM_ROWS * NUM_COLS * NUM_CHS * sizeof(int);
    // Allocate the host image vector
	int *h_image = (int *)malloc(SIZE);
    if(h_image == NULL){
        fprintf(stderr, "Failet do allocate host vector image");
        exit(EXIT_FAILURE);
    }
    // Reads the given file and puts it in the given array starting at the given position
    // For this case, reads the file in the argument 1, stores it in the host image starting in the position 0
    int index = readFile(argv[1],h_image,0);
    index = readFile(argv[2],h_image,index);
    index = readFile(argv[3],h_image,index);
    // Allocate the host newImage vector
	int *h_newImage = (int *)malloc(SIZE);
    if(h_newImage == NULL){
        fprintf(stderr, "Failet do allocate host vector newImage");
        exit(EXIT_FAILURE);
    }
	cudaError_t err = cudaSuccess;
	int *d_image = NULL;
    // Allocate the device image vector
	err = cudaMalloc((void **)&d_image, SIZE);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Allocate the device newImage vector
    int *d_newImage = NULL;
	err = cudaMalloc((void **)&d_newImage, SIZE);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector newImage (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copies the memory from the host image to the device image
    err = cudaMemcpy(d_image, h_image, SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy first dimesion of image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // The threads per block will depend on a given argument of the user
    int threadsPerBlock = atoi(argv[7]);
    int totalThreads = NUM_ROWS * NUM_COLS + threadsPerBlock - 1;
    // Calculates the number of blocks per grid that will depend on the number of rows and cols of the image
    int blocksPerGrid = totalThreads / threadsPerBlock;
    // Launches the blur function with the device image, the device newImage (result image), number of rows, cols, number of kernel and the total elements, i.e. size of each channel
    // Each thread will be responsible of calculate the blur effect on an especific pixel in their three channels
    blur<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_newImage,NUM_ROWS, NUM_COLS, kernel, NUM_ROWS * NUM_COLS);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch blur kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copies the memory from the device newImage to the host newImage
    err = cudaMemcpy(h_newImage, d_newImage, SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy first dimesion of image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Creates three files, in which will be stored the result of the blur, each file will be a channel of the blured image
	const char *files[3] = {"redResult.txt","greenResult.txt","blueResult.txt"};
	int ch;
	for(ch = 0; ch < NUM_CHS; ch++){
		FILE *file = fopen(files[ch],"w");
		for(index = NUM_ROWS * NUM_COLS * ch; index < NUM_ROWS * NUM_COLS * (ch + 1); index++){
			fprintf(file, "%i\t", h_newImage[index]);
		}
		fclose(file);
	}
    // Frees the memory of the device image
    err = cudaFree(d_image);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Frees the memory of the device newImage
    err = cudaFree(d_newImage);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector newImage (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Frees the memory of the host image
    free(h_image);
    // Frees the memory of the host newImage
    free(h_newImage);
}