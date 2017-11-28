#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <fstream>
#include <cstdlib>
#include "pthread.h"

using namespace std;
using namespace cv;

int MAX_KERNEL_LENGTH, numHilos;
string image;
clock_t timeStart, timeFinish;

Mat src; Mat clonHilos; Mat clonSecuencial;
int totalFil, totalCol, delta;
char window_name[] = "Practica 1";
Vec3b newPixel( int fil, int col, int numFil, int numCol, Mat imagen);
int saveImage( string name, Mat image );
double secuencial( string image);
double timeHilos;
struct thread_data {
	int  start;
	int finish;
};
	
void *runThreads( void *threadarg){	
	struct thread_data *my_data;
	my_data = (struct thread_data *) threadarg;

	int start = my_data->start;
	int end = my_data->finish;
	for(int i =start; i < end; i++){
		for(int j = 0; j < src.cols; j++){
			clonHilos.at<Vec3b>(i, j) = newPixel(i,j, clonHilos.rows, clonHilos.cols, clonHilos);		
		}
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){
	image = argv[1];
	MAX_KERNEL_LENGTH = atoi(argv[2]);
	numHilos = atoi(argv[3]);

	src = imread(image);
	clonHilos = imread(image);
	clonSecuencial = imread(image);

	totalFil = src.rows;
	totalCol = src.cols;

	pthread_t threads[numHilos];
	struct thread_data td[numHilos];
	delta = src.rows/numHilos;
	
	int rc;
	for (int i = 0; i < numHilos; ++i){
		td[i].start = delta * i;
		td[i].finish = (delta * i) + delta;
		rc = pthread_create(&threads[i], NULL, runThreads, (void *)&td[i]);
	}
	timeStart = clock();
	for (int i = 0; i < numHilos; ++i){
		void *ret;
		if (pthread_join(threads[i], &ret) != 0) {
			perror("pthread_create() error");
			exit(3);
		}
	}

	timeFinish = clock();
	return 0;
}

double secuencial( string image){
	src = imread(image);
	clock_t timeStart, timeFinish;
	timeStart = clock();
	for(int i = 0; i < src.rows; i++){
		for(int j = 0; j < src.cols; j++){
			clonSecuencial.at<Vec3b>(i, j) = newPixel(i,j, src.rows, src.cols, clonSecuencial);
		}
	}
	timeFinish = clock();
	double time = (double(timeFinish-timeStart)/CLOCKS_PER_SEC);
	return time;
}

Vec3b newPixel(int fil, int col, int numFil, int numCol, Mat imagen){
	int size = (MAX_KERNEL_LENGTH * MAX_KERNEL_LENGTH) - 1; 
	int level = MAX_KERNEL_LENGTH/2;
	Vec3b pos[size];
	for (int i = 0; i < size; ++i){
		pos[i] = 0;
	}
	int index = 0;
	for (int k = 1; k <= level; ++k){
		for (int i = -k; i <=k; ++i){
			for (int j = -k; j <=k; ++j){
				if ( abs(i) == k || abs(j) == k){
					if (fil + i >= 0 && fil + i < numFil  &&  col + j >= 0 && col + j < numCol){
						pos[index]  = imagen.at<Vec3b>(fil + i,col + j);
					}
					index += 1;
				}
			}
		}
	}
	
	int blue = 0;
	int green = 0;
	int red = 0;
	for (int i = 0; i < size; ++i){
		blue += pos[i][0];
		green += pos[i][1];
		red += pos[i][2];
	}

	Vec3b newPixel = imagen.at<Vec3b>(fil, col);

	newPixel.val[0] = blue/size;
	newPixel.val[1] = green/size;
	newPixel.val[2] = red/size;
	return newPixel;
}

int saveImage(string name, Mat image){
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite(name, image, compression_params);
}