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

using namespace std;
using namespace cv;

int MAX_KERNEL_LENGTH, numHilos;
Mat src; Mat dst;
char window_name[] = "Practica 2";
Vec3b newPixel( int fil, int col, int numFil, int numCol);
int saveImage( string name );
double runThreads( string image);
double secuencial( string image);
int main( int argc, char *argv[] )
{
	string image = argv[1];
	MAX_KERNEL_LENGTH = atoi(argv[2]);
	numHilos = atoi(argv[3]);
	double timeThread = runThreads(image);
  	return 0;
}
double secuencial( string image){
	src = imread(image);
	clock_t timeStart, timeFinish;
	timeStart = clock();
	for(int i = 0; i < src.rows; i++){
		for(int j = 0; j < src.cols; j++){
			src.at<Vec3b>(i, j) = newPixel(i,j, src.rows, src.cols);
		}
	}
	timeFinish = clock();
	double time = (double(timeFinish-timeStart)/CLOCKS_PER_SEC);
	return time;
}

double runThreads( string image){
	omp_set_num_threads(numHilos);
	src = imread(image);
	int delta = src.rows/numHilos;
	int limits[numHilos];
	for (int i = 0; i < numHilos; ++i){
		limits[i] = delta * i  ;
	}
	int count = 0;
	int x = numHilos;
	clock_t timeStart, timeFinish;
	timeStart = clock();
	#pragma omp parallel
	{
		int idHilo = omp_get_thread_num();
		int start = limits[count];
		int finish = start + delta;
		count += 1;
		#pragma omp parallel for collapse(2) 
		for(int i = start; i < finish; i++){
			for(int j = 0; j < src.cols; j++){
				src.at<Vec3b>(i, j) = newPixel(i,j, src.rows, src.cols);
			}
		}
	}
	timeFinish = clock();
	double time = (double(timeFinish-timeStart)/CLOCKS_PER_SEC);
	return time;
}

Vec3b newPixel(int fil, int col, int numFil, int numCol){
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
						pos[index]  = src.at<Vec3b>(fil + i, col + j);
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
	Vec3b newPixel = src.at<Vec3b>(fil, col);
	newPixel.val[0] = blue/size;
	newPixel.val[1] = green/size;
	newPixel.val[2] = red/size;
	return newPixel;
}

int saveImage(string name){
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite(name, src, compression_params);
}