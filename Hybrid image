/*
    Author: Meet Maniar
*/

/*
    Description: In this program, there are few alternatives to extract high and low frequencies from the images and merging them.
    One of the way is to extract low frequency by applying GaussianBlur and high frequency by applying gaussianBlur and subtracting
    it from the original image and merging high and low frequency. This method was proposed in SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns.
*/

#include<stdio.h>
#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>
#include<cmath>
#include<iomanip>

using namespace cv;
using namespace std;


int main()
{
	//Mat lap_dat[] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
	
	

	Mat dog = imread("dog.bmp", WINDOW_AUTOSIZE);
	Mat cat = imread("cat.bmp", WINDOW_AUTOSIZE);
	imshow("Original-dog", dog);
	waitKey(0);
	imshow("Original-cat", cat);
	waitKey(0);
	
	Mat lowdog; 
	GaussianBlur(dog, lowdog, Size(25, 11), 5);
	//imshow("lowdog", lowdog);
	waitKey(0);

	Mat lowcat;
	GaussianBlur(cat, lowcat, Size(31, 21), 5);
	//imshow("lowcat", lowcat);
	waitKey(0);

	Mat highcat;
	highcat = cat - lowcat + 128;
	//imshow("highpass", highcat);
	waitKey(0);

	Mat hybsim;
	addWeighted(lowdog, 0.5, highcat, 0.5, 0.0, hybsim, -1);
	imshow("hybrid-part1", hybsim);
	waitKey(0);
	
   //Laplacian of Gaussian
	Mat lap;
	Laplacian(highcat, lap, 3, 1, 1.0, 0.0, 4);
	//filter2D(highcat, lap, -1, lap_dat);
	imshow("LoG", lap);
	waitKey(0);
	Mat hyblog;
	addWeighted(lowdog, 0.5, lap, 0.5, 0.0, hyblog, -1);
	namedWindow("hybrid with LoG", WINDOW_AUTOSIZE);
	imshow("hybrid with LoG", hyblog);  //Hybrid by Laplacian of Gaussian
	waitKey(0);

	//Difference of Gaussian
	Mat cat1,cat2,DOG,hybDOG;
	GaussianBlur(cat, cat1, Size(7, 7), 3.0);
	GaussianBlur(cat, cat2, Size(5, 5), 10.0);
	DOG = cat2 - cat1 + 128;
	//imshow("DOG", DOG);
	waitKey(0);
	addWeighted(lowdog, 0.5, DOG, 0.5, 0.0, hybDOG, -1);
	imshow("hybrid with DoG", hybDOG); //Hybrid by Difference of Gaussian
	waitKey(0);

	//Sobel
	Mat hybsob; 
	Mat cat_gray,sobel,grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	
	Sobel(cat, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT); /*Derivatives in X&Y directions*/
	Sobel(cat, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);/*Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );*/

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y); //convert partial results back to CV_8U
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);  //Merging
	
	
	//imshow("sobel", sobel);
	waitKey(0);
	
	addWeighted(lowdog, 0.5, sobel, 0.3, 0.0, hybsob, -1);  //hybrid by sobel
	imshow("hybrid with sobel", hybsob);
	waitKey(0);
}
