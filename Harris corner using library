/*
	Author: Meet Maniar 
*/
#include<stdio.h>
#include<iostream>
#include<string.h>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>

using namespace cv;
using namespace std;

Mat src, src_gray, src2, src_gray2;
int thresh = 25;
int max_thresh = 255;
void my_harris(int, int);

int main()
{
	src = imread("Yosemite1.jpg", WINDOW_AUTOSIZE);
	src2 = imread("Yosemite2.jpg", WINDOW_AUTOSIZE);
	cvtColor(src, src_gray, CV_BGR2GRAY);
	cvtColor(src2, src_gray2, CV_BGR2GRAY);
	namedWindow("source", WINDOW_AUTOSIZE);
	imshow("source", src);
	namedWindow("source2", WINDOW_AUTOSIZE);
	imshow("source2", src2);
	//namedWindow("gray", WINDOW_AUTOSIZE);
	//imshow("gray", src_gray);
	//namedWindow("gray2", WINDOW_AUTOSIZE);
	//imshow("gray2", src_gray2);
	my_harris(0, 0);
	waitKey(0);
	return(0);
}

void my_harris(int, int)
{
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);
	int blockSize = 10;
	int apertureSize = 3;
	double k = 0.04;

	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	//namedWindow("destination", WINDOW_AUTOSIZE);
	//imshow("destination", dst);
	waitKey(0);

	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//namedWindow("normalized", WINDOW_AUTOSIZE);
	//imshow("normalized", dst_norm);
	//waitKey(0);
	convertScaleAbs(dst_norm, dst_norm_scaled);

	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > thresh)
			{
				circle(src, Point(j, i), 5, Scalar(100), 1, 8, 0);
			}
		}
	}

	namedWindow("detected_corners", WINDOW_AUTOSIZE);
	imshow("detected_corners", src);
}
