/*
	Author: Meet Maniar 
*/
#include<stdio.h>
#include<iostream>
#include<conio.h>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>
#include<cmath>
#include<iomanip>

using namespace cv;
using namespace std;

void meet_filter(Mat ip, Mat op, int x)
{
	int r = x - ip.rows;
	int c = x - ip.cols;
	
	if (r > 0)
	{
		if (c > 0)
		{
			copyMakeBorder(ip, op, r, r, c, c, BORDER_CONSTANT, Scalar(0));
			
		}
		if (c < 0)
		{
			copyMakeBorder(ip, op, r, r, 0, 0, BORDER_CONSTANT, Scalar(0));
			
		}
	}
	else
	{
		if (c > 0)
		{
			copyMakeBorder(ip, op, 0, 0, c, c, BORDER_CONSTANT, Scalar(0));
			
		}
		else
		{
			op = ip;
			
		}
		for (int i = 0; i < op.rows; i++)
		{
			for (int j = 0; j < op.cols; j++)
			{
				op.at<Vec3b>(i, j) = (op.at<Vec3b>(i - 1, j - 1) + op.at<Vec3b>(i - 1, j - 1)) / 2;
			}
		}
	}
	//namedWindow("padded", WINDOW_AUTOSIZE);
	//imshow("padded", op);
	//waitKey(0);

	Mat filtered, ker = getGaussianKernel(x, 20.0, CV_32F);
	filter2D(op, filtered, -1, ker);
	namedWindow("filtered", WINDOW_AUTOSIZE);
	imshow("filtered", filtered);
	waitKey(0);
}

int main()
{	
	Mat src1, src2, dst1, dst2, dst3;
	src1 = imread("dog.bmp", WINDOW_AUTOSIZE);
	src2 = imread("cat.bmp", WINDOW_AUTOSIZE);
	meet_filter(src1, dst1, 7);
	meet_filter(src2, dst2, 7);
	addWeighted(dst1, 0.5, dst2, 0.5, 0.0, dst3, -1);

	
	namedWindow("merged", WINDOW_AUTOSIZE);
	imshow("merged", dst3);
	/*dst3 = src2 - dst2 + 128;*/
	addWeighted(dst1, 0.5, dst3, 0.5, 0.0, hyb, -1);
	namedWindow("hp", WINDOW_AUTOSIZE);
	imshow("hp", hyb);*/
	waitKey(0);

} 
//Read the change description to know about the program.
