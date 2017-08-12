/*
    Author: Meet Maniar
*/
#include<stdio.h>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

float threshold_val = 100;  //threshold_val is very sensitive.

void non_maxima_suppression(const cv::Mat& image, cv::Mat& mask) {
	// find pixels that are equal to the local neighborhood 
	cv::dilate(image, mask, cv::Mat());
	cv::dilate(image, mask, Mat(), Point(-1, -1), 2, 1, 1);
}


int main()
{
	Mat src;
	Mat src_grayscale;
	Mat ip_x_der, ip_y_der;
	Mat ip_x_der_sqr, ip_y_der_sqr, ip_xy_der_mul;
	Mat ip_x_der_sqr_gau, ip_y_der_sqr_gau, ip_xy_der_sqr_gau;
	Mat covarianceH, covarianceH_gau;
	Mat x_sqr_y_sqr, xy_sqr;
	Mat mtrace, dst;
	Mat dst_norm, dst_norm_scaled;

	

	//Get Image
	src = imread("C:\\Users\\Meet-69\\Documents\\image_sets\\yosemite\\Yosemite1.jpg", WINDOW_AUTOSIZE);

	//Window to show source image
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", src);
	waitKey(0);
	cvtColor(src, src_grayscale, CV_BGR2GRAY);  //converting source to gray
	
	//Step A
	//Calculate x and y derivative of image
	Sobel(src_grayscale, ip_x_der, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
	Sobel(src_grayscale, ip_y_der, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);

	//Step B
	//Calculate Covariance Matrix H
	pow(ip_x_der, 2.0, ip_x_der_sqr);
	pow(ip_y_der, 2.0, ip_y_der_sqr);
	multiply(ip_x_der, ip_y_der, ip_xy_der_mul);

	covarianceH = ip_x_der_sqr + ip_y_der_sqr + ip_xy_der_mul;

	//Gaussian Filtering for better results
	GaussianBlur(ip_x_der_sqr, ip_x_der_sqr_gau, Size(3, 3), 2.0, 0.0, BORDER_DEFAULT);
	GaussianBlur(ip_y_der_sqr, ip_y_der_sqr_gau, Size(3, 3), 0.0, 2.0, BORDER_DEFAULT);
	GaussianBlur(ip_xy_der_mul, ip_xy_der_sqr_gau, Size(3, 3), 2.0, 2.0, BORDER_DEFAULT);
	covarianceH_gau = ip_x_der_sqr_gau + ip_y_der_sqr_gau + ip_xy_der_sqr_gau;

	//Step C
	//Haris Response
	multiply(ip_x_der_sqr_gau, ip_y_der_sqr_gau, x_sqr_y_sqr);
	multiply(ip_xy_der_sqr_gau, ip_xy_der_sqr_gau, xy_sqr);
	pow((ip_x_der_sqr_gau + ip_y_der_sqr_gau), 2.0, mtrace);
	dst = (x_sqr_y_sqr - xy_sqr) - 0.04 * mtrace;
  //dst = (x_sqr_y_sqr - xy_sqr) / mtrace;  This does not detect appropriate corners

	//normalizing result from 0 to 255
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
	// Drawing a circle around corners
	int count1 = 0;
	/*for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > threshold_val)
			{
				circle(src, Point(j, i), 5, Scalar(25,25,25), 1, 8, 0);
				count1 = count1 + 1;
			}
		}
	}
	cout << "Count:" << count1 << endl;*/

	int count2 = 0;
	Mat mask;
	non_maxima_suppression(dst_norm, mask); // extract local maxima

	// Drawing a circle around corners
	for (int i = 0; i < src_grayscale.rows; i++)
	{
		for (int j = 0; j < src_grayscale.cols; j++)
		{
			if ((int)mask.at<float>(i, j) > threshold_val)
			{
				circle(src, Point( j, i ), 1,  Scalar(255), 1, 8, 0 );
				count2 = count2 + 1;
			}

		}
	}

	cout << "Count : " << count2 << endl;
	

	namedWindow("X-derivative", WINDOW_AUTOSIZE);
	imshow("X-derivative", ip_x_der);
	
	namedWindow("Y-derivative", WINDOW_AUTOSIZE);
	imshow("Y-derivative", ip_y_der);
	
	namedWindow("convariance matrix without gaussian", WINDOW_AUTOSIZE);
	imshow("convariance matrix without gaussian", covarianceH);
	
	namedWindow("convariance matrix with gaussian", WINDOW_AUTOSIZE);
	imshow("convariance matrix with gaussian", covarianceH_gau);
	
	namedWindow("DetectedPoints", WINDOW_AUTOSIZE);  //display detected points
	imshow("DetectedPoints", src);
	
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", src_grayscale);


	waitKey(0);
	return(0);
}
