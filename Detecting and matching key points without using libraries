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
#include<opencv2\legacy/legacy.hpp>
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\nonfree\nonfree.hpp>
#include<opencv2\nonfree\features2d.hpp>
//#include<opencv2\flann\*>
//#include<cv.h>
#include<vector>

using namespace cv;
using namespace std;

int main()

{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	
	Mat src,src_gaussian,src_gray;
	src = imread("Rainier1.png", WINDOW_AUTOSIZE);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", src);
	waitKey(0);

	//Applying Gaussian Blur
	GaussianBlur(src, src_gaussian, Size(3, 3), 0, 0, BORDER_DEFAULT);
	namedWindow("GaussianBluredImage", WINDOW_AUTOSIZE);
	imshow("GaussianBluredImage", src_gaussian);
	waitKey(0);
	//convert gaussian blured image to gray scale
	cvtColor(src_gaussian, src_gray, CV_BGR2GRAY);
	namedWindow("GrayImage", WINDOW_AUTOSIZE);
	imshow("GrayImage", src_gray);
	waitKey(0);

	// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	namedWindow("Grad-x", WINDOW_AUTOSIZE);
	imshow("Grad-x", abs_grad_x);

	namedWindow("Grad-y", WINDOW_AUTOSIZE);
	imshow("Grad-y", abs_grad_y);
	waitKey(0);

	//taking square of X-derivative and Y-derivative and multiplying X-derivative and Y-derivative
	Mat abs_grad_x_sq, abs_grad_y_sq, abs_grad_xy;
		pow(abs_grad_x, 2.0, abs_grad_x_sq);
		pow(abs_grad_y, 2.0, abs_grad_y_sq);
		multiply(abs_grad_x, abs_grad_y, abs_grad_xy);

	//Applying Gaussian to all the newly formed Matrices
		Mat abs_grad_x_sq_gaussian, abs_grad_y_sq_gaussian, abs_grad_xy_gaussian;
		GaussianBlur(abs_grad_x_sq, abs_grad_x_sq_gaussian, Size(3, 3), 2, 0, BORDER_DEFAULT); 
		GaussianBlur(abs_grad_y_sq, abs_grad_y_sq_gaussian, Size(3, 3), 2, 0, BORDER_DEFAULT);
		GaussianBlur(abs_grad_xy, abs_grad_xy_gaussian, Size(3, 3), 2, 0, BORDER_DEFAULT);

		namedWindow("X-derivative_sq_gaussian", WINDOW_AUTOSIZE);
		imshow("X-derivative_sq_gaussian", abs_grad_x_sq_gaussian);
		namedWindow("Y-derivative_sq_gaussian", WINDOW_AUTOSIZE);
		imshow("Y-derivative_sq_gaussian", abs_grad_y_sq_gaussian);
		namedWindow("XY-derivative_sq_gaussian", WINDOW_AUTOSIZE);
		imshow("XY-derivative_sq_gaussian", abs_grad_xy_gaussian);
		waitKey(0);

		//Computing harris convariance matrix
		Mat H;
		Mat HD;
		Mat H_F;
		Scalar trace_H;
		H = abs_grad_x_sq_gaussian + abs_grad_y_sq_gaussian + abs_grad_xy_gaussian;
		namedWindow("HarrisConvarianceMatrix", WINDOW_AUTOSIZE);
		imshow("HarrisConvarianceMatrix", H);
		waitKey(0);
		Mat Ix_sqIy_sq, IxyIxy;
		multiply(abs_grad_x_sq_gaussian, abs_grad_y_sq_gaussian, Ix_sqIy_sq);
		pow(abs_grad_xy_gaussian, 2, IxyIxy);
		HD = Ix_sqIy_sq - IxyIxy;

		//cout << HD << endl;
		//imshow("HD", HD);
		
		trace_H = trace(H);
		//cout << trace_H << endl;
		//waitKey(0);
		divide(trace_H, HD, H_F);
		//cout << H_F << endl;
		imshow("H_F", H_F);
		waitKey(0);

		//normalizing result from 0 to 255
		Mat H_F_norm, H_F_norm_scaled;
		normalize(H_F, H_F_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(H_F_norm, H_F_norm_scaled);

		float threshold = 250;

		for (int i = 0; i < H_F_norm.rows; i++)
		{
			for (int j = 0; j < H_F_norm.cols; j++)
			{
				if ((int)H_F_norm.at<float>(i,j) > threshold)
				{
					circle(src, Point(j, i), 5, Scalar(-1), 1, 8, 0);
					
				}
			}
		}
		namedWindow("DetectedPoints", WINDOW_AUTOSIZE);  //display detected points
		imshow("DetectedPoints", src);
		waitKey(0);

		
		//Matching interest points
		Mat src2 = imread("Rainier2.png", WINDOW_AUTOSIZE);
		Mat target = src;
		vector<KeyPoint>keypoints1, keypoints2;
		FastFeatureDetector fastDet(80);
		fastDet.detect(src, keypoints1);
		fastDet.detect(src2, keypoints2);

		int size = 11;
		Rect neighborhood(0, 0, size, size);
		Mat patch1, patch2;
		Mat result;
		vector< DMatch > matches;
		vector< DMatch > bestMatch;
		for (int i = 0; i < keypoints1.size(); i++)
		{
			neighborhood.x = keypoints1[i].pt.x - size / 2;
			neighborhood.y = keypoints1[i].pt.y - size / 2;


			if (neighborhood.x < 0 || neighborhood.y < 0 || neighborhood.y + size >= src.cols || neighborhood.y + size >= src.rows)
			{
				continue;
			}
			patch1 = src(neighborhood);
			
			for (int j = 0; j < keypoints2.size(); j++)
			{
				neighborhood.x = keypoints2[j].pt.x - size / 2;
				neighborhood.y = keypoints2[j].pt.x - size / 2;


				if (neighborhood.x < 0 || neighborhood.y < 0 || neighborhood.x + size >= src2.cols || neighborhood.y + size >= src2.rows)
				{
					continue;
				}
				patch2 = src2(neighborhood);

				matchTemplate(patch1, patch2, result, CV_TM_SQDIFF_NORMED);
				if (result.at<float>(i,j) < bestMatch[i].distance)
				{
					bestMatch[i].distance = result.at<float>(0, 0);
					bestMatch[i].queryIdx = i;
					bestMatch[j].trainIdx = j;
				}

			}
			matches.push_back(bestMatch[i]);
		}

		std::nth_element(matches.begin(), matches.begin() + 25, matches.end());
		matches.erase(matches.begin() + 25, matches.end());

		Mat roi(src2, Rect(0, 0, src2.cols, src2.rows / 2));
		matchTemplate( roi, target, result, CV_TM_SQDIFF );
		double minVal, maxVal;
		Point minPt, maxPt;
		minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);

		Mat matched;
		drawMatches(src, keypoints1, src2, keypoints2, bestMatch, matched, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
		namedWindow("Matched", WINDOW_AUTOSIZE);
		imshow("Matched", matched);
		waitKey(0);
		
}
