/*
    Author: Meet Maniar
*/

/* PENDING */
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

using namespace std;
using namespace cv;

int main()
{
	Mat source1 = imread("Rainier1.png", WINDOW_AUTOSIZE);
	namedWindow("source1", WINDOW_AUTOSIZE);
	imshow("source1", source1);

	Mat source2 = imread("Rainier2.png", WINDOW_AUTOSIZE);
	namedWindow("source2", WINDOW_AUTOSIZE);
	imshow("source2", source2);

	waitKey(0);

	Mat 1_gray, 2_gray;
	cvtColor(source1, 1_gray, BGR2GRAY);
	cvtColor(source2, 2_gray, BGR2GRAY);

	int minHessian = 400;
	OrbFeatureDetector detector(minHessian);

	vector< KeyPoint >keypoints1, keypoints2;

	detector.detect(1_gray, keypoints1);
	detector.detect(2_gray, keypoints2);

	OrbDescriptorExtractor extractor;
	Mat des_obj, des_sc;
	extractor.compute(1_gray, keypoints1, des_obj);
	extractor.compute(2_gray, keypoints2, des_sc);

	BFMatcher matcher(NORM_HAMMING, false);
	vector< DMatch > matches;
	matcher.match(des_obj, des_sc, matches);
	double max_dist = 0; double min_dist = 100;

	//distance between keypoints----with ORB it works bettwe without it
	/*for (int i = 0; i < des_obj.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
		if (dist > max_dist)
		{
			max_dist = dist;
		}
	}*/

	vector< DMatch > good_matches;
	for (int j = 0; j < des_obj.rows; j++)
	{
		if (matches[i].distance <= 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	vector<Point2f> obj;
	vector<Point2f>scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

}
