/*
    Author: Meet Maniar
/*

/*
    Description: This program detects important feature between two image that are clicked from different angle and in different lights
                 and matches. In this program, there are two approaches, i.e. Sum of squared difference and Ratio test. They can be found
                 in line no. 132 and 145 respectively. Note the part that to be commeneted and un-commented before running the program
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
	Mat src1, src2;
	src1 = imread("Yosemite1.jpg", WINDOW_AUTOSIZE);
	src2 = imread("Yosemite2.jpg", WINDOW_AUTOSIZE);
	//namedWindow("source1", WINDOW_AUTOSIZE);
	//namedWindow("source2", WINDOW_AUTOSIZE);
	//imshow("source1", src1);
	//imshow("source2", src2);
	//waitKey(0);

	SiftFeatureDetector detector(450, 3, 0.06, 10, 1.6);
	SiftDescriptorExtractor extractor(3.0);



	//Detecting keypoints
	vector<KeyPoint>keypoints1, keypoints2;
	detector.detect(src1, keypoints1);
	detector.detect(src2, keypoints2);

	//size of the keypoints
	int key1 = keypoints1.size();
	int key2 = keypoints2.size();

	cout << "key1:" << key1 << endl;
	cout << "key2:" << key2 << endl;

	//Forming the descriptors
	Mat descriptor1, descriptor2;
	extractor.compute(src1, keypoints1, descriptor1);
	extractor.compute(src2, keypoints2, descriptor2);

	cout << "Descriptor1:" << descriptor1.size();
	cout << "Descriptor2:" << descriptor2.size();

	//std:cout << descriptors2.rows;
	//cout << descriptors1.size();

	Mat diff;
	Mat ssd(descriptor1.rows, descriptor1.rows, CV_32FC1);
	//cout << ssd.at<float>(0,299) << endl;
	float s = 0;


	for (int i = 0; i < descriptor1.rows; i++)
	{

		for (int j = 0; j < descriptor2.rows; j++)
		{
			diff = descriptor1.row(i) - descriptor2.row(j);
			pow(diff, 2, diff);

			for (int x = 0; x < diff.cols; x++)
			{
				s = s + diff.at<float>(0, x);
			}
			//cout << j << endl;
			ssd.at<float>(i, j) = s;
			s = 0.0;

		}

	}

	BFMatcher matcher;
	std::vector< DMatch > matches1;
	std::vector< DMatch > matches2;
	matcher.match(descriptor1, descriptor2, matches1);
	matcher.match(descriptor1, descriptor2, matches2);

	double min, max;
	int min_loc, max_loc;

	for (int i = 0; i < ssd.rows; i++)
	{
		cv::minMaxIdx(ssd.row(i), &min, &max, &min_loc, &max_loc);
		//cout << min_loc << endl;

		matches1[i].distance = min;
		matches1[i].imgIdx = 0;
		matches1[i].trainIdx = min_loc;
		matches1[i].queryIdx = i;

		matches2[i].distance = min;
		matches2[i].imgIdx = 0;
		matches2[i].trainIdx = min_loc;
		matches2[i].queryIdx = i;

	}

	/*BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);
	cout << matches.size();*/

	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(src1, keypoints1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(src2, keypoints2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);

	waitKey(0);
	//Get the min-max distance---SSD
	double max_dist = 0; double min_dist = 1000;
	for (int i = 0; i < descriptor1.rows; i++)
	{
		double dist = matches1[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	cout << "Max distance=" << max_dist << endl;
	cout << "Min distance=" << min_dist << endl;

	//Get the min-max distance---Ratio
	for (int i = 0; i < descriptor1.rows; i++)
	{
		double dist = (matches1[i].distance / matches2[i].distance);
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	cout << "Max distance=" << max_dist << endl;
	cout << "Min distance=" << min_dist << endl;

	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptor1.rows; i++)
	{
		if (matches1[i].distance <= 10 * min_dist)
		{
			good_matches.push_back(matches1[i]);
		}
	}



	//Draw good matches and show detected matches
	Mat matched;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, matched, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matched);
	waitKey(0);

	return 0;

}
