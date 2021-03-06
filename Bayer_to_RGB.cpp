/*
	Author: Meet Maniar 
*/
#include<stdio.h>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat Org;
	Org = imread("../data/oldwell.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", Org);

	waitKey(0);

	Mat channels[3], R, G, B;
	split(Org, channels);
	R = channels[0];
	G = channels[1];
	B = channels[2];
	namedWindow("BLUE", WINDOW_AUTOSIZE);
	imshow("BLUE", channels[0]);
	waitKey(0);
	namedWindow("GREEN", WINDOW_AUTOSIZE);
	waitKey(0);
	imshow("GREEN", channels[1]);
	waitKey(0);
	namedWindow("RED", WINDOW_AUTOSIZE);
	imshow("RED", channels[2]);

	waitKey(0);

	Mat mix;
	merge(channels, 3, mix);
	namedWindow("Output", WINDOW_AUTOSIZE);
	imshow("Output", mix);


	waitKey(0);
	//PART-2


	Mat RG;
	subtract(R, G, RG);
	namedWindow("R-G", WINDOW_AUTOSIZE);
	imshow("R-G", RG);
	waitKey(0);

	Mat BG;
	subtract(B, G, BG);
	namedWindow("B-G", WINDOW_AUTOSIZE);
	imshow("B-G", BG);


	waitKey(0);
	Mat Median_RG, Median_BG;

	for (int o = 1; o < 37; o = o + 2)
	{
		medianBlur(RG, Median_RG, o);
	}

	namedWindow("MEDIAN_RG", WINDOW_AUTOSIZE);
	imshow("MEDIAN_RG", Median_RG);

	waitKey(0);

	for (int p = 1; p < 37; p = p + 2)
	{
		medianBlur(BG, Median_BG, p);
	}

	namedWindow("MEDIAN_BG", WINDOW_AUTOSIZE);
	imshow("MEDIAN_BG", Median_BG);

	waitKey(0);

	Mat Final_RG;
	add(Median_RG, G, Final_RG);
	namedWindow("FINAL_RG", WINDOW_AUTOSIZE);
	imshow("FINAL_RG", Final_RG);

	waitKey(0);

	Mat Final_BG;
	add(Median_BG, G, Final_BG);
	namedWindow("FINAL_BG", WINDOW_AUTOSIZE);
	imshow("FINAL_BG", Final_BG);

	waitKey(0);
	/*Mat mix2;
	merge(Final_RG, Final_BG, mix2);
	namedWindow("Final", WINDOW_AUTOSIZE);
	imshow("Final", mix2);
	waitKey(0);*/
}
