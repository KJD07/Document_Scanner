#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/objdetect.hpp>


using namespace cv;
using namespace std;

Mat picback;
Mat camvid;
Mat warppic;
int w = 420, h = 596;

Mat preProcessing(Mat pic) {
	Mat opic, hlspic, gbpic, canpic, dilatepic, erodepic;

	cvtColor(pic, opic, COLOR_BGR2GRAY);
	
	GaussianBlur(pic, gbpic, Size(5, 5), 3, 0);
	Canny(pic, canpic, 30, 80);

	Mat K = getStructuringElement(MORPH_RECT, Size(10, 10));
	dilate(canpic, dilatepic, K);
	return dilatepic;
}

vector<Point> cofunction(Mat picback) {

	vector<vector<Point>> colet;
	vector<Vec4i> neon;

	findContours(picback, colet, neon, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> arr(colet.size());
	vector<Point> biggest(arr.size());
	vector<Rect> faces(colet.size());

	float AREA = 0.00;

	for (int i = 0; i < colet.size(); i++) {

		float area = contourArea(colet[i]);
		float perimeter = arcLength(colet[i], true);
		approxPolyDP(colet[i], arr[i], 0.02 * perimeter, true); // In this if we change the value with perimeter then the no. of arc length will increase



		if (area > AREA && area > 1000 && arr[i].size() == 4) {

			AREA = max(area, AREA);

			biggest = { arr[i][0],arr[i][1],arr[i][2],arr[i][3] };
		}
	}
	return biggest;
}

vector<Point> reorder(vector<Point> biggest) {
	vector<Point> neworder;
	vector<int> sum, sub;

	for (int i = 0; i < biggest.size(); i++) {
		sum.push_back(biggest[i].x + biggest[i].y);
		sub.push_back(biggest[i].x - biggest[i].y);
	}
	//min_element and max_element are use find the index of the sequence
	neworder.push_back(biggest[min_element(sum.begin(), sum.end()) - sum.begin()]);
	neworder.push_back(biggest[min_element(sub.begin(), sub.end()) - sub.begin()]);
	neworder.push_back(biggest[max_element(sub.begin(), sub.end()) - sub.begin()]);
	neworder.push_back(biggest[max_element(sum.begin(), sum.end()) - sum.begin()]);

	return neworder;
}

vector<Point> initialPoints, newpoints;

void drawPoints(vector<Point> Points, Scalar color) {
	for (int i = 0; i < Points.size(); i++) {
		circle(camvid, Points[i], 5, color, FILLED);
	}
}

Mat warp(vector<Point> newpoints, Mat camvid, float w, float h) {
	Point2f cards_in[4] = { newpoints[1],newpoints[0], newpoints[2], newpoints[3] };
	Point2f cards_ot[4] = { {0.0f,h},{0.0f,0.0f},{w,0.0f},{w,h} };
	Mat matrix = getPerspectiveTransform(cards_in, cards_ot);
	warpPerspective(camvid, warppic, matrix, Point(w, h));
	return warppic;
}

void main() {

	string location = "Resources/paper.jpg";
	camvid = imread(location);

	resize(camvid, camvid, Size(), 0.5, 0.5);


	picback = preProcessing(camvid);

	initialPoints = cofunction(picback);

	newpoints = reorder(initialPoints);

	warppic = warp(newpoints, camvid, w, h);

	cvtColor(warppic, warppic, COLOR_BGR2GRAY);

	imshow("Video", warppic);
	imshow("Video1", camvid);
	waitKey(0);

}