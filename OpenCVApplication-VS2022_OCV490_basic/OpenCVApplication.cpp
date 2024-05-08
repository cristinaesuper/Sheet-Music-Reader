// OpenCVApplication.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <random> 

using namespace std;
using namespace cv;

bool isInside(Mat img, int i, int j) {
	return i >= 0 && i < img.rows&& j >= 0 && j < img.cols;
}

Mat_<Vec3b> label2color(Mat_<int> labels, int label) {
	default_random_engine gen;
	uniform_int_distribution<int> d(0, 255);
	vector<Vec3b> colors(label + 1);
	Mat_<Vec3b> coloredImg(labels.rows, labels.cols);

	for (int i = 0; i < label + 1; i++) {
		if (i == 0) {
			colors[i] = Vec3b(255, 255, 255);
		}
		else {
			colors[i] = Vec3b(d(gen), d(gen), d(gen));
		}
	}

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int label = labels(i, j);
			coloredImg(i, j) = colors[label];
		}
	}

	return coloredImg;
}

pair<float, float> computeCenterOfMass(Mat_<int>& labels, int label) {
	int arie = 0;
	float rb = 0;
	float cb = 0;

	for (int r = 0; r < labels.rows; r++) {
		for (int c = 0; c < labels.cols; c++) {
			if (labels(r, c) == label) {
				arie += 1;
				rb += r;
				cb += c;
			}
		}
	}

	rb /= arie;
	cb /= arie;

	return { rb, cb };
}

Mat_<Vec3b> lab_5_bfs_labeling(Mat_<uchar> img) {
	int label = 0;	// eticheta curenta == nr. de etichere
	Mat_<int> labels(img.rows, img.cols);
	vector<pair<float, float>> centers;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			labels(i, j) = 0;
		}
	}

	int di[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int dj[] = { -1, 0, 1, 1, 1, 0, -1, -1 };

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (img(i, j) == 255 && labels(i, j) == 0) {
				label++;
				queue<pair<int, int>> Q;
				labels(i, j) = label;
				Q.push({ i, j });

				while (!Q.empty()) {
					auto q = Q.front();
					Q.pop();

					for (int k = 0; k < 8; k++) {

						int i2 = q.first + di[k];
						int j2 = q.second + dj[k];

						if (isInside(img, i2, j2) == true) {
							if (img(i2, j2) == 255 && labels(i2, j2) == 0) {
								labels(i2, j2) = label;
								Q.push({ i2, j2 });
							}
						}
					}
				}

				auto center = computeCenterOfMass(labels, label);
				cout << center.first << ", " << center.second << endl;
				centers.push_back(center);
			}
		}
	}

	auto colors = label2color(labels, label);
	return colors;
}

Mat_<uchar> dilatare(Mat_<uchar> src, Mat_<uchar> elstr) {
	Mat_<uchar> dst(src.size());
	dst.setTo(0);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			if (src(i, j) == 255) {
				for (int u = 0; u < elstr.rows; u++) {
					for (int v = 0; v < elstr.cols; v++) {

						if (elstr(u, v) == 0) {
							int i2 = i + u - elstr.rows / 2;
							int j2 = j + v - elstr.cols / 2;

							if (isInside(src, i2, j2)) {
								dst(i2, j2) = 255;
							}
						}
					}
				}
			}
		}
	}

	return dst;
}

Mat_<uchar> eroziune(Mat_<uchar> src, Mat_<uchar> elstr) {
	Mat_<uchar> dst(src.size());
	dst.setTo(0);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == 255) {
				bool onlyObjects = true;

				for (int u = 0; u < elstr.rows; u++) {
					for (int v = 0; v < elstr.cols; v++) {
						if (elstr(u, v) == 0) {
							int i2 = i + u - elstr.rows / 2;
							int j2 = j + v - elstr.cols / 2;

							if (isInside(src, i2, j2) && src(i2, j2) != 255) {
								onlyObjects = false;
							}
						}
					}
				}

				if (onlyObjects) {
					dst(i, j) = 255;
				}
				else {
					dst(i, j) = 0;
				}
			}
		}
	}

	return dst;
}

void fillEmptyNotes(Mat_<uchar>& src) {
	Mat_<uchar> filled_image(src.size());

	Mat_<uchar> kernel1(6, 6);
	kernel1 = (Mat_<uchar>(6, 6) <<
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0);

	Mat_<uchar> kernel2(6, 6);
	kernel2 = (Mat_<uchar>(6, 6) <<
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0);

	filled_image = dilatare(src, kernel1);
	filled_image = eroziune(filled_image, kernel2);

	filled_image = filled_image - src;

	imshow("filled", filled_image);
}

void removeVerticalLines(Mat_<uchar>& src) {
	Mat_<uchar> kernel1(1, 3);
	kernel1.setTo(0);

	src = eroziune(src, kernel1);

	Mat_<uchar> kernel2(2, 3);
	kernel2.setTo(0);

	src = dilatare(src, kernel2);
}

void axaAlungire(Mat_<uchar>& src) {
	double numarator = 0;
	double numitor = 0;

	int arie = 0;
	float rb = 0;
	float cb = 0;

	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			if (src(r, c) == 255) {
				arie += 1;
				rb += r;
				cb += c;
			}
		}
	}

	rb /= arie;
	cb /= arie;

	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			if (src(r, c) == 255) {
				numarator += (r - rb) * (c - cb);
				numitor += (c - cb) * (c - cb) - (r - rb) * (r - rb);
			}
		}
	}

	numarator *= 2;

	double phi = atan2(numarator, numitor) / 2;

	cout << phi;
}

Mat_<Vec3b> getResult(Mat_<uchar> src) {
	Mat_<uchar> dst(src.size());

	Mat_<uchar> kernel1(2, 2);			// cred ca era 3 si 5 inainte. sau 4 si 5 si kernel-urile cu 1 in colturi
	kernel1 = (Mat_<uchar>(2, 2) <<
		0, 0, 
		0, 0);

	Mat_<uchar> kernel2(4, 4);
	kernel2 = (Mat_<uchar>(4, 4) <<
		1, 0, 0, 1, 
		1, 0, 0, 1, 
		1, 0, 0, 1, 
		1, 0, 0, 1);

	cout << kernel2;

	dst = eroziune(src, kernel1);
	dst = dilatare(dst, kernel2);

	fillEmptyNotes(dst);
	removeVerticalLines(dst);

	//axaAlungire(dst);

	Mat_<Vec3b> coloured_img = lab_5_bfs_labeling(dst);

	return coloured_img;
}

void storePitches(Mat_<uchar> img) {
	vector<int> staffLines(img.rows);

	for (int i = 0; i < staffLines.size(); i++) {
		staffLines[i] = 0;
	}

	// Cum fac sa adun toti pixelii de pe o linie (pt. ca o linie de portativ e 2 linii din imagine)? 
	// Fac cu elemente conexe sau harcodez?
	// R: sa fac media randurilor una dupa alta care sunt diferite de 0 ca sa vad unde e mijlocul liniei de portativ (poate fi float, cum am aici daor doua randuri)
	
	
	// La doime, ca sa vad care e si care nu e doime, merg un un el. structural patrat si vad daca 80% e alb, atunci e patrime
	// daca mai putin de 80%, atunci e doime.

	// Vr. sa incerc si cu scadere sa vad ce iese
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 255) {
				staffLines[i]++;
			}
		}
	}

	for (int i = 0; i < staffLines.size(); i++) {
		if (staffLines[i] > 0) {
			cout << i << endl;
		}

		//cout << staffLines[i] << endl;
	}
}

Mat_<uchar> removeStaffLines(Mat_<uchar> src) {
	Mat_<uchar> dst(src.size());

	Mat_<uchar> kernel(5, 33);
	kernel.setTo(1);

	for (int j = 0; j < kernel.cols; j++) {
		kernel(2, j) = 0;
	}

	dst = eroziune(src, kernel);

	storePitches(dst);
	
	return dst;
}

Mat_<uchar> inverse(Mat_<uchar> img) {
	Mat_<uchar> inv_img(img.rows, img.cols);
	inv_img = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) < 100) {
				inv_img(i, j) = 255;
			}
			else {
				inv_img(i, j) = 0;
			}
		}
	}

	return inv_img;
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN) {
		Mat_<Vec3b> img = *(Mat_<Vec3b> *)param;
		Vec3b color = img(y, x);
		cout << color << endl;

		Mat_<uchar> gray(img.rows, img.cols);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				gray(i, j) = img(i, j) == color ? 255 : 0;
			}
		}

		axaAlungire(gray);
	}
}

int main() {
	Mat_<uchar> img = imread("Images/simple.png", IMREAD_GRAYSCALE);

	Mat_<uchar> inv_img = inverse(img);

	Mat_<uchar> kernel(3, 3);
	kernel.setTo(0);

	Mat_<uchar> new_img_only_staff = removeStaffLines(inv_img);
	Mat_<uchar> new_img_staff_removed = inv_img - new_img_only_staff;
	Mat_<Vec3b> result = getResult(new_img_staff_removed);

	//imshow("before", img);
	imshow("inv", inv_img);
	imshow("only_staff", new_img_only_staff);
	imshow("result", result);

	setMouseCallback("result", onMouse, &result);

	waitKey();

	return 0;
}