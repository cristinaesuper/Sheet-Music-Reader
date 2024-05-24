// OpenCVApplication.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <random> 
#include <windows.h>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

#define FULL_NOTE 0
#define HALF_NOTE 1

vector<float> allStaff;
map<int, double> noteFrequencyMap;
float note_height;

void playNote(int frequency, int duration) {
	Beep(frequency, duration);
}

void createNoteFrequencyMap() {
	/*noteFrequencyMap["re"] = 293.66;
	noteFrequencyMap["mi"] = 329.63;
	noteFrequencyMap["fa"] = 349.23;
	noteFrequencyMap["sol"] = 392.00;
	noteFrequencyMap["la"] = 440.00;
	noteFrequencyMap["si"] = 493.88;
	noteFrequencyMap["do2"] = 523.25;
	noteFrequencyMap["re2"] = 587.33;
	noteFrequencyMap["mi2"] = 659.25;
	noteFrequencyMap["fa2"] = 698.46;
	noteFrequencyMap["sol2"] = 783.99;*/

	noteFrequencyMap[10] = 293.66;
	noteFrequencyMap[9] = 329.63;
	noteFrequencyMap[8] = 349.23;
	noteFrequencyMap[7] = 392.00;
	noteFrequencyMap[6] = 440.00;
	noteFrequencyMap[5] = 493.88;
	noteFrequencyMap[4] = 523.25;
	noteFrequencyMap[3] = 587.33;
	noteFrequencyMap[2] = 659.25;
	noteFrequencyMap[1] = 698.46;
	noteFrequencyMap[0] = 783.99;
}

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
				//cout << center.first << ", " << center.second << endl;
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

Mat_<uchar> fillEmptyNotes(Mat_<uchar> src) {
	Mat_<uchar> filled_image(src.size());
	Mat_<uchar> dst(src.size());

	Mat_<uchar> kernel1(7, 7);			// era 6 inaite
	kernel1 = (Mat_<uchar>(7, 7) <<
		1, 1, 0, 0, 0, 1, 1,
		1, 0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 1,
		1, 1, 0, 0, 0, 1, 1);

	/*Mat_<uchar> kernel2(6, 6);
	kernel2 = (Mat_<uchar>(6, 6) <<
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0);*/

	Mat_<uchar> kernel2(7, 7);
	kernel2 = (Mat_<uchar>(7, 7) <<
		1, 0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 1);

	filled_image = dilatare(src, kernel1);
	filled_image = eroziune(filled_image, kernel2);

	dst = dilatare(src, kernel1);
	dst = eroziune(dst, kernel2);

	return dst;
}

void removeVerticalLines(Mat_<uchar>& src) {
	Mat_<uchar> kernel1(1, 3);
	kernel1.setTo(0);

	src = eroziune(src, kernel1);

	Mat_<uchar> kernel2(2, 3);
	kernel2.setTo(0);

	src = dilatare(src, kernel2);
}

void findPitchOfNote(pair<float, float> coords, int noteType) {
	float noteY = coords.first;
	float minDistance = INT_MAX;
	int nearestStaffLine = 0;

	for (int i = 0; i < allStaff.size(); i++) {
		double distance = abs(noteY - allStaff[i]);

		if (distance < minDistance) {
			minDistance = distance;
			nearestStaffLine = i;
		}
	}

	cout << "Nearest staff line: " << nearestStaffLine << endl;

	if (noteType == FULL_NOTE) {
		playNote(noteFrequencyMap[nearestStaffLine], 800);
	}
	else {
		playNote(noteFrequencyMap[nearestStaffLine], 1600);
	}
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

	cout << "arie" << arie << endl;
	cout << "rb" << rb << endl;
	cout << "cb" << cb << endl;

	if (phi > -0.77 && phi < -0.45) {
		if (arie > 400) {
			findPitchOfNote({ rb, cb }, FULL_NOTE);
		}
		else {
			findPitchOfNote({ rb, cb }, HALF_NOTE);
		}
	}

	/*** Elongatia ***/
	int r_min = INT_MAX;
	int r_max = 0;
	int c_min = INT_MAX;
	int c_max = 0;

	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			if (src(r, c) == 255) {
				if (r > r_max) {
					r_max = r;
				}
				if (r < r_min) {
					r_min = r;
				}
				if (c > c_max) {
					c_max = c;
				}
				if (c < c_min) {
					c_min = c;
				}
			}
		}
	}

	cout << "Latime: " << c_max - c_min << endl;
	cout << "Lungime: " << r_max - r_min << endl;
}

Mat_<Vec3b> transformUToV(Mat_<uchar> img) {
	Mat_<Vec3b> new_img(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar value = img(i, j);
			new_img(i, j) = Vec3b(value, value, value);
		}
	}

	return new_img;
}

Mat_<Vec3b> slideKernelAndSing(Mat_<Vec3b> initial, Mat_<uchar> img, Mat_<uchar> filled_image) {
	int kernel_height = static_cast<int>(note_height) + 2; 
	int kernel_width = 26; 

	int total_pixels = kernel_height * kernel_width;

	cout << "kernel_height: " << kernel_height << endl;

	Mat_<Vec3b> visualization = initial.clone();

	set<std::pair<int, int>> detected_pixels;

	for (int j = 0; j <= filled_image.cols - kernel_width; j++) {
		for (int i = 0; i <= filled_image.rows - kernel_height; i++) {

			Mat kernel_filled = filled_image(Rect(j, i, kernel_width, kernel_height));
			Mat kernel_unfilled = img(Rect(j, i, kernel_width, kernel_height));
			int white_pixels_filled = countNonZero(kernel_filled);
			int white_pixels_unfilled = countNonZero(kernel_unfilled);

			if (white_pixels_filled > 0.75 * total_pixels) { // simple2 - 75, simple - 

				if (detected_pixels.find(std::make_pair(j, i)) == detected_pixels.end()) {
					cout << "Note found at (" << j << ", " << i << ") - Sing the note!" << endl;

					double numarator = 0;
					double numitor = 0;

					int arie = 0;
					float rb = 0;
					float cb = 0;

					for (int r = 0; r < kernel_filled.rows; r++) {
						for (int c = 0; c < kernel_filled.cols; c++) {
							int i2 = i + r - kernel_filled.rows / 2;
							int j2 = j + c - kernel_filled.cols / 2;

							if (isInside(filled_image, i2, j2)) {
								detected_pixels.insert(std::make_pair(j2, i2));

								if (filled_image(i2, j2) == 255) {
									arie += 1;
									rb += i2;
									cb += j2;
								}
							}
						}
					}

					rb /= arie;
					cb /= arie;

					for (int r = 0; r < kernel_filled.rows; r++) {
						for (int c = 0; c < kernel_filled.cols; c++) {
							int i2 = i + r - kernel_filled.rows / 2;
							int j2 = j + c - kernel_filled.cols / 2;

							if (isInside(filled_image, i2, j2)) {
								if (filled_image(i2, j2) == 255) {
									numarator += (i2 - rb) * (c - cb);
									numitor += ((j2 - cb) * (j2 - cb)) - ((i2 - rb) * (i2 - rb));
								}
							}
						}
					}

					numarator *= 2;

					double phi = atan2(numarator, numitor) / 2;

					cout << "Phi: " << phi << endl;

					if (phi > -1 && phi < -0.45) {
						if (white_pixels_unfilled < white_pixels_filled) {
							rectangle(visualization, Point(j, i), Point(j + kernel_width, i + kernel_height), Scalar(0, 255, 0), 2);
							findPitchOfNote({ rb, cb }, HALF_NOTE);
						}
						else {
							rectangle(visualization, Point(j, i), Point(j + kernel_width, i + kernel_height), Scalar(0, 0, 255), 2);
							findPitchOfNote({ rb, cb }, FULL_NOTE);
						}
					}
				}
			}
		}
	}

	return visualization;
}

Mat_<Vec3b> getResult(Mat_<uchar> initial, Mat_<uchar> src) {
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

	dst = eroziune(src, kernel1);
	dst = dilatare(dst, kernel2);

	auto filled_image = fillEmptyNotes(dst);

	removeVerticalLines(dst);
	removeVerticalLines(filled_image);

	//imshow("dst", dst);
	//imshow("filled_image", filled_image);

	Mat_<Vec3b> initialV = transformUToV(initial);

	Mat_<Vec3b> recognized = slideKernelAndSing(initialV, dst, filled_image);

	Mat_<Vec3b> coloured_img = lab_5_bfs_labeling(dst);

	imshow("recognized", recognized);

	return coloured_img;
}

void storePitches(Mat_<uchar> img) {
	vector<int> staffLines(img.rows);
	vector<float> staffLinesPairs;

	std::fill(staffLines.begin(), staffLines.end(), 0);
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 255) {
				staffLines[i]++;
			}
		}
	}

	cout << "Array-ul de staff lines: " << endl;

	for (int i = 0; i < staffLines.size(); i++) {
		cout << staffLines[i] << endl;
	}

	for (int i = 0; i < staffLines.size() - 1; i++) {
		if (staffLines[i] > 0 && staffLines[i+1] > 0) {
			staffLinesPairs.push_back((float)(2 * i + 1) / 2);
		}
	}

	float dist = (staffLinesPairs[1] - staffLinesPairs[0]) / 2;

	allStaff.push_back(staffLinesPairs[0] - dist);	// pt. sol de sus

	for (int i = 0; i < staffLinesPairs.size() - 1; i++) {
		allStaff.push_back(staffLinesPairs[i]);
		float intermediate = (staffLinesPairs[i] + staffLinesPairs[i + 1]) / 2;
		allStaff.push_back(intermediate);
	}

	allStaff.push_back(staffLinesPairs[staffLinesPairs.size() - 1]);
	allStaff.push_back(staffLinesPairs[staffLinesPairs.size() - 1] + dist);	// pt. re de jos

	cout << "Array-ul de linii, spatii intre linii: " << endl;

	for (int i = 0; i < allStaff.size(); i++) {
		cout << allStaff[i] << endl;
	}

	note_height = dist * 2;
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
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

	createNoteFrequencyMap();

	Mat_<uchar> img = imread("Images/row_no_key.png", IMREAD_GRAYSCALE);

	Mat_<uchar> inv_img = inverse(img);

	Mat_<uchar> kernel(3, 3);
	kernel.setTo(0);

	Mat_<uchar> new_img_only_staff = removeStaffLines(inv_img);
	Mat_<uchar> new_img_staff_removed = inv_img - new_img_only_staff;
	Mat_<Vec3b> result = getResult(inv_img, new_img_staff_removed);

	//imshow("before", img);
	//imshow("inv", inv_img);
	imshow("only_staff", new_img_only_staff);
	imshow("result", result);

	setMouseCallback("result", onMouse, &result);

	waitKey();

	return 0;
}