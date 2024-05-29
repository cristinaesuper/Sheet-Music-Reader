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

	Mat_<uchar> kernel1(7, 7);
	kernel1 = (Mat_<uchar>(7, 7) <<
		1, 1, 0, 0, 0, 1, 1,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, 1, 1);

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

	if (noteType == FULL_NOTE) {
		playNote(noteFrequencyMap[nearestStaffLine], 800);
	}
	else {
		playNote(noteFrequencyMap[nearestStaffLine], 1600);
	}
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

	Mat_<Vec3b> visualization = initial.clone();

	set<pair<int, int>> detected_pixels;

	for (int j = 0; j <= filled_image.cols - kernel_width; j++) {
		for (int i = 0; i <= filled_image.rows - kernel_height; i++) {

			Mat kernel_filled = filled_image(Rect(j, i, kernel_width, kernel_height));
			Mat kernel_unfilled = img(Rect(j, i, kernel_width, kernel_height));
			int white_pixels_filled = countNonZero(kernel_filled);
			int white_pixels_unfilled = countNonZero(kernel_unfilled);

			if (white_pixels_filled > 0.75 * total_pixels) { 

				if (detected_pixels.find({i, j}) == detected_pixels.end()) {
					double numarator = 0;
					double numitor = 0;

					int arie = 0;
					float rb = 0;
					float cb = 0;

					for (int r = 0; r < kernel_filled.rows + 5; r++) {
						for (int c = 0; c < kernel_filled.cols + 5; c++) {
							int i2 = i + r - (kernel_filled.rows + 5) / 2;
							int j2 = j + c - (kernel_filled.cols + 5) / 2;

							if (isInside(filled_image, i2, j2)) {
								detected_pixels.insert({i2, j2});

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

					for (int r = 0; r < kernel_filled.rows + 5; r++) {
						for (int c = 0; c < kernel_filled.cols + 5; c++) {
							int i2 = i + r - (kernel_filled.rows + 5) / 2;
							int j2 = j + c - (kernel_filled.cols + 5) / 2;

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

					if (phi > -1 && phi < -0.43 && arie < 150) {

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

	Mat_<uchar> kernel1(2, 2);			
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

	Mat_<Vec3b> initialV = transformUToV(initial);

	Mat_<Vec3b> recognized = slideKernelAndSing(initialV, dst, filled_image);

	return recognized;
}

void storePitches(Mat_<uchar> img) {
	allStaff.clear();

	vector<int> staffLines(img.rows);
	vector<float> staffLinesPairs;

	fill(staffLines.begin(), staffLines.end(), 0);
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 255) {
				staffLines[i]++;
			}
		}
	}

	for (int i = 0; i < staffLines.size() - 1; i++) {
		if (staffLines[i] > img.cols / 2 && staffLines[i+1] > img.cols / 2) {
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

	note_height = dist * 2;
}

int padding = 30;

pair<vector<Mat_<uchar>>, vector<Mat_<uchar>>> splitImage(Mat_<uchar> img, Mat_<uchar> src) {
	vector<Mat_<uchar>> staffGroups;
	vector<Mat_<uchar>> new_srcs;
	vector<int> staffLines(img.rows);

	fill(staffLines.begin(), staffLines.end(), 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 255) {
				staffLines[i]++;
			}
		}
	}

	int threshold = img.cols / 2; 
	bool inStaffGroup = false;
	int start = 0;
	int staffLineCount = 0;

	for (int i = 0; i < img.rows; i++) {
		if (staffLines[i] > threshold) {
			if (!inStaffGroup) {
				start = i;
				inStaffGroup = true;
			}
			staffLineCount++;
		}
		else {
			if (inStaffGroup) {
				if (staffLineCount == 10) { 
					int end = i;
					inStaffGroup = false;
					staffLineCount = 0;

					start = max(0, start - padding);
					end = min(img.rows - 1, end + padding);

					Mat_<uchar> staffGroup = img(Range(start, end), Range::all()).clone();
					Mat_<uchar> src_new = src(Range(start, end), Range::all()).clone();
					staffGroups.push_back(staffGroup);
					new_srcs.push_back(src_new);
				}
			}
		}
	}

	return { staffGroups, new_srcs };
}

pair<vector<Mat_<uchar>>, vector<Mat_<uchar>>> removeStaffLines(Mat_<uchar> src) {
	Mat_<uchar> dst(src.size());

	Mat_<uchar> kernel(5, 33);
	kernel.setTo(1);

	for (int j = 0; j < kernel.cols; j++) {
		kernel(2, j) = 0;
	}

	dst = eroziune(src, kernel);

	pair<vector<Mat_<uchar>>, vector<Mat_<uchar>>> splittedImages = splitImage(dst, src);
	
	return splittedImages;
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

vector<Mat_<Vec3b>> processSplits(pair<vector<Mat_<uchar>>, vector<Mat_<uchar>>> splitted) {
	vector<Mat_<Vec3b>> results;

	for (int i = 0; i < splitted.first.size(); i++) {
		Mat_<uchar> splitted_inv_img = splitted.second[i];
		Mat_<uchar> splitted_staff = splitted.first[i];

		storePitches(splitted_staff);

		Mat_<uchar> new_img_staff_removed = splitted_inv_img - splitted_staff;

		Mat_<Vec3b> result = getResult(splitted_inv_img, new_img_staff_removed);

		results.push_back(result);
	}

	return results;
}

Mat_<Vec3b> combineResults(vector<Mat_<Vec3b>> results) {
	int height = 0;

	for (int i = 0; i < results.size(); i++) {
		height += results[i].rows;
	}

	Mat_<Vec3b> result(height, results[0].cols);
	int count = 0;

	for (int k = 0; k < results.size(); k++) {

		for (int i = 0; i < results[k].rows; i++) {
			for (int j = 0; j < results[k].cols; j++) {
				result(count + i, j) = results[k](i, j);
			}
		}

		count += results[k].rows;
	}

	return result;
}

int main() {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

	createNoteFrequencyMap();

	Mat_<uchar> img = imread("Images/sweet_all.png", IMREAD_GRAYSCALE);

	Mat_<uchar> inv_img = inverse(img);

	pair<vector<Mat_<uchar>>, vector<Mat_<uchar>>> splitted = removeStaffLines(inv_img);

	vector<Mat_<Vec3b>> results = processSplits(splitted);

	Mat_<Vec3b> result = combineResults(results);

	imshow("result", result);

	waitKey();

	return 0;
}