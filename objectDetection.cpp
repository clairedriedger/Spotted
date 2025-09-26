#include "Timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>

void printUsage() {
	std::cout << "Usage: " << std::endl;
	std::cout << " ObjectDetector <object image> <scene image> <method>" << std::endl;
	std::cout << " <object image> an image of the object to be detected" << std::endl;
	std::cout << " <scene image> an image of a scene to search for the object" << std::endl;
	std::cout << " <method>  SIFT or ORB detection" << std::endl;
	std::cout << " e.g.: ObjectDetector object.png scene.png SIFT" << std::endl;
}

std::string toLower(const std::string& str) {
	std::string result = str;
	std::transform(result.begin(), result.end(), result.begin(),
		[](unsigned char c) { return std::tolower(c); });

	return result;
}

int main(int argc, char* argv[]) {

	if (argc != 4) {
		printUsage();
		exit(-1);
	}

	cv::Mat objImage = cv::imread(argv[1]);
	if (objImage.empty()) {
		std::cerr << "Failed to read image from " << argv[1] << std::endl;
		exit(-2);
	}

	cv::Mat scnImage = cv::imread(argv[2]);
	if (scnImage.empty()) {
		std::cerr << "Failed to read image from " << argv[2] << std::endl;
		exit(-3);
	}

	std::string method = toLower(argv[3]);

	if (method != "sift" && method != "orb") {
		std::cerr << "Invalid method '" << argv[3] << "'" << std::endl;
		exit(-4);
	}

	///////////////////////////////////////////////////////
	// Code goes here to detect the object in the scene  //
	// You should then draw a box around the object in   //
	// detImage, which has been initialised to be a copy //
	// of the scene.                                     //
	///////////////////////////////////////////////////////
	
	// Uncomment to display the images
	/*
	cv::namedWindow("Object Image");
	cv::imshow("Object Image", objImage);
	cv::namedWindow("Scene Image");
	cv::imshow("Scene Image", scnImage);
	cv::waitKey();
	cv::destroyWindow("Object Image");
	cv::destroyWindow("Scene Image");
	*/

	cv::Ptr<cv::FeatureDetector> detector_obj;
	cv::Ptr<cv::FeatureDetector> detector_scn;
	cv::Ptr<cv::DescriptorMatcher> matcher;


	if (method == "orb") { // use ORB and BF matcher with HAMMING distance
		detector_obj = cv::ORB::create(); 
		detector_scn = cv::ORB::create(10000);
		matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	}
	else { // use SIFT and FLANN matcher
		detector_obj = cv::SIFT::create();
		detector_scn = cv::SIFT::create();
		matcher = cv::FlannBasedMatcher::create();
	}

	// 1. Detect, draw and display features in each image

	// 1.1 Object image
	std::vector<cv::KeyPoint> keypoints_obj;
	cv::Mat descriptors_obj;

	Timer timer1;

	detector_obj->detectAndCompute(objImage, cv::noArray(), keypoints_obj, descriptors_obj);

	double time_detectObj = timer1.elapsed();

	cv::Mat kptImage_obj; 
	cv::drawKeypoints(objImage, keypoints_obj, kptImage_obj, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 
	
	// cv::imshow("Object Features", kptImage_obj);

	// 1.2 Scene image
	std::vector<cv::KeyPoint> keypoints_scn;
	cv::Mat descriptors_scn;
	
	Timer timer2;

	detector_scn->detectAndCompute(scnImage, cv::noArray(), keypoints_scn, descriptors_scn);

	double time_detectScn = timer2.elapsed();

	cv::Mat kptImage_scn;
	cv::drawKeypoints(scnImage, keypoints_scn, kptImage_scn, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// cv::imshow("Scene Features", kptImage_scn);


	// cv::waitKey();
	// cv::destroyWindow("Object Features");
	// cv::destroyWindow("Scene Features");

	// 2. Match features between images, filtering for reliable matches
	std::vector<std::vector<cv::DMatch>> matches;

	Timer timer3;

	matcher->knnMatch(descriptors_obj, descriptors_scn, matches, 2);

	double time_match = timer3.elapsed();

	std::vector<cv::Point2f> scn_pts, obj_pts;
	std::vector<cv::DMatch> goodMatches;
	for (const auto& match : matches) {
		if (match[0].distance < 0.8 * match[1].distance) {
			goodMatches.push_back(match[0]);
			// matching the object (query) TO the scene (train)
			obj_pts.push_back(keypoints_obj[match[0].queryIdx].pt); 
			scn_pts.push_back(keypoints_scn[match[0].trainIdx].pt);
		}
	}

	// 3. Draw the matches
	cv::Mat matchImgORB;
	cv::drawMatches(objImage, keypoints_obj, scnImage, keypoints_scn, goodMatches, matchImgORB);
	/*
	if (method == "orb") {
		cv::namedWindow("Good Matches with ORB and B.F.");
		cv::imshow("Good Matches with ORB and B.F.", matchImgORB);
		cv::waitKey();
		cv::destroyWindow("Good Matches with ORB and B.F.");
	}
	else { 
		cv::namedWindow("Good Matches with SIFT and FLANN");
		cv::imshow("Good Matches with SIFT and FLANN", matchImgORB);
		cv::waitKey();
		cv::destroyWindow("Good Matches with SIFT and FLANN");
	}
	*/

	std::vector<unsigned char> inliers;
	cv::Mat H = cv::findHomography(obj_pts, scn_pts, inliers, cv::RANSAC);

	// 4. Draw a bounding box around the object

	// 4.1 Define the four corners
	cv::Mat tl(3, 1, CV_64F); tl.at<double>(0, 0) = 0; tl.at<double>(1, 0) = 0; tl.at<double>(2, 0) = 1;
	cv::Mat tr(3, 1, CV_64F); tr.at<double>(0, 0) = objImage.cols; tr.at<double>(1, 0) = 0; tr.at<double>(2, 0) = 1;
	cv::Mat br(3, 1, CV_64F); br.at<double>(0, 0) = objImage.cols; br.at<double>(1, 0) = objImage.rows; br.at<double>(2, 0) = 1;
	cv::Mat bl(3, 1, CV_64F); bl.at<double>(0, 0) = 0; bl.at<double>(1, 0) = objImage.rows; bl.at<double>(2, 0) = 1;


	// 4.2 Apply the homography transformation
	cv::Mat tl_trans = H * tl;
	cv::Mat tr_trans = H * tr;
	cv::Mat br_trans = H * br;
	cv::Mat bl_trans = H * bl;

	// 4.3 Convert the homogenous form back to coordinates
	auto convertPoint = [](const cv::Mat& pt) -> cv::Point {

		int x = (int)(pt.at<double>(0) / pt.at<double>(2) + 0.5);
		int y = (int)(pt.at<double>(1) / pt.at<double>(2) + 0.5);
		return cv::Point(x, y);
	};

	cv::Point tl_conv = convertPoint(tl_trans);
	cv::Point tr_conv = convertPoint(tr_trans);
	cv::Point br_conv = convertPoint(br_trans);
	cv::Point bl_conv = convertPoint(bl_trans);

	// 4.4 Draw the box
	cv::Mat detImage = scnImage.clone();
	cv::circle(detImage, tl_conv, 6, cv::Scalar(0, 255, 0), 2);
	cv::circle(detImage, tr_conv, 6, cv::Scalar(0, 255, 0), 2);
	cv::circle(detImage, br_conv, 6, cv::Scalar(0, 255, 0), 2);
	cv::circle(detImage, bl_conv, 6, cv::Scalar(0, 255, 0), 2);

	cv::line(detImage, tl_conv, tr_conv, cv::Scalar(0, 255, 0), 1);
	cv::line(detImage, tr_conv, br_conv, cv::Scalar(0, 255, 0), 1);
	cv::line(detImage, br_conv, bl_conv, cv::Scalar(0, 255, 0), 1);
	cv::line(detImage, bl_conv, tl_conv, cv::Scalar(0, 255, 0), 1);
	

	// 5. Experimental Data

	// 5.1 Speed tests
	std::cout << "Scn detection took " << time_detectScn << " seconds" << std::endl;
	std::cout << "Matching took " << time_match << " seconds" << std::endl;


	// 5.2 Accuracy data
	std::cout << "Corner locations for " << method << " for an image of size " << scnImage.size << " are: Top left: " << tl_conv << " Top Right: " << tr_conv << " Bottom Right: " << br_conv << " Bottom Left " << bl_conv << std::endl;

	// 5.3 Saturation Data
	std::cout << "Time for scene detection and matching was: " << time_detectScn + time_match << " seconds" << std::endl;
	std::cout << "Total number of matches was: " << matches.size() << std::endl;
	std::cout << "Num good matches " << goodMatches.size() << std::endl;


	// 6. Save the detected object
	cv::imwrite("detectedObject.png", detImage);
	cv::namedWindow("Detection");
	cv::imshow("Detection", detImage);
	cv::waitKey();
	cv::destroyWindow("Detection");


	return 0;
		 
}