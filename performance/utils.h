//
// Created by ibrahim on 3/9/21.
//

#ifndef PERFORMANCE_UTILS_H
#define PERFORMANCE_UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "rapidxml-1.13/rapidxml.hpp""

#include <opencv4/opencv2/opencv.hpp>

void read_directory(const std::string& path, std::vector<std::string>& v);
std::vector<cv::Rect> readGtboxesPESMOT(std::string path);
void enlargeRect(cv::Rect &rect, int a=5, int width = 1920, int height = 1080);
void findCombinedRegions(const cv::Mat &mask, cv::Mat &maskOutput, cv::Mat &smallRegionMask, std::vector<cv::Rect> &rectangles, int minArea=10);
bool checkRectOverlap(const cv::Rect &rectGT, const cv::Rect &r, float &intersectRatio);
void compareResults(const std::vector<cv::Rect> &gtBoxes, const std::vector<cv::Rect> &bboxes,
                    int &totalGT, int &totalFound, float &totalIntersectRatio, int &totalTP, int &totalFP, int &totalTN, int &totalFN);

#endif //PERFORMANCE_UTILS_H
