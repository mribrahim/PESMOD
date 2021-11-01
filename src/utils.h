//
// Created by ibrahim on 3/9/21.
//

#ifndef PERFORMANCE_UTILS_H
#define PERFORMANCE_UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "rapidxml-1.13/rapidxml.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <torch/torch.h>


void read_directory(const std::string& path, std::vector<std::string>& v);
void showMat(std::string text, const cv::Mat &img);
void showGPUMat(std::string window_text, const cv::cuda::GpuMat &d_matInput, const cv::cuda::GpuMat &mask, cv::Scalar factor, int channel);
void getChannel(const cv::Mat &img, uint8_t channelId, cv::Mat &channel);
void getChannel(const cv::cuda::GpuMat &img, uint8_t channelId, cv::cuda::GpuMat &channel);
void flow2MagAngle(const cv::cuda::GpuMat &d_flow, cv::cuda::GpuMat &d_magnitude, cv::cuda::GpuMat &d_angle);
void flow2img(const cv::cuda::GpuMat &flow, cv::cuda::GpuMat &img);
void maxOf3bands(const cv::cuda::GpuMat &d_diff, cv::cuda::GpuMat &d_diffMax);
void findMeanBG(const cv::cuda::GpuMat d_flow, cv::cuda::GpuMat &d_magnitude, float &meanBG, float &stdDevBG);
void affineDiff(int kernelSize, const cv::cuda::GpuMat &d_bg, const cv::cuda::GpuMat& d_frameC, cv::cuda::GpuMat &result);
void applyOpening(const cv::cuda::GpuMat &maskInput, cv::cuda::GpuMat &maskOutput);

std::vector<cv::Rect> readGtboxesPESMOT(std::string path);
void enlargeRect(cv::Rect &rect, int a=5);
void findCombinedRegions(const cv::Mat &mask, cv::Mat &maskOutput, cv::Mat &smallRegionMask, std::vector<cv::Rect> &rectangles, int minArea=1);
bool checkRectOverlap(const cv::Rect &rectGT, const cv::Rect &r, float &intersectRatio);
void compareResults(const std::vector<cv::Rect> &gtBoxes, const std::vector<cv::Rect> &bboxes,
                    int &totalGT, int &totalFound, float &totalIntersectRatio, int &totalTP, int &totalFP, int &totalTN, int &totalFN);

float average(std::vector<float> const& v);
torch::Tensor imgToTensor(cv::Mat img);
float cosineSimilarity(float *A, float *B, unsigned int Vector_Length);
float torchSimilarity(torch::jit::Module model, cv::Mat frame_roi, cv::Mat bg_roi);
float calculateScore(cv::Mat frame, cv::Mat bg);

#endif //PERFORMANCE_UTILS_H
