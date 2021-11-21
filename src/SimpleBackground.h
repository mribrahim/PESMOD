//
// Created by ibrahim on 12/4/20.
//

#ifndef PERCEPTION_SIMPLE_BACKGROUNDMODEL_H
#define PERCEPTION_SIMPLE_BACKGROUNDMODEL_H

#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include "opencv4/opencv2/cudaobjdetect.hpp"



class SimpleBackground {

public:
    cv::cuda::GpuMat d_background;
    cv::cuda::GpuMat d_age, d_ageGray;

    void init(const cv::cuda::GpuMat &d_frame, int _minAge=5, int _maxAge=30);
    bool update(const cv::Mat &homoMat, const cv::cuda::GpuMat &d_hsv, cv::cuda::GpuMat &d_fgMask);
    void reset();
    void getBackground(cv::Mat &background);
private:
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback;
    int dtype = CV_32FC3;
    int minAge;
    int maxAge;
    int fgThreshDiff = 25;

    cv::cuda::GpuMat d_dummmy, d_temp;
    cv::cuda::GpuMat d_frame, d_frameGray, d_frameGrayPrev;
    cv::cuda::GpuMat d_backgroundWarped, d_bgGray, d_backgroundTemp, d_frameRatio;
    cv::cuda::GpuMat d_ageWarped, d_ageMaskTemp, d_ageMaskTempWarped, d_age_mask, tempAge;
    cv::cuda::GpuMat d_alpha, d_alpha_inv;
    cv::cuda::GpuMat d_age3band, d_age_f;
    cv::cuda::GpuMat d_magMask, d_magnitudeNorm, d_diffModified;
    cv::cuda::GpuMat d_diff, d_resultBasicDiff, d_diffPrev, d_result, d_resultBG, d_resultPrev;
    cv::cuda::GpuMat d_flowFarneback, d_magnitude, d_regionMask;

    void processDiff(const cv::cuda::GpuMat &d_diff, float fgThresh, const cv::cuda::GpuMat &d_magnitude, cv::cuda::GpuMat &d_result);
    void processDiff(const cv::cuda::GpuMat &d_diff, float fgThresh, cv::cuda::GpuMat &d_result);
};


#endif //PERCEPTION_SIMPLE_BACKGROUNDMODEL_H
