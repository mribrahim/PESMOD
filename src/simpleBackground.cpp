//
// Created by ibrahim on 12/4/20.
//

#include "SimpleBackground.h"

#include "gflags/gflags.h"

#include "utils.h"
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/cudawarping.hpp>

DECLARE_int32(width);
DECLARE_int32(height);
DECLARE_bool(debug_mode);

using namespace std;
using namespace cv;

void SimpleBackground::init(const cuda::GpuMat &d_frame, int _minAge, int _maxAge) {

    farneback = cuda::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2);
    int width = 1920, height = 1080;

    d_age = cuda::GpuMat(height, width, CV_8UC1);
    d_background = cuda::GpuMat(height, width, dtype);
    d_ageMaskTemp = cv::cuda::GpuMat(d_frame.size(), CV_8UC1, cv::Scalar(1));
    map = Mat(d_frame.size(), CV_32FC2);
    d_map = cuda::GpuMat(d_frame.size(), CV_32FC2);

    d_frame.convertTo(d_background, dtype);
    d_age.setTo(0);
    minAge = _minAge;
    maxAge = _maxAge;
}

void SimpleBackground::reset() {
    d_background.setTo(0);
    d_ageGray.setTo(0);
}


bool SimpleBackground::update(const Mat &homoMat, const cuda::GpuMat &d_hsv, cuda::GpuMat &d_fgMask) {

    d_hsv.convertTo(d_frame, dtype);
    getChannel(d_hsv, 2, d_frameGray);

    cv::cuda::warpPerspective( d_background, d_backgroundWarped, homoMat, d_hsv.size(), INTER_LINEAR);
    cv::cuda::warpPerspective(d_age , d_ageWarped, homoMat, d_hsv.size());
    d_backgroundWarped.copyTo(d_background);
    d_ageWarped.copyTo(d_age);

    // ************** age ***************
    cv::cuda::warpPerspective(d_ageMaskTemp , d_ageMaskTempWarped, homoMat, d_hsv.size());

    cuda::multiply(d_age, d_ageMaskTempWarped, tempAge);
    cuda::add(tempAge, d_ageMaskTempWarped, d_age);
    cuda::threshold(d_age, d_age, maxAge, maxAge, THRESH_TRUNC);
    cuda::threshold(d_age, d_age_mask, minAge, 1, THRESH_BINARY);
//    showGPUMat("d_age_mask", d_age_mask, d_dummmy, 255);
    // ***************************************************************************

// *********************** learning rate *************************
    cuda::cvtColor(d_age, d_age3band, COLOR_GRAY2BGR);
    d_age3band.convertTo(d_age_f, CV_32FC3);
    cuda::divide(Scalar (1.0,1.0,1.0), d_age_f, d_alpha);
    cuda::subtract(Scalar (1.0,1.0,1.0), d_alpha, d_alpha_inv);
// ****************************************************************

//    cuda::absdiff(d_background, d_frame, d_diff);
    affineDiff(3, d_background, d_frame, d_diff);
    maxOf3bands(d_diff, d_resultBasicDiff);
//    showGPUMat("d_resultBasicDiff", d_resultBasicDiff, d_dummmy, 10);

    if (d_frameHistory.size()>10){
        d_frameHistory.erase(d_frameHistory.begin());
    }

    if (0 == d_frameHistory.size()){
        d_frameGray.copyTo(d_frameComparison);
    }else {
        d_frameComparison = d_frameHistory.at(d_frameHistory.size() - 1);
    }

    farneback->calc(d_frameComparison, d_frameGray, d_flowFarneback);
//    flow2img(d_flowFarneback, d_temp);
//    showGPUMat("d_flowFarneback", d_temp, d_dummmy);

    float meanBG, stdDevBG;
    findMeanBG(d_flowFarneback, d_magnitude, meanBG, stdDevBG);
//    cout <<"meanBG: " <<meanBG<<"  stdDev: " <<stdDevBG<< setprecision(4)<<endl;

    if (meanBG>19){
        meanBG=19;
    }
    float ratioThresh = 1 + 0.005 * exp( (1 + meanBG) / 4) ;
//    cout <<"ratioThresh: " <<ratioThresh<< "  thresh Final: "<< fgThreshDiff * ratioThresh<<endl;

    getChannel(d_background, 2, d_bgGray);
    farneback->calc(d_bgGray, d_frameGray, d_flowFarneback);
    flow2MagAngle(d_flowFarneback, d_magnitude, d_temp);
    processDiff(d_resultBasicDiff, fgThreshDiff * ratioThresh, d_magnitude, d_result);
//    showGPUMat("d_magnitude", d_magnitude, d_dummmy, 20);

    // **************** background(mean)***************
    cuda::multiply(d_background, d_alpha_inv, d_backgroundTemp);
    cuda::multiply(d_frame, d_alpha, d_frameRatio);
    cuda::add(d_backgroundTemp, d_frameRatio, d_background);
//  showGPUMat("d_background", d_background, d_dummmy, 1, 2);
    // ************************************************

    d_result.convertTo(d_fgMask, CV_8UC1);
    cuda::bitwise_and(d_fgMask, d_age_mask, d_fgMask);
    applyOpening(d_fgMask, d_fgMask);

    d_frameHistory.push_back(d_frameGray.clone());
    return true;
}

void SimpleBackground::processDiff(const cuda::GpuMat &d_diff, float fgThresh, cuda::GpuMat &d_result)
{
    cuda::threshold(d_diff, d_result, fgThresh, 1, THRESH_BINARY);
}

void SimpleBackground::processDiff(const cuda::GpuMat &d_diff, float fgThresh, const cuda::GpuMat &d_magnitude, cuda::GpuMat &d_result)
{
    cuda::threshold(d_magnitude, d_magMask, 5, 1, THRESH_BINARY);
    cuda::normalize(d_magnitude, d_magnitudeNorm, 0, 1.0,  NORM_MINMAX, -1);
    cuda::multiply(d_magnitudeNorm, d_magMask, d_magnitudeNorm);
    cuda::add(d_magnitudeNorm, 1.0, d_magnitudeNorm);
    cuda::multiply(d_diff, d_magnitudeNorm, d_diffModified);

    processDiff(d_diffModified, fgThresh, d_result);
}

void SimpleBackground::getBackground(Mat &bg) {
    cuda::GpuMat d_temp, d_bg;
    getChannel(d_background, 2, d_temp);
    d_temp.convertTo(d_bg, CV_8UC1);
    d_bg.download(bg);
}