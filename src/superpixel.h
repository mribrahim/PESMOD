//
// Created by ibrahim on 11/1/21.
//

#ifndef BGS_SUPERPIXEL_H
#define BGS_SUPERPIXEL_H



#include "gSLICr_Lib/gSLICr.h"

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>


class SuperPixel {

public:
    SuperPixel();
    void run(const cv::Mat &frame, cv::Mat &segments, cv::Mat &segmentsEdges, bool showResult=false);
    void runRegions(const cv::Mat &frame, cv::Mat &segments);
private:
    int width = 1920;
    int height = 1080;
    gSLICr::engines::core_engine* gSLICr_engine;
    gSLICr::UChar4Image* in_img, *out_img;
    unsigned short *matrix;

    void load_image(const cv::Mat& inimg, gSLICr::UChar4Image* outimg);
    void load_image(const gSLICr::UChar4Image* inimg, cv::Mat& outimg);
};


#endif //BGS_SUPERPIXEL_H
