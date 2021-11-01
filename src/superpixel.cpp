//
// Created by ibrahim on 11/1/21.
//


#include <stdio.h>

#include "superpixel.h"
#include "gflags/gflags.h"

#include <iostream>
#include <string>

using namespace std;
using namespace cv;


SuperPixel::SuperPixel() {

    // gSLICr settings
    gSLICr::objects::settings my_settings;
    my_settings.img_size.x = width;
    my_settings.img_size.y = height;
    my_settings.no_segs = 2000;
    my_settings.spixel_size = 16;
    my_settings.coh_weight = 0.6f;
    my_settings.no_iters = 5;
    my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
    my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
    my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step

    // instantiate a core_engine
    gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

    // gSLICr takes gSLICr::UChar4Image as input and out put
    in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
    out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

    matrix = new unsigned short [width * height];
    memset(matrix, 0x00, width * height);
}


void SuperPixel::run(const Mat &frame, Mat &segments, Mat &segmentsEdges, bool showResult)
{
    segments = Mat(Size2d(width, height), CV_16UC1);
    segmentsEdges = Mat(Size2d(width, height), CV_8UC3);

    load_image(frame, in_img);
    gSLICr_engine->Process_Frame(in_img);

    if (showResult){
        gSLICr_engine->Draw_Segmentation_Result(out_img);
        load_image(out_img, segmentsEdges);
        imshow("Superpixel result", segmentsEdges);
    }
    gSLICr_engine->Draw_Segmentation_Edges(out_img);

    gSLICr_engine->Write_Seg_Res_To_PGM("abc", matrix);
    std::memcpy(segments.data, matrix, width*height*sizeof(short int));

    load_image(out_img, segmentsEdges);
}

void SuperPixel::runRegions(const Mat &frame, Mat &segments)
{
    segments = Mat(Size2d(width, height), CV_16UC1);
    load_image(frame, in_img);
    gSLICr_engine->Process_Frame(in_img);
    gSLICr_engine->Write_Seg_Res_To_PGM("abc",matrix);
    std::memcpy(segments.data, matrix, width*height*sizeof(short int));
}

void SuperPixel::load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
    gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < outimg->noDims.y;y++)
        for (int x = 0; x < outimg->noDims.x; x++)
        {
            int idx = x + y * outimg->noDims.x;
            outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
            outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
            outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
        }
}

void SuperPixel::load_image(const gSLICr::UChar4Image* inimg, Mat& outimg) {
    const gSLICr::Vector4u *inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < inimg->noDims.y; y++)
        for (int x = 0; x < inimg->noDims.x; x++) {
            int idx = x + y * inimg->noDims.x;
            outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
            outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
            outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
        }
}