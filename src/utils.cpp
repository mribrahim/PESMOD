//
// Created by ibrahim on 3/9/21.
//

#include "utils.h"

#include <dirent.h>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;

Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
cv::Ptr<cv::cuda::Filter> morphFilterOpen = cuda::createMorphologyFilter(MORPH_OPEN, CV_8UC1, kernel);


// flow2img
cuda::GpuMat d_magnitude, d_angle, d_magn_norm;
cuda::GpuMat d_hsv_parts[3], d_hsv, d_hsv8;
cuda::GpuMat ones;

cuda::GpuMat img_channels[3];
//showGPUMat
cuda::GpuMat d_mat, d_temp;
// flow2MagAngle
cuda::GpuMat d_flowXY[2];
//maxOf3Bands
cuda::GpuMat d_diffTemp, d_channels[3];
//findMeanBG
cv::cuda::GpuMat d_flowImg, d_flowImgGray;

void read_directory(const string& path, vector<string>& v)
{
    DIR* dirp = opendir(path.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {

        string filePath = path + dp->d_name;
        if (std::string::npos == filePath.find(".png") && std::string::npos == filePath.find(".jpg"))
            continue;

        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

void showMat(string text, const Mat &img){
    Mat tempMat;

    if (img.rows>1000){
        resize(img, tempMat, Size(2* img.cols/3, 2* img.rows/3));
    }
    else{
        img.copyTo(tempMat);
    }

    imshow(text, tempMat);
}

void showGPUMat(string window_text, const cuda::GpuMat &d_matInput, const cuda::GpuMat &mask,
                Scalar factor, int channel) {

    if (d_matInput.rows>1000){
        cuda::resize(d_matInput, d_mat, Size(d_matInput.cols/2, d_matInput.rows/2));
    }
    else{
        d_matInput.copyTo(d_mat);
    }

    if (channel>0){
        getChannel(d_mat, channel, d_mat);
    }
    Mat temp;
    if (!mask.empty())
    {
        cuda::multiply(d_mat, mask, d_mat);
    }
    if (factor.val[0]>0)
    {
        cuda::multiply(d_mat, factor, d_mat);
    }
    if (d_mat.type() == CV_32FC1 || d_mat.type() == CV_16UC1)
    {
        d_mat.convertTo(d_temp, CV_8UC1);
        d_temp.download(temp);
    }
    else if (d_mat.type() == CV_32FC3)
    {
        d_mat.convertTo(d_temp, CV_8UC3);
        d_temp.download(temp);
    }
    else
    {
        d_mat.download(temp);
    }
    imshow(window_text, temp);
}

void getChannel(const Mat &img, uint8_t channelId, Mat &channel)
{
    Mat img_channels[3];
    split(img, img_channels);
    img_channels[channelId].copyTo(channel);
}

void getChannel(const cuda::GpuMat &img, uint8_t channelId, cuda::GpuMat &channel)
{
    cuda::split(img, img_channels);
    img_channels[channelId].copyTo(channel);
}

void flow2MagAngle(const cuda::GpuMat &d_flow, cuda::GpuMat &d_magnitude, cuda::GpuMat &d_angle)
{
    cuda::split(d_flow, d_flowXY);
    cv::cuda::cartToPolar(d_flowXY[0], d_flowXY[1], d_magnitude, d_angle, true);
}

void flow2img(const cuda::GpuMat &flow, cuda::GpuMat &img)
{
    cuda::split(flow, d_flowXY);
    cuda::cartToPolar(d_flowXY[0], d_flowXY[1], d_magnitude, d_angle, true);
    cuda::normalize(d_magnitude, d_magn_norm, 0.0f, 1.0f, NORM_MINMAX, -1);
    cuda::multiply(d_angle, ((1.f / 360.f) * (180.f / 255.f)), d_angle);
    //build hsv image
    if (ones.empty()){
        ones= cuda::GpuMat(d_angle.size(), CV_32F);
    }
    ones.setTo(0);
    d_hsv_parts[0] = d_angle;
    d_hsv_parts[1] = ones;
    d_hsv_parts[2] = d_magn_norm;
    merge(d_hsv_parts, 3, d_hsv);
    d_hsv.convertTo(d_hsv8, CV_8U, 255.0);
    cuda::cvtColor(d_hsv8, img, COLOR_HSV2BGR);
}

void maxOf3bands(const cuda::GpuMat &d_diff, cuda::GpuMat &d_diffMax)
{
    cuda::split(d_diff, d_channels);
    cuda::max(d_channels[1], d_channels[2], d_diffMax);
//    cuda::max(d_channels[2], d_diffTemp, d_diffMax);
}

void findMeanBG(const cuda::GpuMat d_flow, cuda::GpuMat &d_magnitude, float &meanBG, float &stdDevBG)
{
    flow2MagAngle(d_flow, d_magnitude, d_angle);

    Mat changes;
    d_magnitude.download(changes);
    float toplam=0, cnt=0;

    for (int i = 0; i < 1920; i+=32) {
        for (int j = 0; j < 1080; j+=32) {
            cnt++;
            toplam += changes.at<float>(j,i);
        }
    }

    meanBG = (float)toplam/cnt;

    toplam=0;
    for (int i = 0; i < 1920; i+=32) {
        for (int j = 0; j < 1080; j+=32) {
            toplam += pow(fabs(changes.at<float>(j,i) - meanBG), 2);
        }
    }
    stdDevBG = sqrt(toplam/cnt);
}

void affineDiff(int kernelSize, const cuda::GpuMat &d_bg, const cuda::GpuMat& d_frameC, cuda::GpuMat &result)
{
    Mat par = Mat(2, 3, CV_64FC1); // Allocate memory
    Mat imgAffine;

    par.at<double>(0,0)= 1; //p1
    par.at<double>(1,0)= 0; //p2;
    par.at<double>(0,1)= 0; //p3;
    par.at<double>(1,1)= 1; //p4;
//    par.at<double>(0,2)= 9; //p5;
//    par.at<double>(1,2)= 9; //p6;

    int cnt = 0;
    int offset = kernelSize / 2;
    Mat temp;
    cuda::GpuMat d_affine, d_diff[kernelSize*kernelSize];
    for (int i = -offset; i <= offset; ++i) {
        par.at<double>(0,2)= i; //p5;
        for (int j = -offset; j <= offset; ++j) {
            par.at<double>(1,2)= j; //p6;

            cuda::warpAffine(d_frameC, d_affine, par, d_frameC.size());
            cuda::absdiff(d_affine, d_bg, d_diff[cnt]);
            cnt++;
        }
    }

    result = d_diff[0];
    for (int k = 1; k < kernelSize*kernelSize; ++k)
    {
        cuda::min(result, d_diff[k], result);
    }
}

void applyOpening(const cuda::GpuMat &maskInput, cuda::GpuMat &maskOutput)
{
    morphFilterOpen->apply(maskInput, maskOutput);
}


vector<Rect> readGtboxesPESMOT(string path) {

    string gtPath = path;
    std::size_t found = path.find("images/");

    gtPath.replace(found, 7, "annotations/");
    gtPath.replace(gtPath.length()-3,3,"xml" );

    rapidxml::xml_document<> doc;
    rapidxml::xml_node<> * root_node;

    vector<Rect> bboxes;
    ifstream theFile(gtPath);
    if (theFile)
    {
        vector<char> buffer((istreambuf_iterator<char>(theFile)),                             istreambuf_iterator<char>());
        buffer.push_back('\0');
        // Parse the buffer using the xml file parsing library into doc
        doc.parse<0>(&buffer[0]);
        // Find our root node

        root_node = doc.first_node("annotation");

        for (rapidxml::xml_node<> * object_node = root_node->first_node("object"); object_node; object_node = object_node->next_sibling())
        {
            rapidxml::xml_node<> * node = object_node->first_node("bndbox");

            int x1 = stoi(node->first_node("xmin")->value());
            int y1 = stoi(node->first_node("ymin")->value());
            int x2 = stoi(node->first_node("xmax")->value());
            int y2 = stoi(node->first_node("ymax")->value());
            bboxes.push_back(Rect(x1, y1, x2-x1, y2-y1));
        }
    }

    return bboxes;
}


void enlargeRect(cv::Rect &rect, int a)
{
    rect.x -=a;
    rect.y -=a;
    rect.width +=a;
    rect.height +=a;
}

void findCombinedRegions(const Mat &mask, Mat &maskRegionOutput, Mat &maskSmallregions, vector<Rect> &rectangles, int minArea)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
    maskSmallregions = Mat::zeros( mask.size(), CV_8UC1 );

    for( size_t i = 0; i< contours.size(); i++ )
    {
        if (contourArea(contours[i]) <= minArea)
        {
            continue;
        }

        Rect rect = boundingRect(contours[i]);
        enlargeRect(rect);
        rectangle(maskSmallregions, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), 1);
        //        drawContours( drawing, contours, (int)i, color, 2, LINE_8 );
    }
    maskRegionOutput = Mat::zeros( mask.size(), CV_8UC1 );
    findContours( maskSmallregions, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Rect rect = boundingRect(contours[i]);
        rectangles.push_back(rect);

        rectangle(maskRegionOutput, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), 1, FILLED);
        //        drawContours( drawing2, contours, (int)i, 255, 2, LINE_8 );
    }
    //    imshow("drawing2", drawing2);
}


bool checkRectOverlap(const Rect &rectGT, const Rect &r, float &intersectRatio)
{
    Rect rectIntersection = rectGT & r;
    float areaUnion = rectGT.area() + r.area() - rectIntersection.area();
    if (rectIntersection.area() / areaUnion > 0.5){
        float ratio = (float)rectIntersection.area() / rectGT.area();
        intersectRatio += ratio;
        return true;
    }
    return false;
}


void compareResults(const vector<cv::Rect> &gtBoxes, const vector<cv::Rect> &bboxes, int &totalGT, int &totalFound, float &intersectRatio, int &totalTP, int &totalFP, int &totalTN, int &totalFN)
{
    int tp=0, fp=0, tn=0, fn=0;

    vector<int> boxesMatchedIndexes;
    for (Rect gt : gtBoxes)
    {
        bool found = false;
        for (int i=0; i<bboxes.size(); ++i)
        {
            Rect box = bboxes.at(i);
            if (checkRectOverlap(gt, box, intersectRatio)){
                found = true;
                boxesMatchedIndexes.push_back(i);
            }
        }
        if (found) {
            tp++;
        }
        else{
            fn++;
        }
    }

    for (int i=0; i<bboxes.size(); ++i)
    {
        vector<int>::iterator it= std::find (boxesMatchedIndexes.begin(), boxesMatchedIndexes.end(), i);
        if (it != boxesMatchedIndexes.end()) {
            continue;
        }
        fp++;
    }

    totalGT += gtBoxes.size();
    totalFound += bboxes.size();
    totalTP += tp;
    totalFP += fp;
    totalFN += fn;
}

cv::Mat crop_center(const cv::Mat &img)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const int cropSize = std::min(rows,cols);
    const int offsetW = (cols - cropSize) / 2;
    const int offsetH = (rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

    return img(roi);
}

std::vector<double> norm_mean = {0.485, 0.456, 0.406};
std::vector<double> norm_std = {0.229, 0.224, 0.225};

torch::Tensor imgToTensor(Mat img)
{
    img = crop_center(img);
    cv::resize(img, img, cv::Size(224,224));

    if (img.channels()==1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    img.convertTo( img, CV_32FC3, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);

    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

    return img_tensor.clone();
}

float cosineSimilarity(float *A, float *B, unsigned int Vector_Length)
{
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int i = 0u; i < Vector_Length; ++i) {
        dot += A[i] * B[i] ;
        denom_a += A[i] * A[i] ;
        denom_b += B[i] * B[i] ;
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b)) ;
}

float calculateScoreTemplate(Mat frame, Mat bg) {

    cv::Mat matMatchTemplate;
    cv::matchTemplate(frame, bg, matMatchTemplate, TM_SQDIFF_NORMED);
    double distanceMatchingTemplate;
    cv::minMaxIdx(matMatchTemplate, &distanceMatchingTemplate);
    return distanceMatchingTemplate;
}

float calculateScoreHist(Mat frame, Mat bg) {

    Mat hist1, hist2;
    vector<int> channels = {0};
    vector<int> binNumArray = {256};
    vector<float> ranges = { 0, 256 };
    vector<Mat> mats1 = { frame };
    vector<Mat> mats2 = { bg };
    calcHist( mats1, channels, Mat(), hist1, binNumArray, ranges) ;
    calcHist( mats2, channels, Mat(), hist2, binNumArray, ranges) ;
    return compareHist(hist1, hist2, HISTCMP_CORREL);
}

float calculateScore(Mat frame, Mat bg)
{
    return calculateScoreTemplate(frame, bg);
}