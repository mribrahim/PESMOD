#include <iostream>
#include "vector"
#include "regex"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "utils.h"
#include "SimpleBackground.h"
#include "superpixel.h"

using namespace std;
using namespace  cv;

int MAX_COUNT;
std::vector<uchar> status;
std::vector<cv::Point2f>  pointsPrev, pointsCurrent;

Mat findHomographyMatrix(const Mat &prevGray, const Mat &currentGray);
Mat makeHomoGraphy(int *pnMatch, int nCnt);
void applySuperpixel(SuperPixel superPiksel, const Mat &frame, const cuda::GpuMat &d_frame, cuda::GpuMat &d_fgMask);

int main() {

    string folderName = "Pexels-Wolfgang";
    string path =  "/home/ibrahim/Desktop/Dataset/my IHA dataset/PESMOD/";

    vector<string> imageList, maskList;
    bool maskFound = false;

    read_directory(path + folderName + "/images/", imageList);
    sort(imageList.begin(), imageList.end());

    int width = 1920, height = 1080;
    int gridSizeW = 32;
    int gridSizeH = 24;
    MAX_COUNT =  (width / gridSizeW + 1) * (height / gridSizeH + 1);

    for (int i = 0; i < width / gridSizeW - 1; ++i) {
        for (int j = 0; j < height / gridSizeH - 1; ++j) {
            pointsPrev.push_back(Vec2f(i * gridSizeW + gridSizeW / 2.0, j * gridSizeH + gridSizeH / 2.0));
        }
    }

    Mat frame, frameGray, frameGrayPrev, fgMask;
    cuda::GpuMat d_frame, d_hsv, d_frameGray, d_fgMask;
    bool isInitialized = false;
    SimpleBackground bgs;
    SuperPixel superPiksel;
//    auto model = torch::jit::load("/home/ibrahim/MyProjects/traced_resnet_model.pt");
//    model.eval();

    vector<float> gtCosinusScores, gtTemplateScores;

    int totalGT=0, totalFound = 0, totalTP=0, totalFP=0, totalTN=0, totalFN=0;
    float totalIntersectRatio = 0;
    int i = 0;
    bool processOneStep = false;
    bool isStopped = false;
    int keyboard;
    do{

        keyboard = waitKey(2);
        int64 startTime = cv::getTickCount();

        if ('s' == keyboard)
        {
            isStopped = !isStopped;
            processOneStep = false;
        }
        else if ( 83 == keyboard )
        {
            processOneStep = true;
        }
        else if (keyboard == 'q' || keyboard == 27) {
            break;
        }

        if (isStopped){
            continue;
        }

        vector<Rect> bboxesGT, bboxesFound;
        string filename = imageList.at(i);
//        cout<<"filename: " <<filename<<endl;
        string fullPathFrame = path + folderName + "/images/" + filename;
        Mat frame = imread(fullPathFrame);
        Mat frameShow;
        frame.copyTo(frameShow);

        d_frame.upload(frame);
        cuda::cvtColor(d_frame, d_hsv, COLOR_BGR2HSV);
        cuda::cvtColor(d_frame, d_frameGray, COLOR_BGR2GRAY);

        if (!isInitialized){
            d_frameGray.download(frameGrayPrev);
            bgs.init(d_hsv);
            isInitialized = true;
            i++;
            continue;
        }

        d_frameGray.download(frameGray);
        Mat homoMat = findHomographyMatrix(frameGrayPrev, frameGray);

        bgs.update(homoMat, d_hsv, d_fgMask);
//        applySuperpixel(superPiksel, frame, d_frame, d_fgMask);
        cuda::multiply(d_fgMask, 255, d_fgMask);
        d_fgMask.download(fgMask);

        if (fgMask.empty()){
            i++;
            continue;
        }
        showMat("Foreground mask", fgMask);
        frameGray.copyTo(frameGrayPrev);

        Mat background;
        bgs.getBackground(background);
//        showMat("background", background);

        bboxesGT = readGtboxesPESMOT(fullPathFrame);
        for (Rect box: bboxesGT){

            if (box.x + box.width > frame.cols){
                box.width = frame.cols - box.x;
            }
            if (box.y + box.height > frame.rows){
                box.height = frame.rows - box.y;
            }

            Mat frame_roi = frameGray(box);
            Mat bg_roi= background(box);

//            float cosSimilarity = torchSimilarity(model, frame_roi, bg_roi);
//            float score = calculateScore(frame_roi, bg_roi);
//            gtCosinusScores.push_back(cosSimilarity);
//            gtTemplateScores.push_back(score);

            rectangle(frameShow, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(0,255,0));
        }

        Mat maskRegions, maskSmallregions;
        findCombinedRegions(fgMask, maskRegions, maskSmallregions, bboxesFound, 10);

        vector<Rect> selectedBoxes;
        for(Rect box: bboxesFound)
        {

            Mat frame_roi = frameGray(box);
            Mat bg_roi= background(box);

//            Mat temp;
//            resize(frame_roi, temp, Size(224,224));
//            imshow("frame_roi",temp);
//            resize(bg_roi, temp, Size(224,224));
//            imshow("bg_roi", temp);
//            cout << "score: "<<score<< "   cosSimilarity: " << cosSimilarity << endl;
//            waitKey(0);
//
//            float cosSimilarity = torchSimilarity(model, frame_roi, bg_roi);
//            float score = calculateScore(frame_roi, bg_roi);

//            int x1 = (box.x < 20) ? 20 : box.x;
//            int y1 = (box.y < 20) ? 20 : box.y;
//
//            if (abs(score) < 0.1 || cosSimilarity > 0.8){
//                rectangle(frame, box, Scalar (255,0,0), 2, 1);
//                putText(frameShow, to_string(cosSimilarity), cv::Point(x1, y1), 2,1, Scalar(255,0,0));
//                continue;
//            }
            selectedBoxes.push_back(box);
            rectangle(frameShow, box, Scalar (0,0,255), 2, 1);
//            putText(frameShow, to_string(cosSimilarity), cv::Point(x1, y1), 2,1, Scalar(0,0,255));
        }
        compareResults(bboxesGT, selectedBoxes, totalGT, totalFound, totalIntersectRatio, totalTP, totalFP, totalTN, totalFN);

        showMat("frame", frameShow);

        i++;

        if ( 0 == i % 20) {
            double secs = (cv::getTickCount() - startTime) / cv::getTickFrequency();
            cout << "elapsed time: " << secs << endl;
        }
    } while (i < imageList.size());


    float precision = float(totalTP) / (totalTP+totalFP);
    float recall = float(totalTP) / (totalTP+totalFN);
    float f1 = 2*precision*recall / (precision+recall);
    float pwc = 100 * (float)(totalFN + totalFP) / (totalTP + totalFP + totalFN + totalTN);
    cout << " sequence: " << path << endl;
    cout << " folderName: " << folderName << endl;
    cout << " totalGT: " << totalGT << endl;
    cout << " (totalTP + totalFN): " << (totalTP + totalFN) << endl;
    cout << " totalFound: " << totalFound << endl;
    cout << " intersectRatio average: " << totalIntersectRatio/totalTP  << endl;
    cout << " precision: " << precision << "  recall: " << recall << "  f1: " << f1 << "  pwc: "<< pwc << endl;

    cout<< "gtTemplateScores: " << average(gtTemplateScores) << endl;
    cout<< "gtCosinusScores: " << average(gtCosinusScores) << endl;
    return 0;
}


Mat findHomographyMatrix(const Mat &prevGray, const Mat &currentGray)
{
    int* nMatch = (int*)alloca(sizeof(int) * MAX_COUNT);
    int count;
    int flags = 0;
    int i =0, k=0;
    if (!pointsPrev.empty())
    {
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 20, 0.03);
        calcOpticalFlowPyrLK(prevGray, currentGray, pointsPrev, pointsCurrent, status, noArray(), Size(15, 15), 2, criteria, flags);

        for (i = k = 0; i < status.size(); i++) {
            if (!status[i]) {
                continue;
            }

            nMatch[k++] = i;
        }
        count = k;
    }
    if (k >= 10) {
        // Make homography matrix with correspondences
        return makeHomoGraphy(nMatch, count);
        //homoMat = findHomography(points0, points1, RANSAC, 1);
    } else {
        return (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}

Mat makeHomoGraphy(int *pnMatch, int nCnt)
{
    vector<Point2f> pt1;
    vector<Point2f> pt2;
    for (int i = 0; i < nCnt; ++i)
    {
        pt1.push_back(pointsPrev[pnMatch[i]]);
        pt2.push_back(pointsCurrent[pnMatch[i]]);

    }
    return findHomography(pt1, pt2, RANSAC, 1);
}


void applySuperpixel(SuperPixel superPiksel, const Mat &frame, const cuda::GpuMat &d_frame, cuda::GpuMat &d_fgMask)
{
    cv::Mat segments, segmentsEdges, maskSegments;
    superPiksel.run(frame, segments, segmentsEdges, false);

//    showMat("Superpixel segments", segments);
//    showMat("Superpixel edges", segmentsEdges);

    cvtColor(segmentsEdges, maskSegments, COLOR_BGR2GRAY);
    threshold(maskSegments, maskSegments, 1, 255, THRESH_BINARY_INV);

    Mat fgMask, fg16U;
    d_fgMask.download(fgMask);

    // ******* set border to zero ************
    cv::Rect border(cv::Point(0, 0), maskSegments.size());
    cv::Scalar color(0);
    int thickness = 1;
    cv::rectangle(maskSegments, border, color, thickness);
    // ***************************************

    Mat mask;
    multiply(maskSegments, fgMask, mask);

    vector<vector<Point> > contoursIntersect, contoursFG;
    vector<Vec4i> hierarchy;

    findContours( mask, contoursIntersect, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE );

    for( size_t i = 0; i< contoursIntersect.size(); i++ ) {

        unsigned short regionId = segments.at<unsigned short>(contoursIntersect[i].at(0));
        Mat region = segments==regionId;

        int areaOfsuperPixelRegion = countNonZero(region);
        int areaOfMotion = contourArea(contoursIntersect[i]);
        if (areaOfMotion > (areaOfsuperPixelRegion/5) ){
            fgMask.setTo(1, region);
//            cout << "use superpixel region  " << regionId<< endl;
        }
    }
    d_fgMask.upload(fgMask);
}

