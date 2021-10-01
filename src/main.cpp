#include <iostream>
#include "vector"
#include "regex"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

#include "utils.h"
#include "SimpleBackground.h"

using namespace std;
using namespace  cv;

int MAX_COUNT;
std::vector<uchar> status;
std::vector<cv::Point2f>  pointsPrev, pointsCurrent;

Mat findHomographyMatrix(const Mat &prevGray, const Mat &currentGray);
Mat makeHomoGraphy(int *pnMatch, int nCnt);

int main() {

    string folderName = "Pexels-Wolfgang";
    string path =  "/home/ibrahim/Desktop/Dataset/my IHA dataset/PESMOD/";
    string pathMask = "/home/ibrahim/MyProjects/pesmod_dataset/MySimple-PESMOD-results/";

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
    cuda::GpuMat d_frame, d_hsv, d_frameGray, d_fgMask, d_background;
    bool isInitialized = false;
    SimpleBackground bgs;

    int totalGT=0, totalFound = 0, totalTP=0, totalFP=0, totalTN=0, totalFN=0;
    float totalIntersectRatio = 0;
    int i = 0;
    bool processOneStep = false;
    bool isStopped = false;
    int keyboard;
    do{

        keyboard = waitKey(2);
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
        cout<<"filename: " <<filename<<endl;
        string fullPathFrame = path + folderName + "/images/" + filename;
        Mat frame = imread(fullPathFrame);

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
        bgs.getBackground(d_background);

        cuda::multiply(d_fgMask, 255, d_fgMask);
        d_fgMask.download(fgMask);

//        fgMask = imread(pathMask + folderName + "/" + filename, 0);
//        string replaceStr = "/home/ibrahim/Desktop/Dataset/my IHA dataset/PESMOD/";
//        string wfilename = std::regex_replace(fullPathFrame, std::regex(replaceStr), "/home/ibrahim/MyProjects/pesmod_dataset/MyNew-PESMOD-results/");
//        wfilename = std::regex_replace(wfilename, std::regex("images/"), "");
//        imwrite("temp.jpg", fgMask );
//        fgMask = imread("temp.jpg", 0);


        if (fgMask.empty()){
            i++;
            continue;
        }
        showMat("Foreground mask", fgMask);
        frameGray.copyTo(frameGrayPrev);

        bboxesGT = readGtboxesPESMOT(fullPathFrame);
        for (Rect box: bboxesGT){
            rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(0,255,0));
        }

        Mat maskRegions, maskSmallregions;
        findCombinedRegions(fgMask, maskRegions, maskSmallregions, bboxesFound, 10);

        vector<Rect> selectedBoxes;
        for(Rect box: bboxesFound)
        {
            unsigned int x1 = box.x;
            unsigned int y1 = box.y;
            unsigned int x2 = box.x + box.width;
            unsigned int y2 = box.y + box.height;

            if (x1 < 5 or y1 < 5 or x2 > frame.cols-5 or y2 > frame.rows-5)
            {
                continue;
            }
            selectedBoxes.push_back(box);
            rectangle(frame, box, Scalar (255,0,0), 2, 1);
        }
        compareResults(bboxesGT, selectedBoxes, totalGT, totalFound, totalIntersectRatio, totalTP, totalFP, totalTN, totalFN);

        showMat("frame", frame);

        i++;
    } while (i < imageList.size());


    float precision = float(totalTP) / (totalTP+totalFP);
    float recall = float(totalTP) / (totalTP+totalFN);
    float f1 = 2*precision*recall / (precision+recall);
    float pwc = 100 * (float)(totalFN + totalFP) / (totalTP + totalFP + totalFN + totalTN);
    cout << " sequence: " << path << endl;
    cout << " totalGT: " << totalGT << endl;
    cout << " (totalTP + totalFN): " << (totalTP + totalFN) << endl;
    cout << " totalFound: " << totalFound << endl;
    cout << " intersectRatio average: " << totalIntersectRatio/totalTP  << endl;
    cout << " precision: " << precision << "  recall: " << recall << "  f1: " << f1 << "  pwc: "<< pwc << endl;

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

