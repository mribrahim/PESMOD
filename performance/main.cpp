//
// Created by ibrahim on 09/03/2021
//

#include <iostream>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "utils.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    string folderName = "Pexels-Marian";
    string path =  "/home/ibrahim/Desktop/Dataset/my IHA dataset/PESMOD/";
    string pathMask = "/home/ibrahim/MyProjects/pesmod_dataset/SCBU-PESMOD-results/";
    bool applyOpening = false;

    for (int i = 0; i < argc; ++i)
    {
        if (0 == strcmp("-d", argv[i]))
        {
            path = argv[i+1];
        }
        else if (0 == strcmp("-m", argv[i]))
        {
            pathMask = argv[i+1];
        }
        else if (0 == strcmp("-f", argv[i]))
        {
            folderName = argv[i+1];
        }
        else if (0 == strcmp("-o", argv[i]))
        {
            applyOpening = true;
        }
    }

    path =  path + folderName +"/";
    pathMask = pathMask + folderName +"/";

    cout<< "\n\nImage sequence main folder: " << path << endl;
    cout<< "Image sequence mask main folder: " << pathMask << endl;
    cout<< "Image sequence folder name: " << folderName << endl;

    vector<string> imageList, maskList;
    bool maskFound = false;

    read_directory(path +"/images/", imageList);
    sort(imageList.begin(), imageList.end());

    if (!pathMask.empty()){
        maskFound = true;
        read_directory(pathMask, maskList);
        sort(maskList.begin(), maskList.end());
    }
    else{
        cout<<" No mask folder input, No performance comparison..." << endl;
    }

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

    int totalGT=0, totalFound = 0, totalTP=0, totalFP=0, totalTN=0, totalFN=0;
    float totalIntersectRatio = 0;
    int i = 0;
    bool processOneStep = false;
    bool isStopped = false;
    int keyboard;
    do{

        keyboard = waitKey(10);
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
        string fullPath = path + "/images/" + filename;
        Mat frame = imread(fullPath);

        bboxesGT = readGtboxesPESMOT(fullPath);
        for (Rect box: bboxesGT){
            rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(0,255,0));
        }

        if (maskFound){
            // implement mask to Bboxes
            Mat mask, maskResized;
            mask = imread(pathMask + filename, 0);

            if (mask.empty()){
                i++;
                continue;
            }


            if (applyOpening){
                morphologyEx( mask, mask, MORPH_OPEN, kernel );
            }

            resize(mask, maskResized, Size(960, 540));
            imshow("FG mask", maskResized);

            Mat frame_channels[3];
            split(frame, frame_channels);
            add(frame_channels[2], mask, frame_channels[2]);
            merge(frame_channels, 3, frame);


            Mat maskRegions, maskSmallregions;
            findCombinedRegions(mask, maskRegions, maskSmallregions, bboxesFound, 10);

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

        }

        resize(frame, frame, Size(960, 540));
        imshow("frame", frame);


        if (processOneStep){
            keyboard = waitKey(0);
            if (83 != keyboard){
                processOneStep = false;
            }
        }
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
