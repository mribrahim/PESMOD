//
// Created by ibrahim on 3/9/21.
//

#include "utils.h"

#include <dirent.h>


using namespace std;
using namespace cv;

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


void enlargeRect(cv::Rect &rect, int a, int w, int h)
{
    rect.x -=a;
    rect.y -=a;
    rect.width += (a*2);
    rect.height += (a*2);

    if(rect.x < 0){
        rect.x = 0;
    }
    if(rect.y < 0){
        rect.y = 0;
    }
    if( (rect.x + rect.width) >= w){
        rect.width = w - rect.x - 1;
    }
    if( (rect.y + rect.height) >= h){
        rect.height = h - rect.y - 1;
    }
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