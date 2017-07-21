#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
using namespace std;
//人臉最小的範圍
int min_face_height = 50;
int min_face_width = 50;
int main( int argc , char ** argv ){
    string image_name="C:/face15.jpg";
    // 讀取圖片
    IplImage* image_detect=cvLoadImage(image_name.c_str(), 1);
    string cascade_name="C:/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";

    //讀取cascade
    CvHaarClassifierCascade* classifier=(CvHaarClassifierCascade*)cvLoad(cascade_name.c_str(), 0, 0, 0);
    if(!classifier){
        cerr<<"ERROR: Could not load classifier cascade."<<endl;
         system("pause");
        return -1;
    }
    //配置記憶體
    CvMemStorage* facesMemStorage=cvCreateMemStorage(0);
    IplImage* tempFrame=cvCreateImage(cvSize(image_detect->width, image_detect->height), IPL_DEPTH_8U, image_detect->nChannels);
    if(image_detect->origin==IPL_ORIGIN_TL){
        cvCopy(image_detect, tempFrame, 0);    }
    else{
        cvFlip(image_detect, tempFrame, 0);    }

    cvClearMemStorage(facesMemStorage);
    //偵測人臉
CvSeq* faces=cvHaarDetectObjects(tempFrame, classifier, facesMemStorage, 1.1, 3
, CV_HAAR_DO_CANNY_PRUNING, cvSize(min_face_width, min_face_height));
    if(faces){
        for(int i=0; i<faces->total; ++i){
            //畫出偵測到的人臉框
            CvPoint point1, point2;
            CvRect* rectangle = (CvRect*)cvGetSeqElem(faces, i);
            point1.x = rectangle->x;
            point2.x = rectangle->x + rectangle->width;
            point1.y = rectangle->y;
            point2.y = rectangle->y + rectangle->height;
            cvRectangle(tempFrame, point1, point2, CV_RGB(255,0,0), 3, 8, 0);
        }
    }
    //另存成02.bmp圖片
    cvSaveImage("02.bmp", tempFrame);
    //視窗名稱
    cvNamedWindow("Face Detection Result", 1);
    //展示圖片
    cvShowImage("Face Detection Result", tempFrame);
    //讓視窗停住
    cvWaitKey(0);
    cvDestroyWindow("Face Detection Result");
    //釋放記憶體
    cvReleaseMemStorage(&facesMemStorage);
    cvReleaseImage(&tempFrame);
    cvReleaseHaarClassifierCascade(&classifier);
    cvReleaseImage(&image_detect);
    system("pause");
    return EXIT_SUCCESS;
}
