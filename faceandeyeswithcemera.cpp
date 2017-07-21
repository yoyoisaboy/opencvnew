#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<iostream>
#include<stdio.h>

using namespace std;
using namespace cv;

string harrEye = "C:\\opencv\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
string harrFace = "C:\\opencv\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
CascadeClassifier faceCascade;
CascadeClassifier EyeCascade;
string windownName = "Capture faces and eyes ";
void detectAndDiapley(Mat frame);

int main()
{
    Mat frame;

    // load the cascades
    if (!EyeCascade.load(harrEye))
        cout << "load harrEye failed" << endl;
    if (!faceCascade.load(harrFace))
        cout << "load harrFace failed" << endl;

    // read the video stream
    VideoCapture capture(0);
    if (capture.isOpened())
    {
        while (true)
        {
            capture >> frame;

            // apply the cascaders to the frame
            if (!frame.empty())
            {
                detectAndDiapley(frame);
            }
            else
            {
                cout << "input video frame is empty" << endl;
            }
            if (waitKey(30) >= 0)break;
        }
    }
    return 0;
}

void detectAndDiapley(Mat frame)
{
    vector<Rect> faces;
    Mat frameGray;
    cvtColor(frame, frameGray, CV_BGR2GRAY);
    equalizeHist(frameGray, frameGray);

    //Detect faces
    faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (int i = 0; i < faces.size(); i++)
    {
        Point Vertex1(faces[i].x, faces[i].y);
        Point Vertex2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        rectangle(frame, Vertex1, Vertex2, Scalar(0, 0, 255), 2, 8, 0);
        Mat faceROI = frameGray(faces[i]);
        vector<Rect> eyes;

        // detect eyes in each face
        EyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (int j = 0; j < eyes.size(); j++)
        {
            Point center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
      int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            Size axes(eyes[j].width / 2, 13);
            ellipse(frame, center, axes, 0, 0, 360, Scalar(255, 255, 0), 2, 8, 0);
        }
    }
    // show the faces and eyes detected
    imshow(windownName, frame);
}
