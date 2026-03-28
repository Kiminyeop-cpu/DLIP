#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    /* Read Image (Grayscale) */
    Mat src = imread("../../Image/HGU_logo.jpg", 0);

    if (src.empty())
    {
        cout << "Image load failed!" << endl;
        return -1;
    }

    /* Create destination images */
    Mat crop_img = Mat::ones(480, 464, CV_8UC1) * 255; //Background is 255(White)
    Mat trans_img = Mat::ones(src.size(), CV_8UC1) * 255; //Background is 255(White)
    Mat resize_img = Mat::ones(src.size(), CV_8UC1) * 255; //Background is 255(White)

    /* 1. Crop 175x200 */
    int crop_w = 175;
    int crop_h = 200;

    // Start cropping point
    int src_x = (src.cols - crop_w) / 2; //((Original Size - Crop Size)) / 2)
    int src_y = (src.rows - crop_h) / 2; //((Original Size - Crop Size)) / 2)

    // Locate Image
    int dst_start_u = 464 - crop_w;  // 464-175 = 289
    int dst_start_v = 480 - crop_h;  // 480-200 = 280

    for (int v = 0; v < crop_h; v++)
    {
        for (int u = 0; u < crop_w; u++)
        {
            // Read Pixel
            uchar pixel = src.at<uchar>(src_y + v, src_x + u);
            // Paste pixel result
            crop_img.at<uchar>(dst_start_v + v, dst_start_u + u) = pixel;
        }
    }

    /* 2. Translation (move image) */
    int T_half_rows = src.rows / 2; //Original Size / 2
    int T_half_cols = src.cols / 2; //Original Size / 2

    int start_v = (src.rows - T_half_rows) / 2; //center an image
    int start_u = (src.cols - T_half_cols) / 2; //center an image



    for (int v = 0; v < T_half_rows; v++)
    {
        for (int u = 0; u < T_half_cols; u++)
        {
            int src_v = T_half_rows - 1 - v; // Translation 180 degree
            int src_u = T_half_cols - 1 - u; // Translation 180 degree
            // Paste pixel result
            trans_img.at<uchar>(v + start_v, u + start_u) = src.at<uchar>(src_v * 2, src_u * 2);
        }
    }

    /* 3. Resize */
    int half_rows = src.rows / 2; //Original Size / 2
    int half_cols = src.cols / 2; //Original Size / 2


    for (int v = 0; v < half_rows; v++)
    {
        for (int u = 0; u < half_cols; u++)
        {
            resize_img.at<uchar>(v, u) = src.at<uchar>(v * 2, u * 2); // Half Size
        }
    }

    /* Show images */
    imshow("Original", src);
    imshow("Crop (Output #3)", crop_img);
    imshow("Translation (Output #2)", trans_img);
    imshow("Resize (Output #1)", resize_img);

    waitKey(0);
    return 0;
}