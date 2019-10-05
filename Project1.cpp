#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

/***********************************************************************
  This is the only file you need to change for your assignment. The
  other files control the UI (in case you want to make changes.)
************************************************************************/

/***********************************************************************
  The first eight functions provide example code to help get you started
************************************************************************/

// aditional dependancy
void Array_sort(int *array , int n)
{
    // declare some local variables
    int i=0 , j=0 , temp=0;

    for(i=0 ; i<n ; i++)
    {
        for(j=0 ; j<n-1 ; j++)
        {
            if(array[j]>array[j+1])
            {
                temp        = array[j];
                array[j]    = array[j+1];
                array[j+1]  = temp;
            }
        }
    }

    printf("\nThe array after sorting is..\n");
    for(i=0 ; i<n ; i++)
    {
        printf("\narray_1[%d] : %d",i,array[i]);
    }
}

float Find_median(int array[] , int n)
{
    float median=0;

    // if number of elements are even
    if(n%2 == 0)
        median = (array[(n-1)/2] + array[n/2])/2.0;
    // if number of elements are odd
    else
        median = array[n/2];

    return median;
}



// Convert an image to grayscale
void MainWindow::BlackWhiteImage(QImage *image)
{
    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int noiseMag = mag*2;

    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            int red = qRed(pixel), green = qGreen(pixel), blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            // otherwise add the same amount of noise to each channel
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;
                red += noise; green += noise; blue += noise;
            }
            image->setPixel(c, r, qRgb(max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))));
        }
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    int w = image.width();
    int h = image.height();
    QImage buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(int r=0;r<h/2;r++)
        for(int c=0;c<w/2;c++)
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
}

// Round float values to the nearest integer values and make sure the value lies in the range [0,255]
QRgb restrictColor(double red, double green, double blue)
{
    int r = (int)(floor(red+0.5));
    int g = (int)(floor(green+0.5));
    int b = (int)(floor(blue+0.5));

    return qRgb(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)));
}

// Normalize the values of the kernel to sum-to-one
void NormalizeKernel(double *kernel, int kernelWidth, int kernelHeight)
{
    double denom = 0.000001; int i;
    for(i=0; i<kernelWidth*kernelHeight; i++)
        denom += kernel[i];
    for(i=0; i<kernelWidth*kernelHeight; i++)
        kernel[i] /= denom;
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it's not done here.
// Note: This function is written using QImage form of the input image. But all other functions later use the double form
void MainWindow::MeanBlurImage(QImage *image, int radius)
{
    if(radius == 0)
        return;
    int size = 2*radius + 1; // This is the size of the kernel

    // Note: You can access the width and height using 'imageWidth' and 'imageHeight' respectively in the functions you write
    int w = image->width();
    int h = image->height();

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders (zero-padding)
    QImage buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute the kernel to convolve with the image
    double *kernel = new double [size*size];

    for(int i=0;i<size*size;i++)
        kernel[i] = 1.0;

    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // For each pixel in the image...
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            double rgb[3];
            rgb[0] = rgb[1] = rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(int rd=-radius;rd<=radius;rd++)
                for(int cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     //For the functions you write, check the ConvertQImage2Double function to see how to get the pixel value
                     QRgb pixel = buffer.pixel(c + cd + radius, r + rd + radius);

                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }
            // Store the pixel in the image to be returned
            // You need to store the RGB values in the double form of the image
            image->setPixel(c, r, restrictColor(rgb[0],rgb[1],rgb[2]));
        }
    }
    // Clean up (use this carefully)
    delete[] kernel;
}

// Convert QImage to a matrix of size (imageWidth*imageHeight)*3 having double values
void MainWindow::ConvertQImage2Double(QImage image)
{
    // Global variables to access image width and height
    imageWidth = image.width();
    imageHeight = image.height();

    // Initialize the global matrix holding the image
    // This is how you will be creating a copy of the original image inside a function
    // Note: 'Image' is of type 'double**' and is declared in the header file (hence global variable)
    // So, when you create a copy (say buffer), write "double** buffer = new double ....."
    Image = new double* [imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
            Image[i] = new double[3];

    // For each pixel
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
        {
            // Get a pixel from the QImage form of the image
            QRgb pixel = image.pixel(c,r);

            // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
            Image[r*imageWidth+c][0] = (double) qRed(pixel);
            Image[r*imageWidth+c][1] = (double) qGreen(pixel);
            Image[r*imageWidth+c][2] = (double) qBlue(pixel);
        }
}

// Convert the matrix form of the image back to QImage for display
void MainWindow::ConvertDouble2QImage(QImage *image)
{
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
            image->setPixel(c, r, restrictColor(Image[r*imageWidth+c][0], Image[r*imageWidth+c][1], Image[r*imageWidth+c][2]));
}


/**************************************************
 TIME TO WRITE CODE
**************************************************/

/**************************************************
 TASK 1
**************************************************/

// Convolve the image with the kernel
void MainWindow::Convolution(double** image, double *kernel, int kernelWidth, int kernelHeight, bool add)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * kernel: 1-D array of kernel values
 * kernelWidth: width of the kernel
 * kernelHeight: height of the kernel
 * add: a boolean variable (taking values true or false)
*/
{

/*
    // Add your code here
*/

    // check if kernel width is odd or not
    if (((int)kernelWidth%2) == 0 || ((int)kernelHeight%2) == 0)
    {
        throw std::invalid_argument( "all of kernel side leangth should be odd" );
        //return;
    }

    // set up flags for fixed padding or reflected padding
    bool flag_zeropadding = true;
    bool flag_reflectedpadding = !flag_zeropadding;

    int kHW = floor(kernelWidth/2);
    int kHH = floor(kernelHeight/2);
    int bufferCap = (imageHeight+2*kHH)*(imageWidth+2*kHW);
    double** buffer = new double* [bufferCap];

    // initialize my buffer with zeropadding
    if (flag_zeropadding)
    {
        for (int i =0;i<bufferCap;i++)
        {
            buffer[i] = new double[3];
            buffer[i][0] = buffer[i][1] = buffer[i][2] =0;
        }
    }
    // initialize my buffer with reflected padding
    else if(flag_reflectedpadding){
        for (int i =0;i<bufferCap;i++)
        {
            if (i==0)
            {
                buffer[i][0] = image[0][0];
                buffer[i][1] = image[0][1];
                buffer[i][2] = image[0][2];
            }
            else if (i==imageWidth+2*kHW-1) {
                buffer[i][0] = image[imageWidth-1][0];
                buffer[i][1] = image[imageWidth-1][1];
                buffer[i][2] = image[imageWidth-1][2];
            }
            else if (i==(imageWidth+2*kHW)*(imageHeight+2*kHH)-imageWidth-2*kHW) {
                buffer[i][0] = image[(imageWidth-1)*imageHeight+1][0];
                buffer[i][1] = image[(imageWidth-1)*imageHeight+1][1];
                buffer[i][2] = image[(imageWidth-1)*imageHeight+1][2];
            }
            else if (i==(imageWidth+2*kHW)*(imageHeight+2*kHH)-1) {
                buffer[i][0] = image[(imageWidth)*imageHeight-1][0];
                buffer[i][1] = image[(imageWidth)*imageHeight-1][1];
                buffer[i][2] = image[(imageWidth)*imageHeight-1][2];
            }
            else if (0<i<imageWidth+2*kHW-1) {
                buffer[i][0] = image[i-1][0];
                buffer[i][1] = image[i-1][1];
                buffer[i][2] = image[i-1][2];
            }
            else if (((imageWidth+2*kHW)*(imageHeight+2*kHH)-imageWidth-2*kHW)<i<((imageWidth+2*kHW)*(imageHeight+2*kHH)-1)) {
                buffer[i][0] = image[i-imageWidth-2*kHW][0];
                buffer[i][1] = image[i-imageWidth-2*kHW][1];
                buffer[i][2] = image[i-imageWidth-2*kHW][2];
            }
            else if (i%(imageWidth+2*kHW)==0) {
                buffer[i][0] = image[i+1][0];
                buffer[i][1] = image[i+1][1];
                buffer[i][2] = image[i+1][2];
            }
            else if ((i+1)%(imageWidth+2*kHW)==0) {
                buffer[i][0] = image[i-1][0];
                buffer[i][1] = image[i-1][1];
                buffer[i][2] = image[i-1][2];
            }
            else {
                buffer[i][0] = 0.0;
                buffer[i][1] = 0.0;
                buffer[i][2] = 0.0;
            }
        }
    }
    else {
        throw std::invalid_argument( "What kind of padding do you want" );
    }

    // copy the image to my buffer
    for (int r =0; r<imageHeight;r++)
    {
        for (int c=0;c<imageWidth;c++)
        {
            int bufferPointer = (imageWidth+2*kHW)*(r+kHH)+kHW+c;
            int Imagepointer = c+r*imageWidth;
            buffer[bufferPointer][0] = image[Imagepointer][0];
            buffer[bufferPointer][1] = image[Imagepointer][1];
            buffer[bufferPointer][2] = image[Imagepointer][2];

        }
    }

    //doing convolution
    for (int r =0;r<imageHeight;r++)
    {
        for (int c=0;c<imageWidth;c++)
        {
            double RGB[3];
            if (add == true)
            {
                RGB[0]=RGB[1]=RGB[2]=128.0;
            }
            else {
                RGB[0]=RGB[1]=RGB[2]=0.0;
            }
            for (int kr=-kHH;kr<kernelHeight-kHH;kr++)
            {
                for(int kc=-kHW;kc<kernelWidth-kHW;kc++)
                {
                    int rel_bufferPointer = (r+kHH+kr)*(imageWidth+2*kHW)+c+kHW+kc;
                    double weight =kernel[(kr+kHH)*kernelWidth+kc+kHW];
                    RGB[0]+=buffer[rel_bufferPointer][0]*weight;
                    RGB[1]+=buffer[rel_bufferPointer][1]*weight;
                    RGB[2]+=buffer[rel_bufferPointer][2]*weight;
                }
            }
            image[r*imageWidth+c][0] = RGB[0];
            image[r*imageWidth+c][1] = RGB[1];
            image[r*imageWidth+c][2] = RGB[2];
        }
    }





}

/**************************************************
 TASK 2
**************************************************/

// Apply the 2-D Gaussian kernel on an image to blur it
void MainWindow::GaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here

    // check the sigma value to be applicable
    if (sigma<=0)
    {
        return;
    }


    int radius = int (ceil(sigma*3));
    int kernel_monolength = 2*radius+1;
    double* kernel = new double [kernel_monolength*kernel_monolength];
    int kernel_pointer = 0;
    double denominator = (2.0*3.14159*sigma*sigma);
    double exp_deno = 2.0*sigma*sigma;

    // doing gaussian blur convolution
    for (int ker_row=-radius;ker_row<=radius;ker_row++)
    {
        for (int ker_col=-radius;ker_col<=radius;ker_col++)
        {
            kernel[kernel_pointer] = 1/denominator*exp(-(ker_row*ker_row+ker_col*ker_col)/exp_deno);
            kernel_pointer++;
        }
    }
    NormalizeKernel(kernel,kernel_monolength,kernel_monolength);
    Convolution(image,kernel,kernel_monolength,kernel_monolength,false);
    delete[] kernel;



}

/**************************************************
 TASK 3
**************************************************/

// Perform the Gaussian Blur first in the horizontal direction and then in the vertical direction
void MainWindow::SeparableGaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    // check the sigma value to be applicable
    if (sigma<0)
    {
        return;
    }
    int radius = int(ceil(3*sigma));
    int kerMonoLen = 2*radius+1;
    double* kernel = new double [kerMonoLen];
    int ker_pointer = 0;
    double denominator = sqrt(2*3.1415*sigma*sigma);
    double exp_deno = 2*sigma*sigma;

    // doing gaussian blur convolution
    for (int x=-radius;x<=radius;x++)
    {
        kernel[ker_pointer] = 1/denominator*exp(-x*x/exp_deno);
        ker_pointer++;
    }
    NormalizeKernel(kernel,kerMonoLen,1);
    Convolution(image,kernel,kerMonoLen,1,false);
    Convolution(image,kernel,1,kerMonoLen,false);
    delete[] kernel;

    //-------------------

}

/********** TASK 4 (a) **********/

// Compute the First derivative of an image along the horizontal direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_x(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here

    // check the sigma value to be applicable
    if (sigma<=0)
        return;
    double kernel[9] = {0.0,0.0,0.0,
                        -1.0,0.0,1.0,
                        0.0,0,0.0};
    Convolution(image,kernel,3,3,true);
    GaussianBlurImage(image,sigma);

}

/********** TASK 4 (b) **********/

// Compute the First derivative of an image along the vertical direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_y(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    // check the sigma value to be applicable
    if (sigma<=0)
        return;
    double kernel[9] = {0.0,-1.0,0.0,
                        0.0,0.0,0.0,
                        0.0,1.0,0.0};
    Convolution(image,kernel,3,3,true);
    GaussianBlurImage(image,sigma);
}

/********** TASK 4 (c) **********/

// Compute the Second derivative of an image using the Laplacian operator and then apply Gaussian blur
void MainWindow::SecondDerivImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    // check the sigma value to be applicable
    if (sigma<=0)
        return;
    double kernel[9] = {0.0,1.0,0.0,
                        1.0,-4.0,1.0,
                        0.0,1.0,0.0};
    Convolution(image,kernel,3,3,true);
    GaussianBlurImage(image,sigma);


}

/**************************************************
 TASK 5
**************************************************/

// Sharpen an image by subtracting the image's second derivative from the original image
void MainWindow::SharpenImage(double** image, double sigma, double alpha)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
 * alpha: constant by which the second derivative image is to be multiplied to before subtracting it from the original image
*/
{
    // Add your code here

    // create a buffer to store values in original image
    double** buffer = new double*[imageWidth*imageHeight];
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        buffer[i] = new double[3];
        buffer[i][0] = image[i][0];
        buffer[i][1] = image[i][1];
        buffer[i][2] = image[i][2];
    }

    // second derivative of buffer
    SecondDerivImage(buffer,sigma);

    // for each pixel, I use original pizel value subtracted by alpha*(second derivative of buffer -128.0)
    for (int r=0;r<imageHeight;r++)
    {
        for (int c=0;c<imageWidth;c++)
        {
            image[r*imageWidth+c][0] =image[r*imageWidth+c][0] -alpha*(buffer[r*imageWidth+c][0]-128.0);
            image[r*imageWidth+c][1] =image[r*imageWidth+c][1] -alpha*(buffer[r*imageWidth+c][1]-128.0);
            image[r*imageWidth+c][2] =image[r*imageWidth+c][2] -alpha*(buffer[r*imageWidth+c][2]-128.0);

        }
    }
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        delete [] buffer[i];
    }
    delete [] buffer;



}

/**************************************************
 TASK 6
**************************************************/

// Display the magnitude and orientation of the edges in an image using the Sobel operator in both X and Y directions
void MainWindow::SobelImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
*/
{
    // Add your code here
    //create a buffer to store image data
    double** buffer = new double* [imageWidth*imageHeight];
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        buffer[i] = new double [3];
        buffer[i][0] = image[i][0];
        buffer[i][1] = image[i][1];
        buffer[i][2] = image[i][2];
    }

    // convert the RGB to grayscale
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        buffer[i][0] = 0.3*buffer[i][0]+0.59*buffer[i][1]+0.11*buffer[i][2];
        buffer[i][1]= buffer[i][0];
        buffer[i][2]= buffer[i][0];
        image[i][0] = 0.3*image[i][0]+0.59*image[i][1]+0.11*image[i][2];
        image[i][1]= image[i][0];
        image[i][2]= image[i][0];
    }
    // kind of derivative of x direction but focus on horizontal pixels
    double x_dir[9] = {-1.0,0.0,1.0,
                       -2.0,0.0,2.0,
                       -1.0,0.0,1.0};
    // kind of derivative of y direction but focus on vertical pixels
    double y_dir[9] = { -1.0, -2.0, -1.0,
                        0.0, 0.0, 0.0,
                        1.0, 2.0, 1.0};

    // doing convolution on image and buffer
    Convolution(image,x_dir,3,3,false);
    Convolution(buffer,y_dir,3,3,false);

    // divided by 8 to avoid spurious edges
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        image[i][0] = image[i][0]/8;
        buffer[i][0] = buffer[i][0]/8;
    }

    // initialize the magnitude and orientation
    double mag= 0.0;
    double orien = 0.0;
    //int count = 0;
    for (int r=0;r<imageHeight;r++)
    {
        for (int c=0;c<imageWidth;c++)
        {

            mag = sqrt(image[r*imageWidth+c][0]*image[r*imageWidth+c][0]+buffer[r*imageWidth+c][0]*buffer[r*imageWidth+c][0]);
            orien = atan2(buffer[r*imageWidth+c][0],image[r*imageWidth+c][0]);
            image[r*imageWidth+c][0] = mag*4.0*((sin(orien) + 1.0)/2.0);
            image[r*imageWidth+c][1] = mag*4.0*((cos(orien) + 1.0)/2.0);
            image[r*imageWidth+c][2] = mag*4.0 - image[r*imageWidth+c][0] - image[r*imageWidth+c][1];
            //count++;
        }
    }
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        delete [] buffer[i];
    }
    delete [] buffer;


    // Use the following 3 lines of code to set the image pixel values after computing magnitude and orientation
    // Here 'mag' is the magnitude and 'orien' is the orientation angle in radians to be computed using atan2 function
    // (sin(orien) + 1)/2 converts the sine value to the range [0,1]. Similarly for cosine.

    // image[r*imageWidth+c][0] = mag*4.0*((sin(orien) + 1.0)/2.0);
    // image[r*imageWidth+c][1] = mag*4.0*((cos(orien) + 1.0)/2.0);
    // image[r*imageWidth+c][2] = mag*4.0 - image[r*imageWidth+c][0] - image[r*imageWidth+c][1];
}

/**************************************************
 TASK 7
**************************************************/

// Compute the RGB values at a given point in an image using bilinear interpolation.
void MainWindow::BilinearInterpolation(double** image, double x, double y, double rgb[3])
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * x: x-coordinate (corresponding to columns) of the position whose RGB values are to be found
 * y: y-coordinate (corresponding to rows) of the position whose RGB values are to be found
 * rgb[3]: array where the computed RGB values are to be stored
*/
{



    // Add your code here
    // to prevent some position of pixel is out of display border
    if ((x<0)||(y<0)||(x>=imageWidth-1)||(y>=imageHeight-1))
    {
        rgb[0] = rgb[1] = rgb[2]= 0;
        return;
    }

    // initialize the 4 reference points' x y coordinates
    int x_l = floor(x);
    int y_l = floor(y);
    int x_h ;
    int y_h ;
    // avoid x y is an integer than expand 1 unit
    if (ceil(x)==x)
    {
        x_h = int (ceil(x)+1);
    }
    else {
        x_h = int (ceil(x));
    }

    if (ceil(y)==y)
    {
        y_h = int (ceil(y)+1);
    }
    else {
        y_h = int (ceil(y));
    }

    // initialize (or as placeholder for) four reference points pixel RGB value
    double v00 =0.0;
    double v01 =0.0;
    double v10 =0.0;
    double v11 =0.0;

    // points index number
    int p00 = y_l*imageWidth+x_l;
    int p01 = y_h*imageWidth+x_l;
    int p10 = y_l*imageWidth+x_h;
    int p11 = y_h*imageWidth+x_h;

    // doing bilinear interpolation
    for (int i=0;i<3;i++)
    {
        v00 = image[p00][i];
        v01 = image[p01][i];
        v10 = image[p10][i];
        v11 = image[p11][i];
        //rgb[i] = v00*(x_h-x)*(y_h-y)+v10*(x-x_l)*(y_h-y)+v01*(x_h-x)*(y-y_l)+v11*(x_h-x)*(y_h-y);
        rgb[i] = 1/((x_h-x_l)*(y_h-y_l))*(v00*(x_h-x)*(y_h-y)+v01*(x_h-x)*(y-y_l)+v10*(x-x_l)*(y_h-y)+v11*(x-x_l)*(y-y_l));
    }




    ////

}

/*******************************************************************************
 Here is the code provided for rotating an image. 'orien' is in degrees.
********************************************************************************/

// Rotating an image by "orien" degrees
void MainWindow::RotateImage(double** image, double orien)

{
    double radians = -2.0*3.141*orien/360.0;

    // Make a copy of the original image and then re-initialize the original image with 0
    double** buffer = new double* [imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
    {
        buffer[i] = new double [3];
        for(int j = 0; j < 3; j++)
            buffer[i][j] = image[i][j];
        image[i] = new double [3](); // re-initialize to 0
    }

    for (int r = 0; r < imageHeight; r++)
       for (int c = 0; c < imageWidth; c++)
       {
            // Rotate around the center of the image
            double x0 = (double) (c - imageWidth/2);
            double y0 = (double) (r - imageHeight/2);

            // Rotate using rotation matrix
            double x1 = x0*cos(radians) - y0*sin(radians);
            double y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (imageWidth/2);
            y1 += (double) (imageHeight/2);

            double rgb[3];
            BilinearInterpolation(buffer, x1, y1, rgb);

            // Note: "image[r*imageWidth+c] = rgb" merely copies the head pointers of the arrays, not the values
            image[r*imageWidth+c][0] = rgb[0];
            image[r*imageWidth+c][1] = rgb[1];
            image[r*imageWidth+c][2] = rgb[2];
        }
}

/**************************************************
 TASK 8
**************************************************/

// Find the peaks of the edge responses perpendicular to the edges
void MainWindow::FindPeaksImage(double** image, double thres)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
 * thres: threshold value for magnitude
*/
{

    // Add your code here

    // from here to line 830 is Sobel operator in order to get the orientation and magnitude
    double** buffer = new double* [imageWidth*imageHeight];
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        buffer[i] = new double [3];
        buffer[i][0] = image[i][0];
        buffer[i][1] = image[i][1];
        buffer[i][2] = image[i][2];
    }
    // convert the RGB to grayscale
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        buffer[i][0] = 0.3*buffer[i][0]+0.59*buffer[i][1]+0.11*buffer[i][2];
        buffer[i][1]= buffer[i][0];
        buffer[i][2]= buffer[i][0];
        image[i][0] = 0.3*image[i][0]+0.59*image[i][1]+0.11*image[i][2];
        image[i][1]= image[i][0];
        image[i][2]= image[i][0];
    }
    // x_direction sobel mask
    double x_dir[9] = {-1.0,0.0,1.0,
                       -2.0,0.0,2.0,
                       -1.0,0.0,1.0};
    // x_directyon sobel mask
    double y_dir[9] = {-1.0,-2.0,-1.0,
                        0.0, 0.0, 0.0,
                        1.0, 2.0, 1.0};
    Convolution(image,x_dir,3,3,false);
    Convolution(buffer,y_dir,3,3,false);

    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        image[i][0] = image[i][0]/8;
        buffer[i][0] = buffer[i][0]/8;
    }

    double mag= 0.0;
    double orien = 0.0;
    int count = 0;
    for (int r=0;r<imageHeight;r++)
    {
        for (int c=0;c<imageWidth;c++)
        {
            mag = sqrt(image[count][0]*image[count][0]+buffer[count][0]*buffer[count][0]);
            orien = atan2(buffer[count][0],image[count][0]);
            buffer[count][1] = mag;
            image[count][1] = orien;
            count++;
        }
    }


    int i=0;
    for (int r=0;r<imageWidth;r++)
    {
        for (int c=0;c<imageHeight;c++)
        {
            // doing interpolation around certain pixel point
            double e1x = c+cos(image[i][1]);
            double e1y = r+sin(image[i][1]);
            double e2x = c-cos(image[i][1]);
            double e2y = r-sin(image[i][1]);
            double rgb[3];
            BilinearInterpolation(buffer,e1x,e1y,rgb);
            double v1 = rgb[1];
            BilinearInterpolation(buffer,e2x,e2y,rgb);
            double v2 = rgb[1];
            // compare with perpendicular edgemagnitude and check if is satidy the condition to be a peak
            if ((buffer[i][1]>thres)&&(buffer[i][1]>=v1)&&(buffer[i][1]>=v2))
            {
                image[i][0] =image[i][1] =image[i][2] = 255.0;
            }
            else {
                image[i][0] =image[i][1] =image[i][2] = 0.0;
            }
            i++;
        }
    }

    ///////////////////////////////////////////////////////////////
    ////////////////////////
    ///

}

/**************************************************
 TASK 9 (a)
**************************************************/




// Perform K-means clustering on a color image using random seeds
void MainWindow::RandomSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
    double epsilon = 30;
    int max_iter = 100;
    int imgsize =imageWidth*imageHeight;
    // my by buffer store classes
    int* buffer = new int [imgsize];
    double* cost = new double[num_clusters];
    //double* pre_cost = new double [num_clusters];
    double* loss = new double[num_clusters];

    double sum_pre_cost=10.0;
    double sum_cost =1000.0;
    int iter = 0;

    // initiate buffer  (my by buffer store classes)
    for (int i =0;i<imgsize;i++)
    {
        buffer[i]=-1;
    }
    for (int i =0;i<num_clusters;i++)
    {
        cost[i] =0.0;
        //pre_cost[i]=0.0;
    }


    double** cluster = new double*[num_clusters];
    for (int i=0;i<num_clusters;i++)
    {
        cluster[i] = new double[3];
        cluster[i][0] = rand()%256;
        cluster[i][1] = rand()%256;
        cluster[i][2] = rand()%256;
    }


    for (int i=0;i<num_clusters;i++)
    {
        loss[i] = 100.0;
    }

    int* counter = new int[num_clusters];
    for (int i=0;i<num_clusters;i++)
    {
        counter[i] = 0;
    }

    // start my loop
    while ((fabs(sum_pre_cost-sum_cost) >epsilon*num_clusters)&&(iter<=(max_iter-1)))
    {
/*
        for (int i=0;i<num_clusters;i++)
        {
            pre_cost[i] =cost[i];
        }
*/
        sum_pre_cost=sum_cost;
        for (int i=0;i<imgsize;i++)
        {
            // initialize index
            int index=0;
            for(int k =0;k<num_clusters;k++)
            {
                loss[k]=fabs(image[i][0]-cluster[k][0])+fabs(image[i][1]-cluster[k][1])+fabs(image[i][2]-cluster[k][2]);
            }
            for (int j=0;j<num_clusters-1;j++)
            {
                if (loss[j]>loss[j+1])
                {
                    index = j+1;
                }
            }
            buffer[i] = index;
            cost[index] += loss[index];
        }

        // initial clusters
        for (int i=0;i<num_clusters;i++)
        {
            cluster[i][0] =0.0;
            cluster[i][1] =0.0;
            cluster[i][2] =0.0;
        }

        //reset counter for each class
        for (int i =0;i<num_clusters;i++)
        {
            counter[i] =0;
        }

        // accumulate RGB value for each cluster
        for (int i=0;i<imgsize;i++)
        {
            cluster[buffer[i]][0]+=image[i][0];
            cluster[buffer[i]][1]+=image[i][1];
            cluster[buffer[i]][2]+=image[i][2];
            for (int m=0;m<num_clusters;m++)
            {
                // accumulate counter for how many pixels belongs to this image
                if (buffer[i] ==m)
                {
                    counter[m]+=1;
                }
            }
        }
        // reference : Yes How to reinitialize the seed
        for (int i=0;i<num_clusters;i++)
        {
            if (counter[i] !=0)
            {
                cluster[i][0]/=counter[i];
                cluster[i][1]/=counter[i];
                cluster[i][2]/=counter[i];
            }
            else {
                cluster[i][0]=rand()%256;
                cluster[i][1]=rand()%256;
                cluster[i][2]=rand()%256;
            }

        }
        sum_cost =0;
        sum_pre_cost =0;
        for (int i=0;i<num_clusters;i++)
        {
            sum_cost += cost[i];
            //sum_pre_cost+=pre_cost[i];
        }
        iter++;
    }

    // end of my loop


    // assign value to my image
    for (int i=0;i<imgsize;i++)
    {
        image[i][0] = floor(cluster[buffer[i]][0]);
        image[i][1] = floor(cluster[buffer[i]][1]);
        image[i][2] = floor(cluster[buffer[i]][2]);
    }


    delete [] buffer;
    delete [] cost;
    delete [] loss;
    delete [] counter;
    for (int i=0;i<num_clusters;i++)
    {
        delete [] cluster[i];
    }
    delete [] cluster;
}

/**************************************************
 TASK 9 (b)
**************************************************/

// Perform K-means clustering on a color image using seeds from the image itself
void MainWindow::PixelSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
    double epsilon = 30;
    int max_iter = 100;
    int imgsize =imageWidth*imageHeight;
    // my by buffer store classes
    int* buffer = new int [imgsize];
    double* cost = new double[num_clusters];
    //double* pre_cost = new double [num_clusters];
    double* loss = new double[num_clusters];

    double sum_pre_cost=10.0;
    double sum_cost =1000.0;
    int iter = 0;

    // initiate buffer  (my by buffer store classes)
    for (int i =0;i<imgsize;i++)
    {
        buffer[i]=-1;
    }
    for (int i =0;i<num_clusters;i++)
    {
        cost[i] =0.0;
        //pre_cost[i]=0.0;
    }


    double** cluster = new double*[num_clusters];
    for (int i=0;i<num_clusters;i++)
    {
        cluster[i] = new double[3];
        int g =rand()%(imgsize);
        cluster[i][0] = image[g][0];
        cluster[i][1] = image[g][1];
        cluster[i][2] = image[g][2];
    }

    // initialize the loss of each cluster
    for (int i=0;i<num_clusters;i++)
    {
        loss[i] = 100.0;
    }
    //initialize the clusters
    int* counter = new int[num_clusters];
    for (int i=0;i<num_clusters;i++)
    {
        counter[i] = 0;
    }

    // start my loop
    while ((fabs(sum_pre_cost-sum_cost*num_clusters) >epsilon)&&(iter<=(max_iter-1)))
    {
/*
        for (int i=0;i<num_clusters;i++)
        {
            pre_cost[i] =cost[i];
        }
*/
        sum_pre_cost=sum_cost;
        for (int i=0;i<imgsize;i++)
        {
            // initialize index
            int index=0;
            for(int k =0;k<num_clusters;k++)
            {
                loss[k]=fabs(image[i][0]-cluster[k][0])+fabs(image[i][1]-cluster[k][1])+fabs(image[i][2]-cluster[k][2]);
            }
            for (int j=0;j<num_clusters-1;j++)
            {
                if (loss[j]>loss[j+1])
                {
                    index = j+1;
                }
            }
            buffer[i] = index;
            cost[index] += loss[index];
        }

        // initial clusters
        for (int i=0;i<num_clusters;i++)
        {
            cluster[i][0] =0.0;
            cluster[i][1] =0.0;
            cluster[i][2] =0.0;
        }

        //reset counter for each class
        for (int i =0;i<num_clusters;i++)
        {
            counter[i] =0;
        }

        // accumulate RGB value for each cluster
        for (int i=0;i<imgsize;i++)
        {
            cluster[buffer[i]][0]+=image[i][0];
            cluster[buffer[i]][1]+=image[i][1];
            cluster[buffer[i]][2]+=image[i][2];
            for (int m=0;m<num_clusters;m++)
            {
                // accumulate counter for how many pixels belongs to this image
                if (buffer[i] ==m)
                {
                    counter[m]+=1;
                }
            }
        }

        for (int i=0;i<num_clusters;i++)
        {
            if (counter[i] !=0)
            {
                cluster[i][0]/=counter[i];
                cluster[i][1]/=counter[i];
                cluster[i][2]/=counter[i];
            }
            else {
                int k = rand()% imgsize;
                cluster[i][0]=image[k][0];
                cluster[i][1]=image[k][1];
                cluster[i][2]=image[k][2];
            }
        }
        sum_cost =0;
        sum_pre_cost =0;
        for (int i=0;i<num_clusters;i++)
        {
            sum_cost += cost[i];
            //sum_pre_cost+=pre_cost[i];
        }
        iter++;
    }

    // end of my loop


    // assign value to my image
    for (int i=0;i<imgsize;i++)
    {
        image[i][0] = floor(cluster[buffer[i]][0]);
        image[i][1] = floor(cluster[buffer[i]][1]);
        image[i][2] = floor(cluster[buffer[i]][2]);
    }


    delete [] buffer;
    delete [] cost;
    //delete [] pre_cost;
    delete [] loss;
    delete [] counter;
    for (int i=0;i<num_clusters;i++)
    {
        delete [] cluster[i];
    }
    delete [] cluster;
}




/**************************************************
 EXTRA CREDIT TASKS
**************************************************/

// Perform K-means clustering on a color image using the color histogram
void MainWindow::HistogramSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
    double epsilon = 30;
    int max_iter = 100;
    int imgsize =imageWidth*imageHeight;
    // my by buffer store classes
    int* buffer = new int [imgsize];
    double* cost = new double[num_clusters];
    //double* pre_cost = new double [num_clusters];
    double* loss = new double[num_clusters];

    double sum_pre_cost=10.0;
    double sum_cost =1000.0;
    int iter = 0;

    // initiate buffer  (my by buffer store classes)
    for (int i =0;i<imgsize;i++)
    {
        buffer[i]=-1;
    }
    for (int i =0;i<num_clusters;i++)
    {
        cost[i] =0.0;
        //pre_cost[i]=0.0;
    }

    // initialize the RGB loss for each classes
    for (int i=0;i<num_clusters;i++)
    {
        loss[i] = 100.0;
    }
    // initialize the cluster
    int* counter = new int[num_clusters];
    for (int i=0;i<num_clusters;i++)
    {
        counter[i] = 0;
    }

    // create  grayscale image
    int *intensity = new int [imgsize];
    for (int i=0;i<imgsize;i++)
    {
        intensity[i] = round(image[i][0]*0.3+image[i][1]*0.59+image[i][2]*0.11);

    }

    int *histogram = new int [256];
    for (int i=0;i<256;i++)
    {
        histogram[i]=0;
    }

    for (int i=0;i<imgsize;i++)
    {
        histogram[intensity[i]]+=1;
    }

    // initialize the cluster's gray scale value
    double* cluster = new double[num_clusters];
    if (num_clusters>1)
    {
        for (int i=0;i<num_clusters;i++)
        {
            cluster[i]=i*floor(256/(num_clusters-1));
        }
    }
    else {
        throw std::invalid_argument( "number of clusters must be greater than 1" );
    }

    // start the loop to get enough clusters
    while((fabs(sum_pre_cost-sum_cost) >epsilon)&&(iter<=(max_iter-1)))
    {
        sum_pre_cost=sum_cost;
        for (int i=0;i<imgsize;i++)
        {
            // initialize index
            int index=0;
            for(int k =0;k<num_clusters;k++)
            {
                loss[k]=fabs(intensity[i]-cluster[k]);
            }
            for (int j=0;j<num_clusters-1;j++)
            {
                if (loss[j]>loss[j+1])
                {
                    index = j+1;
                }
            }
            // store the class information for which pizel belongs to
            buffer[i] = index;
            // sum up the class loss
            cost[index] += loss[index];
        }

        // initial clusters
        for (int i=0;i<num_clusters;i++)
        {
            cluster[i] =0;
        }

        //reset counter for each class
        for (int i =0;i<num_clusters;i++)
        {
            counter[i] =0;
        }

        // accumulate RGB value for each cluster
        for (int i=0;i<imgsize;i++)
        {
            cluster[buffer[i]]+=intensity[i];

            for (int m=0;m<num_clusters;m++)
            {
                // accumulate counter for how many pixels belongs to this image
                if (buffer[i] ==m)
                {
                    counter[m]+=1;
                }
            }
        }
        // reinitialize the cluster grayscale value
        for (int i=0;i<num_clusters;i++)
        {
            if (counter[i] !=0)
            {
                cluster[i]/=counter[i];
            }
            else {
                cluster[i]=rand()%256;
            }
        }
        // initialize the cost and pre_cost
        sum_cost =0;
        sum_pre_cost =0;
        for (int i=0;i<num_clusters;i++)
        {
            sum_cost += cost[i];
            //sum_pre_cost+=pre_cost[i];
        }
        iter++;
    }
    for (int i=0;i<imgsize;i++)
    {
        image[i][0] = floor(cluster[buffer[i]]);
        image[i][1] = floor(cluster[buffer[i]]);
        image[i][2] = floor(cluster[buffer[i]]);
    }


    delete [] buffer;
    delete [] cost;
    //delete [] pre_cost;
    delete [] loss;
    delete [] counter;
    delete [] cluster;
    delete [] intensity;
    delete [] histogram;


}

// Apply the median filter on a noisy image to remove the noise
void MainWindow::MedianImage(double** image, int radius)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * radius: radius of the kernel
*/
{
    // Add your code here
    double** buffer =new double*[imageWidth*imageHeight];
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        buffer[i] = new double[3];
        buffer[i][0] = buffer[i][1] = buffer [i][2] =0;
    }

    for (int r=0;r<imageHeight;r++)
    {
        for (int c=0;c<imageWidth;c++)
        {

            if (r==0 && c==0)
            {
                int p1=0;
                int p2=1;
                int p3=int (imageWidth+p1);
                int p4=int (imageWidth+p2);
                int *pts = new int [4];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;


                int array[3][4];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<4;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],4);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 4);
                }
                delete [] pts;
            }
            else if (r==0 && c == imageWidth) {
                int p1=int(imageWidth-2);
                int p2=int(imageWidth-1);
                int p3=int (imageWidth+p1);
                int p4=int (imageWidth+p2);
                int *pts = new int [4];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;


                int array[3][4];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<4;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],4);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 4);
                }
                delete [] pts;

            }
            else if (r==imageHeight &&  c==0) {
                int p1=int((imageHeight-2)*imageWidth+1);
                int p2=int((imageHeight-2)*imageWidth+2);
                int p3=int (imageWidth+p1);
                int p4=int (imageWidth+p2);
                int *pts = new int [4];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;


                int array[3][4];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<4;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],4);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 4);
                }
                delete [] pts;
            }
            else if (r==imageHeight && c==imageWidth) {
                int p1=int((imageHeight-1)*imageWidth-1);
                int p2=int((imageHeight-1)*imageWidth);
                int p3=int (imageWidth+p1);
                int p4=int (imageWidth+p2);
                int *pts = new int [4];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;


                int array[3][4];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<4;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],4);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 4);
                }
                delete [] pts;
            }
            else if (r==0 && c!=0 && c!=imageWidth) {

                int p4=int ((r+1)*imageWidth+c-1);
                int p5=int ((r+1)*imageWidth+c);
                int p6=int((r+1)*imageWidth+c+1);
                int p1=p4-imageWidth;
                int p2=p5-imageWidth;
                int p3=p6-imageWidth;
                int *pts = new int [6];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;
                pts[4] = p5;
                pts[5] = p6;

                int array[3][6];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<6;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],6);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 6);
                }
                delete [] pts;

            }
            else if (r==imageHeight && c!=0 && c!=imageWidth) {
                int p4=int (r*imageWidth+c-1);
                int p5=int (r*imageWidth+c);
                int p6=int(r*imageWidth+c+1);
                int p1=p4-imageWidth;
                int p2=p5-imageWidth;
                int p3=p6-imageWidth;
                int *pts = new int [6];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;
                pts[4] = p5;
                pts[5] = p6;

                int array[3][6];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<6;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],6);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 6);
                }
                delete [] pts;
            }
            else if (c==0 && r!=0 && r!=imageHeight) {
                int p1 = (r*imageWidth+c-imageWidth);
                int p2 = (r*imageWidth+c);
                int p3 = (r*imageWidth+c+imageWidth);
                int p4 = p1+1;
                int p5 = p2+1;
                int p6 = p3+1;
                int *pts = new int [6];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;
                pts[4] = p5;
                pts[5] = p6;

                int array[3][6];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<6;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],6);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 6);
                }
                delete [] pts;
            }
            else if (c==imageWidth && r!=0 && r!=imageHeight) {
                int p4 = (r*imageWidth+c-imageWidth);
                int p5 = (r*imageWidth+c);
                int p6 = (r*imageWidth+c+imageWidth);
                int p1 = p4-1;
                int p2 = p5-1;
                int p3 = p6-1;
                int *pts = new int [6];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;
                pts[4] = p5;
                pts[5] = p6;

                int array[3][6];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<6;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],6);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 6);
                }
                delete [] pts;
            }
            else {
                int p4=r*imageWidth+c-1;
                int p5=r*imageWidth+c;
                int p6=r*imageWidth+c+1;
                int p1=p4-imageWidth;
                int p2=p5-imageWidth;
                int p3=p6-imageWidth;
                int p7=p4+imageWidth;
                int p8=p5+imageWidth;
                int p9=p6+imageWidth;
                int *pts = new int [9];
                pts[0] = p1;
                pts[1] = p2;
                pts[2] = p3;
                pts[3] = p4;
                pts[4] = p5;
                pts[5] = p6;
                pts[6] = p7;
                pts[7] = p8;
                pts[8] = p9;

                int array[3][9];
                for (int i=0;i<3;i++)
                {
                    for (int j=0;j<9;j++)
                    {
                        array[i][j] = image[pts[j]][i];
                    }
                }
                for (int i =0;i<3;i++)
                {
                    Array_sort(array[i],9);
                    buffer[r*imageWidth+c][i]= Find_median(array[i] , 9);
                }
                delete [] pts;
            }

        }
    }
    for (int i =0;i<imageWidth*imageHeight;i++)
    {
        image[i][0] = buffer[i][0];
        image[i][1] = buffer[i][1];
        image[i][2] = buffer[i][2];
    }
    for (int i=0;i<imageWidth*imageHeight;i++)
    {
        delete [] buffer[i];
    }
    delete [] buffer;

}

// Apply Bilater filter on an image
void MainWindow::BilateralImage(double** image, double sigmaS, double sigmaI)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigmaS: standard deviation in the spatial domain
 * sigmaI: standard deviation in the intensity/range domain
*/
{
    // Add your code here.  Should be similar to GaussianBlurImage.
}

// Perform the Hough transform
void MainWindow::HoughImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}

// Perform smart K-means clustering
void MainWindow::SmartKMeans(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}

