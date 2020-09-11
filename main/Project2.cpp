#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "Matrix.h"
#include <iostream>
/*******************************************************************************
    The following are helper routines with code already written.
    The routines you'll need to write for the assignment are below.
*******************************************************************************/

/*******************************************************************************
Blur a single channel floating point image with a Gaussian.
    image - input and output image
    w - image width
    h - image height
    sigma - standard deviation of Gaussian
*******************************************************************************/
void MainWindow::GaussianBlurImage(double *image, int w, int h, double sigma)
{
    int r, c, rd, cd, i;
    int radius = max(1, (int) (sigma*3.0));
    int size = 2*radius + 1;
    double *buffer = new double [w*h];

    memcpy(buffer, image, w*h*sizeof(double));

    if(sigma == 0.0)
        return;

    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        double dist = (double) (i - radius);

        kernel[i] = exp(-(dist*dist)/(2.0*sigma*sigma));
    }

    double denom = 0.000001;

    for(i=0;i<size;i++)
        denom += kernel[i];
    for(i=0;i<size;i++)
        kernel[i] /= denom;

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double val = 0.0;
            double denom = 0.0;

            for(rd=-radius;rd<=radius;rd++)
                if(r + rd >= 0 && r + rd < h)
                {
                     double weight = kernel[rd + radius];

                     val += weight*buffer[(r + rd)*w + c];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }

    memcpy(buffer, image, w*h*sizeof(double));

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double val = 0.0;
            double denom = 0.0;

            for(cd=-radius;cd<=radius;cd++)
                if(c + cd >= 0 && c + cd < w)
                {
                     double weight = kernel[cd + radius];

                     val += weight*buffer[r*w + c + cd];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }


    delete [] kernel;
    delete [] buffer;
}


/*******************************************************************************
Bilinearly interpolate image (helper function for Stitch)
    image - input image
    (x, y) - location to interpolate
    rgb - returned color values
*******************************************************************************/
bool MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{

    int r = (int) y;
    int c = (int) x;
    double rdel = y - (double) r;
    double cdel = x - (double) c;
    QRgb pixel;
    double del;

    rgb[0] = rgb[1] = rgb[2] = 0.0;

    if(r >= 0 && r < image->height() - 1 && c >= 0 && c < image->width() - 1)
    {
        pixel = image->pixel(c, r);
        del = (1.0 - rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r);
        del = (1.0 - rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c, r+1);
        del = (rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r+1);
        del = (rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);
    }
    else
        return false;

    return true;
}


/*******************************************************************************
Draw detected Harris corners
    cornerPts - corner points
    numCornerPts - number of corner points
    imageDisplay - image used for drawing

    Draws a red cross on top of detected corners
*******************************************************************************/
void MainWindow::DrawCornerPoints(CIntPt *cornerPts, int numCornerPts, QImage &imageDisplay)
{
   int i;
   int r, c, rd, cd;
   int w = imageDisplay.width();
   int h = imageDisplay.height();

   for(i=0;i<numCornerPts;i++)
   {
       c = (int) cornerPts[i].m_X;
       r = (int) cornerPts[i].m_Y;

       for(rd=-2;rd<=2;rd++)
           if(r+rd >= 0 && r+rd < h && c >= 0 && c < w)
               imageDisplay.setPixel(c, r + rd, qRgb(255, 0, 0));

       for(cd=-2;cd<=2;cd++)
           if(r >= 0 && r < h && c + cd >= 0 && c + cd < w)
               imageDisplay.setPixel(c + cd, r, qRgb(255, 0, 0));
   }
}

/*******************************************************************************
Compute corner point descriptors
    image - input image
    cornerPts - array of corner points
    numCornerPts - number of corner points

    If the descriptor cannot be computed, i.e. it's too close to the boundary of
    the image, its descriptor length will be set to 0.

    I've implemented a very simple 8 dimensional descriptor.  Feel free to
    improve upon this.
*******************************************************************************/
void MainWindow::ComputeDescriptors(QImage image, CIntPt *cornerPts, int numCornerPts)
{
    int r, c, cd, rd, i, j;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    // Descriptor parameters
    double sigma = 2.0;
    int rad = 4;

    // Computer descriptors from green channel
    for(r=0;r<h;r++)
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[r*w + c] = (double) qGreen(pixel);
        }

    // Blur
    GaussianBlurImage(buffer, w, h, sigma);

    // Compute the desciptor from the difference between the point sampled at its center
    // and eight points sampled around it.
    for(i=0;i<numCornerPts;i++)
    {
        int c = (int) cornerPts[i].m_X;
        int r = (int) cornerPts[i].m_Y;

        if(c >= rad && c < w - rad && r >= rad && r < h - rad)
        {
            double centerValue = buffer[(r)*w + c];
            int j = 0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                    if(rd != 0 || cd != 0)
                {
                    cornerPts[i].m_Desc[j] = buffer[(r + rd*rad)*w + c + cd*rad] - centerValue;
                    j++;
                }

            cornerPts[i].m_DescSize = DESC_SIZE;
        }
        else
        {
            cornerPts[i].m_DescSize = 0;
        }
    }

    delete [] buffer;
}

/*******************************************************************************
Draw matches between images
    matches - matching points
    numMatches - number of matching points
    image1Display - image to draw matches
    image2Display - image to draw matches

    Draws a green line between matches
*******************************************************************************/
void MainWindow::DrawMatches(CMatches *matches, int numMatches, QImage &image1Display, QImage &image2Display)
{
    int i;
    // Show matches on image
    QPainter painter;
    painter.begin(&image1Display);
    QColor green(0, 250, 0);
    QColor red(250, 0, 0);

    for(i=0;i<numMatches;i++)
    {
        painter.setPen(green);
        painter.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter.setPen(red);
        painter.drawEllipse((int) matches[i].m_X1-1, (int) matches[i].m_Y1-1, 3, 3);
    }

    QPainter painter2;
    painter2.begin(&image2Display);
    painter2.setPen(green);

    for(i=0;i<numMatches;i++)
    {
        painter2.setPen(green);
        painter2.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter2.setPen(red);
        painter2.drawEllipse((int) matches[i].m_X2-1, (int) matches[i].m_Y2-1, 3, 3);
    }

}


/*******************************************************************************
Given a set of matches computes the "best fitting" homography
    matches - matching points
    numMatches - number of matching points
    h - returned homography
    isForward - direction of the projection (true = image1 -> image2, false = image2 -> image1)
*******************************************************************************/
bool MainWindow::ComputeHomography(CMatches *matches, int numMatches, double h[3][3], bool isForward)
{
    int error;
    int nEq=numMatches*2;

    dmat M=newdmat(0,nEq,0,7,&error);
    dmat a=newdmat(0,7,0,0,&error);
    dmat b=newdmat(0,nEq,0,0,&error);

    double x0, y0, x1, y1;

    for (int i=0;i<nEq/2;i++)
    {
        if(isForward == false)
        {
            x0 = matches[i].m_X1;
            y0 = matches[i].m_Y1;
            x1 = matches[i].m_X2;
            y1 = matches[i].m_Y2;
        }
        else
        {
            x0 = matches[i].m_X2;
            y0 = matches[i].m_Y2;
            x1 = matches[i].m_X1;
            y1 = matches[i].m_Y1;
        }


        //Eq 1 for corrpoint
        M.el[i*2][0]=x1;
        M.el[i*2][1]=y1;
        M.el[i*2][2]=1;
        M.el[i*2][3]=0;
        M.el[i*2][4]=0;
        M.el[i*2][5]=0;
        M.el[i*2][6]=(x1*x0*-1);
        M.el[i*2][7]=(y1*x0*-1);

        b.el[i*2][0]=x0;
        //Eq 2 for corrpoint
        M.el[i*2+1][0]=0;
        M.el[i*2+1][1]=0;
        M.el[i*2+1][2]=0;
        M.el[i*2+1][3]=x1;
        M.el[i*2+1][4]=y1;
        M.el[i*2+1][5]=1;
        M.el[i*2+1][6]=(x1*y0*-1);
        M.el[i*2+1][7]=(y1*y0*-1);

        b.el[i*2+1][0]=y0;

    }
    int ret=solve_system (M,a,b);
    if (ret!=0)
    {
        freemat(M);
        freemat(a);
        freemat(b);

        return false;
    }
    else
    {
        h[0][0]= a.el[0][0];
        h[0][1]= a.el[1][0];
        h[0][2]= a.el[2][0];

        h[1][0]= a.el[3][0];
        h[1][1]= a.el[4][0];
        h[1][2]= a.el[5][0];

        h[2][0]= a.el[6][0];
        h[2][1]= a.el[7][0];
        h[2][2]= 1;
    }

    freemat(M);
    freemat(a);
    freemat(b);

    return true;
}
//************************************************************
//************    h e l p e r s  **********************************************
//**************************************************************
//*************************************************************
// Convolve the image with the kernel
void Convolution(double* image, double *kernel, int kernelWidth, int kernelHeight, bool add,int imageHeight, int imageWidth)
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
    double* buffer = new double [bufferCap];

    // initialize my buffer with zeropadding
    if (flag_zeropadding)
    {
        for (int i =0;i<bufferCap;i++)
        {
            buffer[i] =0;
        }
    }
    // initialize my buffer with reflected padding
    else if(flag_reflectedpadding){
        /*
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
        */
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
            buffer[bufferPointer] = image[Imagepointer];
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
                    RGB[0]+=buffer[rel_bufferPointer]*weight;
                }
            }
            image[r*imageWidth+c] = RGB[0];
        }
    }
    delete [] buffer;

}

//************************************
//***********************************
//**********************************
// NMS ver 1
/*
int NMS(double* image,double thres,int w, int h)
{
    double* buffer = new double[w*h];
    int numCorPts = 0;
    for (int i =0;i<w*h;i++)
    {
        buffer[i]= image[i];
    }
    for (int i=0;i<w;i++)
    {
        for (int j=0;j<h;j++)
        {
            bool PeakChecker = true;
            if(buffer[i*w+j]<thres)
            {
                PeakChecker = false;
            }
            // compare to neighbors
            for (int rx=-1;rx<=1 && PeakChecker;rx++)
            {
                for(int ry=-1;ry<=1 && PeakChecker;ry++)
                {
                    int neiberX = i + rx;
                    int neiberY = j + ry;
                    if (neiberX>=0 && neiberY>=0 && neiberX<w && neiberY<h && ~(neiberX == i && neiberY ==j))
                    {
                        if (buffer[i*w+j]<=buffer[neiberX*w+neiberY])
                        {
                            PeakChecker = false;
                        }
                    }
                }
            }
            if (PeakChecker)
            {
                image[i*w+j] = 255.0;
                numCorPts++;
            }
            else {
                image[i*w+j] = 0.0;
            }
        }

    }
    delete [] buffer;
    return numCorPts;
}
*/
int NMS(double* image,double thres,int w, int h)
{
    double* buffer = new double[w*h];
    int numCorPts = 0;
    for (int i =0;i<w*h;i++)
    {
        buffer[i]= image[i];
    }
    for (int x=0;x<w;x++)
    {
        for (int y=0;y<h;y++)
        {
            bool PeakChecker = true;
            if(buffer[y*w+x]<thres)
            {
                PeakChecker = false;
            }
            // compare to neighbors
            for (int j=-1;j<=1 && PeakChecker;j++)
            {
                for(int i=-1;i<=1 && PeakChecker;i++)
                {
                    int neiberX = i + x;
                    int neiberY = j + y;
                    if (neiberX>=0 && neiberY>=0 && neiberX<w && neiberY<h && (neiberX != x || neiberY !=y))
                    {
                        if (buffer[y*w+x]<=buffer[neiberY*w+neiberX])
                        {
                            PeakChecker = false;
                        }
                    }
                }
            }
            if (PeakChecker)
            {
                image[y*w+x] = 255.0;
                numCorPts++;
            }
            else {
                image[y*w+x] = 0.0;
            }
        }

    }
    delete [] buffer;
    return numCorPts;
}
/*******************************************************************************
*******************************************************************************
*******************************************************************************

    The routines you need to implement are below

*******************************************************************************
*******************************************************************************
*******************************************************************************/


/*******************************************************************************
Detect Harris corners.
    image - input image
    sigma - standard deviation of Gaussian used to blur corner detector
    thres - Threshold for detecting corners
    cornerPts - returned corner points
    numCornerPts - number of corner points returned
    imageDisplay - image returned to display (for debugging)
*******************************************************************************/
void MainWindow::HarrisCornerDetector(QImage image, double sigma, double thres, CIntPt **cornerPts, int &numCornerPts, QImage &imageDisplay)
{
    int r, c;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    numCornerPts = 0;

    // Compute the corner response using just the green channel
    for(r=0;r<h;r++)
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);

            buffer[r*w + c] = (double) qGreen(pixel);
        }

    // Write your Harris corner detection code here.
    double *I_x = new double [w*h];
    double *I_y = new double [w*h];
    double *I_xy = new double [w*h];
    // Compute the corner response using just the green channel
    for (r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            I_x[r*w + c] = (double) qGreen(pixel);
            I_y[r*w + c] = (double) qGreen(pixel);
            I_xy[r*w + c] = (double) qGreen(pixel);
        }
    }
    // diravitive first convolution
    double ker[3] = {-1,0,1};
    Convolution(I_x,ker,3,1,false,h,w);
    Convolution(I_y,ker,1,3,false,h,w);
    // generate Ix^2 Iy^2 IX*Iy
    for (int i =0;i<w*h;i++)
    {
        I_xy[i] = I_x[i]*I_y[i];
        I_x[i] = I_x[i]*I_x[i];
        I_y[i] = I_y[i]*I_y[i];
    }
    // gaussianblur
    GaussianBlurImage(I_x,w,h,sigma);
    GaussianBlurImage(I_y,w,h,sigma);
    GaussianBlurImage(I_xy,w,h,sigma);
    // get harris corner matrix H and R calculation
    double determine = 0;
    double trace = 0;
    double CornerResponce = 0;
    for (int i =0;i<w*h;i++)
    {
        trace = I_x[i]+I_y[i];
        determine = I_x[i]*I_y[i]-I_xy[i]*I_xy[i];
        if (trace!=0.0)
        {
            CornerResponce = determine/trace;
        }
        else{
            CornerResponce =0;
        }

        if (thres<CornerResponce)
        {
            buffer[i] = CornerResponce;
            //std::cout << "corR"<<CornerResponce << std::endl;
        }
        else {
            buffer[i] =0;
        }
    }
    // rescale to 255

    double maxR = 0.0;
    for (int i=0;i<w*h;i++)
    {
        if (buffer[i]>maxR)
        {
            maxR=buffer[i];
        }
    }
    for (int i=0;i<w*h;i++)
    {
        buffer[i] = buffer[i] * 255/maxR;
        //std::cout << "buffer value"<<buffer[i] << std::endl;
    }

    // get corner points from NMS
    numCornerPts = NMS(buffer,thres,w,h);
    std::cout << "number of corners"<<numCornerPts << std::endl;
    // Once you know the number of corner points allocate an array as follows:
    *cornerPts = new CIntPt [numCornerPts];
    // let's get corner x and y
    int counter = 0;
    for (int x =0;x<w;x++)
    {
        for (int y=0;y<h;y++)
        {
            if (buffer[y*w+x]>0.0)
            {
                (*cornerPts)[counter].m_X = x;
                (*cornerPts)[counter].m_Y = y;
                counter++;
            }
        }
    }
    // Access the values using: (*cornerPts)[i].m_X = 5.0;
    //
    // The position of the corner point is (m_X, m_Y)
    // The descriptor of the corner point is stored in m_Desc
    // The length of the descriptor is m_DescSize, if m_DescSize = 0, then it is not valid.

    // Once you are done finding the corner points, display them on the image
    DrawCornerPoints(*cornerPts, numCornerPts, imageDisplay);

    delete [] buffer;
    delete [] I_y;
    delete [] I_x;
    delete [] I_xy;
}


/*******************************************************************************
Find matching corner points between images.
    image1 - first input image
    cornerPts1 - corner points corresponding to image 1
    numCornerPts1 - number of corner points in image 1
    image2 - second input image
    cornerPts2 - corner points corresponding to image 2
    numCornerPts2 - number of corner points in image 2
    matches - set of matching points to be returned
    numMatches - number of matching points returned
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::MatchCornerPoints(QImage image1, CIntPt *cornerPts1, int numCornerPts1,
                             QImage image2, CIntPt *cornerPts2, int numCornerPts2,
                             CMatches **matches, int &numMatches, QImage &image1Display, QImage &image2Display)
{

    //***********************************************************************
    // ver 4 (previously, I don't expand my matches but maybe it's a good try)

    numMatches= 0;

    // Compute the descriptors for each interest point.
    // You can access the descriptor for each interest point using interestPts1[i].m_Desc[j].
    // If interestPts1[i].m_DescSize = 0, it was not able to compute a descriptor for that point
    ComputeDescriptors(image1, cornerPts1, numCornerPts1);
    ComputeDescriptors(image2, cornerPts2, numCornerPts2);
    int matchCounter = 0;
    for (int i1 = 0; i1 < numCornerPts1; i1++)
    {
        // check the corner descriptor of image1 size
        if (cornerPts1[i1].m_DescSize == 0)
        {
            continue;
        }
        if (numMatches == matchCounter)
        {
            // dynamically increase my matches size
            int expand = matchCounter + 10;
            CMatches *temp = new CMatches[expand];
            if (numMatches > 0)
            {
                for (int i = 0; i < matchCounter; i++)
                {
                    temp[i] = (*matches)[i];
                }
                delete[] (*matches);
            }
            *matches = temp;
            matchCounter = expand;
        }

        int closeImg2Pts = -1;
        double DistImg2Pts = 0;

        for (int img2 = 0; img2 < numCornerPts2; img2++)
        {
            // check the corner descriptor of image2 size
            if (cornerPts2[img2].m_DescSize == 0)
            {
                continue;
            }
            double dist = 0;
            // 8 bins
            for (int m = 0; m < 8; m++)
            {
                 dist += pow(cornerPts1[i1].m_Desc[m]-cornerPts2[img2].m_Desc[m], 2);
            }
            dist= sqrt( dist);
            // if new dist is smaller then just replace the previous one
            if (closeImg2Pts == -1 ||  dist < DistImg2Pts)
            {
                closeImg2Pts = img2;
                DistImg2Pts =  dist;
            }
        }

        // If we found a good match, add it to the array!
        if (closeImg2Pts != -1)
        {
            (*matches)[numMatches].m_X1 = cornerPts1[i1].m_X;
            (*matches)[numMatches].m_Y1 = cornerPts1[i1].m_Y;
            (*matches)[numMatches].m_X2 = cornerPts2[closeImg2Pts].m_X;
            (*matches)[numMatches].m_Y2 = cornerPts2[closeImg2Pts].m_Y;
            numMatches++;
        }
    }

    // get rid of those matches we don't need
    if (numMatches != matchCounter)
    {
        CMatches *temp = new CMatches[numMatches];
        for (int i = 0; i < numMatches; i++)
        {
            temp[i] = (*matches)[i];
        }
        delete[] (*matches);
        *matches = temp;
    }

    // Draw the matches
    DrawMatches(*matches, numMatches, image1Display, image2Display);
}

/*******************************************************************************
Project a point (x1, y1) using the homography transformation h
    (x1, y1) - input point
    (x2, y2) - returned point
    h - input homography used to project point
*******************************************************************************/
void MainWindow::Project(double x1, double y1, double &x2, double &y2, double h[3][3])
{
    // Add your code here.

    double temp_x = h[0][0]*x1+h[0][1]*y1+h[0][2];
    double temp_y = h[1][0]*x1+h[1][1]*y1+h[1][2];
    double temp_u = h[2][0]*x1+h[2][1]*y1+h[2][2]; // once have wrong in here shoot
    x2 = temp_x/temp_u;
    y2 = temp_y/temp_u;
}

/*******************************************************************************
Count the number of inliers given a homography.  This is a helper function for RANSAC.
    h - input homography used to project points (image1 -> image2
    matches - array of matching points
    numMatches - number of matchs in the array
    inlierThreshold - maximum distance between points that are considered to be inliers

    Returns the total number of inliers.
*******************************************************************************/
int MainWindow::ComputeInlierCount(double h[3][3], CMatches *matches, int numMatches, double inlierThreshold)
{
    // Add your code here.
    int numIn = 0;
    double OG_x,OG_y;
    double pro_x,pro_y;
    double dist;
    for (int i =0;i<numMatches;i++)
    {
        OG_x = matches[i].m_X1;
        OG_y = matches[i].m_Y1;
        pro_x = matches[i].m_X2;
        pro_y = matches[i].m_Y2;
        Project(OG_x,OG_y,pro_x,pro_y,h);
        dist = ((sqrt(pow(pro_y-matches[i].m_Y2,2)+0.01+pow(pro_x-matches[i].m_X2,2))));
        std::cout<<"pro_y = "<<pro_y <<std::endl;
        std::cout<<"matches[i].m_Y2 = "<<matches[i].m_Y2 <<std::endl;
        std::cout<<"pro_x = "<<pro_x <<std::endl;
        std::cout<<"matches[i].m_X2 = "<<matches[i].m_X2 <<std::endl;
        std::cout<<"dist = "<<dist <<std::endl;
        if (dist<inlierThreshold)
        {
            numIn++;
        }
    }
    return numIn;
    //return 0;
}


/*******************************************************************************
Compute homography transformation between images using RANSAC.
    matches - set of matching points between images
    numMatches - number of matching points
    numIterations - number of iterations to run RANSAC
    inlierThreshold - maximum distance between points that are considered to be inliers
    hom - returned homography transformation (image1 -> image2)
    homInv - returned inverse homography transformation (image2 -> image1)
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::RANSAC(CMatches *matches, int numMatches, int numIterations, double inlierThreshold,
                        double hom[3][3], double homInv[3][3], QImage &image1Display, QImage &image2Display)
{
    // retry
    // ver 3 (doesn't show at first time, due to wrong projection pro_x)
    // Add your code here.

    int randSelectNum = 4*2;
    int curInNum, maxInNum =0;
    double h[3][3];
    for (int i =0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            h[i][j]=0;
        }
    }
    CMatches *SelectMatch = new CMatches[randSelectNum];
    // start 200 iterations to get best homography matrix
    for (int i =0;i<numIterations;i++)
    {
        for (int j=0;j<randSelectNum;j++)
        {
            SelectMatch[j]=matches[rand()%numMatches];
        }
        if (ComputeHomography(SelectMatch,randSelectNum,h,true))
        {
            curInNum=ComputeInlierCount(h,SelectMatch,randSelectNum,inlierThreshold);
            if(maxInNum<curInNum)
            {
                maxInNum=curInNum;
                for(int m=0;m<3;m++)
                {
                    for(int n=0;n<3;n++)
                    {
                        hom[m][n] = h[m][n];
                    }
                }
            }
        }
    }
    // let all inliers go to the corresponding points
    int finalNumIn = ComputeInlierCount(hom,matches,numMatches,inlierThreshold);
    CMatches* finalInliars = new CMatches[finalNumIn];
    int counter = 0;
    for (int i =0;i<numMatches;i++)
    {
        if(ComputeInlierCount(hom,&matches[i],1,inlierThreshold)==1)
        {
            finalInliars[counter].m_X1 = matches[i].m_X1;
            finalInliars[counter].m_Y1 = matches[i].m_Y1;
            finalInliars[counter].m_X2 = matches[i].m_X2;
            finalInliars[counter].m_Y2 = matches[i].m_Y2;
            counter++;
        }
    }
    // get HOM and HOM inverse
    ComputeHomography(finalInliars,finalNumIn,hom,true);
    ComputeHomography(finalInliars,finalNumIn,homInv,false);
    // After you're done computing the inliers, display the corresponding matches.
    DrawMatches(finalInliars, finalNumIn, image1Display, image2Display);

}

/*******************************************************************************
Stitch together two images using the homography transformation
    image1 - first input image
    image2 - second input image
    hom - homography transformation (image1 -> image2)
    homInv - inverse homography transformation (image2 -> image1)
    stitchedImage - returned stitched image
*******************************************************************************/
void MainWindow::Stitch(QImage image1, QImage image2, double hom[3][3], double homInv[3][3], QImage &stitchedImage)
{

    bool flag_no_blending = true;
    // Width and height of stitchedImage
    // ver 5 works

    if (flag_no_blending)
    {
        int ImgWid1 = image1.width();
        int ImgHei1 = image1.height();
        int ImgWid2 = image2.width();
        int ImgHei2 = image2.height();
        double corner[4][2];
        Project(0, 0, corner[0][0], corner[0][1], homInv);
        Project(ImgWid2-1, 0, corner[1][0], corner[1][1], homInv);
        Project(ImgWid2-1, ImgHei2-1, corner[2][0], corner[2][1], homInv);
        Project(0, ImgHei2-1, corner[3][0], corner[3][1], homInv);
        int xoff = abs(min(0, int(floor(min(min(corner[0][0], corner[1][0]), min(corner[2][0], corner[3][0]))))));
        int yoff = abs(min(0, int(floor(min(min(corner[0][1], corner[1][1]), min(corner[2][1], corner[3][1]))))));
        int totalWidth = xoff + max(ImgWid1, int(ceil(max(max(corner[0][0], corner[1][0]), max(corner[2][0], corner[3][0])))));
        int totalHeight = yoff + max(ImgHei1, int(ceil(max(max(corner[0][1],corner[1][1]), max(corner[2][1], corner[3][1])))));

        stitchedImage = QImage(totalWidth, totalHeight, QImage::Format_RGB32);
        stitchedImage.fill(qRgb(0,0,0));

        // Copy image1 into stitchedImage
        for (int i = 0; i < ImgHei1; i++)
        {
            for (int j = 0; j < ImgWid1; j++)
            {
                stitchedImage.setPixel(xoff+j, yoff+i, image1.pixel(j, i));
            }
        }
        for (int i = 0; i < totalHeight; i++)
        {
            for (int j = 0; j < totalWidth; j++)
            {
                double Pro_x;
                double Pro_y;
                Project(j-xoff, i-yoff, Pro_x, Pro_y, hom);

                if (0 <= Pro_x &&Pro_x < ImgWid2 && 0 <= Pro_y && Pro_y < ImgHei2)
                {
                    double rgb[3];
                    BilinearInterpolation(&image2, Pro_x, Pro_y, rgb);

                    stitchedImage.setPixel(j, i, qRgb(int(floor(rgb[0])),int(floor(rgb[1])),int(floor(rgb[2]))));
                }
            }
        }
    }



    /*
    for (int r = 0; r < totalHeight; r++)
    {
        for (int c = 0; c < totalWidth; c++)
        {
            double pixel = stitchedImage.pixel(c, r);
            if ( (qGreen(pixel)<10) && (qRed(pixel)<10) && (qBlue(pixel)<10))
            {
                double R = 0;
                double G = 0;
                double B = 0;
                if (c-1>0 && r-1>0 && c<totalWidth-2 && r<totalHeight-2)
                {
                    for (int x=-1;x<2;x++)
                    {
                        for(int y=-1;y<2;y++)
                        {
                            R+=(double) qRed(stitchedImage.pixel(c+x, r+y));
                            G+=(double) qGreen(stitchedImage.pixel(c+x, r+y));
                            B+=(double) qBlue(stitchedImage.pixel(c+x, r+y));
                        }
                    }
                    R=R/8;
                    B=B/8;
                    G=G/8;
                    stitchedImage.setPixel(c, r, qRgb(int(floor(R)),int(floor(G)),int(floor(B))));
                }
                else {
                    continue;
                }

            }
        }

    }
    */


    //*************************************************************************


    //*****************************************************************************
    // ver 7 blending (something not improved enough, but looks fine)

    // Width and height of stitchedImage
    else {
        int stitchWidth = 0;
        int stitchHeigh = 0;
        // image1 and 2, W and H
        int imgWid1 = image1.width();
        int imghei1 = image1.height();
        int imgWid2 = image2.width();
        int imghei2 = image2.height();
        // Add your code to compute ws and hs here.
        //Compute the size of "stitchedImage. "
        //To do this project the four corners of "image2" onto "image1" using
        //Project and "homInv". Allocate the image.
        double stitchSize[4][2];
        Project(        0,        0,stitchSize[0][0],stitchSize[0][1],homInv);
        Project(imgWid2-1,imghei2-1,stitchSize[3][0],stitchSize[3][1],homInv);
        Project(        0,imghei2-1,stitchSize[1][0],stitchSize[1][1],homInv);
        Project(imgWid2-1,        0,stitchSize[2][0],stitchSize[2][1],homInv);
        // compute size
        double WidMin = 1000000;
        double HeiMin = 1000000;
        double WidMax = 0.00001;
        double HeiMax = 0.00001;

        WidMin = (min(min(stitchSize[0][0],stitchSize[1][0]),min(stitchSize[2][0],stitchSize[3][0])));
        HeiMin = (min(min(stitchSize[0][1],stitchSize[1][1]),min(stitchSize[2][1],stitchSize[3][1])));
        WidMax = (max(max(stitchSize[0][0],stitchSize[1][0]),max(stitchSize[2][0],stitchSize[3][0])));
        HeiMax = (max(max(stitchSize[0][1],stitchSize[1][1]),max(stitchSize[2][1],stitchSize[3][1])));
        //  min > 0
        stitchWidth = abs(min(0,(int)floor(WidMin)))+max(imgWid1,(int)ceil(WidMax));
        stitchHeigh = abs(min(0,(int)floor(HeiMin)))+max(imgWid1,(int)ceil(HeiMax));

        stitchedImage = QImage(stitchWidth, stitchHeigh, QImage::Format_RGB32);
        stitchedImage.fill(qRgb(0,0,0));

        // Add you code to warp image1 and image2 to stitchedImage here.

        // put image1 to stiched image
        for (int i=0;i<imgWid1;i++)
        {
            for(int j=0;j<imghei1;j++)
            {
               stitchedImage.setPixel(i+abs(min(0,(int)floor(WidMin))),j+abs(min(0,(int)floor(HeiMin))),image1.pixel(i,j));
            }
        }
        // For each pixel in "stitchedImage", project the point onto "image2
        //double weight;
        double ioff = abs(min(0,(int)floor(WidMin)));
        double joff = abs(min(0,(int)floor(HeiMin)));
        double interval = 0.4;
        for(int i=0;i<stitchWidth;i++)
        {
            for(int j=0;j<stitchHeigh;j++)
            {
                double new_x2,new_y2,rgb[3];
                Project(i-abs(min(0,(int)floor(WidMin))),j-abs(min(0,(int)floor(HeiMin))),new_x2,new_y2,hom);

                if(new_x2>=0 && new_y2>=0 && new_x2<imgWid2 && new_y2<imghei2)
                {
                    BilinearInterpolation(&image2,new_x2,new_y2,rgb);
                    //blending
                    if (i>abs(min(0,(int)floor(WidMin))) && i<abs(min(0,(int)floor(WidMin)))+imgWid1 && \
                        j>abs(min(0,(int)floor(HeiMin))) && j<abs(min(0,(int)floor(HeiMin)))+imghei1  )
                    {

                        double d1 = min(min(i-ioff,imgWid1-i+ioff),min(j-joff,imghei1-j+joff));
                        double d2 = min(min(new_x2,imgWid2-new_x2),min(new_y2,imghei2-new_y2));
                        double weight = 0;
                        if(d2<interval*imgWid2)
                        {
                            /*
                            if (d2==0.0)
                            {
                                d2=1;
                            }
                            */
                            if (d1<interval*imgWid2 && d1<interval*d2)
                            {
                                weight = 1-d2/(interval*imgWid2);
                            }
                            else {
                                weight = d2/(interval*imgWid2);
                            }

                        }
                        for(int a=0;a<3;a++)
                        {

                            if(a==0)
                            {
                                rgb[a] = rgb[a]*weight+(1-weight)*(double)qRed(image1.pixel(i-ioff,j-joff));
                            }
                            else if (a==1) {
                                rgb[a] = rgb[a]*weight+(1-weight)*(double)qGreen(image1.pixel(i-ioff,j-joff));
                            }
                            else {
                                rgb[a] = rgb[a]*weight+(1-weight)*(double)qBlue(image1.pixel(i-ioff,j-joff));
                            }
                        }
                    }
                    stitchedImage.setPixel(i,j,qRgb(((int)floor(rgb[0])),((int)floor(rgb[1])),((int)floor(rgb[2]))) );
                }
            }
        }
    }



}

