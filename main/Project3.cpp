#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "stdlib.h"
#include <algorithm>


/**************************************************
CODE FOR K-MEANS COLOR IMAGE CLUSTERING (RANDOM SEED)
**************************************************/
// refine the random seed
void Clustering(QImage *image, int num_clusters, int maxit)
{
        int w = image->width(), h = image->height();
        QImage buffer = image->copy();

        std::vector<QRgb> centers, centers_new;

        //initialize random centers
        int n = 1;
        while (n <= num_clusters)
        {
            QRgb center = qRgb(rand() % 256, rand() % 256, rand() % 256);
            centers.push_back(center);
            centers_new.push_back(center);
            n++;
        }

        //iterative part
        int it = 0;
        std::vector<int> ids;
        while (it < maxit)
        {
                ids.clear();
                //assign pixels to clusters
                for (int r = 0; r < h; r++)
                    for (int c = 0; c < w; c++)
                    {
                        int maxd = 999999, id = 0;
                        for (int n = 0; n < num_clusters; n++)
                        {
                                QRgb pcenter = centers[n];
                                QRgb pnow = buffer.pixel(c, r);
                                int d = abs(qRed(pcenter) - qRed(pnow)) + abs(qGreen(pcenter) - qGreen(pnow)) + abs(qBlue(pcenter) - qBlue(pnow));
                                if (d < maxd)
                                {
                                        maxd = d; id = n;
                                }
                        }
                        ids.push_back(id);
                    }

                //update centers
                std::vector<int> cnt, rs, gs, bs;
                for (int n = 0; n < num_clusters; n++)
                {
                        rs.push_back(0); gs.push_back(0); bs.push_back(0); cnt.push_back(0);
                }
                for (int r = 0; r < h; r++)
                    for (int c = 0; c < w; c++)
                    {
                        QRgb pixel = buffer.pixel(c,r);
                        rs[ids[r * w + c]] += qRed(pixel);
                        gs[ids[r * w + c]] += qGreen(pixel);
                        bs[ids[r * w + c]] += qBlue(pixel);
                        cnt[ids[r * w + c]]++;
                    }
                for (int n = 0; n < num_clusters; n++)
                    if (cnt[n] == 0) // no pixels in a cluster
                        continue;
                    else
                        centers_new[n] = qRgb(rs[n]/cnt[n], gs[n]/cnt[n], bs[n]/cnt[n]);

                centers = centers_new; it++;
        }
        //render results
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                image->setPixel(c, r, qRgb(ids[r * w + c],ids[r * w + c],ids[r * w + c]));
}
/**************************************************
CODE FOR FINDING CONNECTED COMPONENTS
**************************************************/

#include "utils.h"

#define MAX_LABELS 80000

#define I(x,y)   (image[(y)*(width)+(x)])
#define N(x,y)   (nimage[(y)*(width)+(x)])

void uf_union( int x, int y, unsigned int parent[] )
{
    while ( parent[x] )
        x = parent[x];
    while ( parent[y] )
        y = parent[y];
    if ( x != y ) {
        if ( y < x ) parent[x] = y;
        else parent[y] = x;
    }
}

int next_label = 1;

int uf_find( int x, unsigned int parent[], unsigned int label[] )
{
    while ( parent[x] )
        x = parent[x];
    if ( label[x] == 0 )
        label[x] = next_label++;
    return label[x];
}

void conrgn(int *image, int *nimage, int width, int height)
{
    unsigned int parent[MAX_LABELS], labels[MAX_LABELS];
    int next_region = 1, k;

    memset( parent, 0, sizeof(parent) );
    memset( labels, 0, sizeof(labels) );

    for ( int y = 0; y < height; ++y )
    {
        for ( int x = 0; x < width; ++x )
        {
            k = 0;
            if ( x > 0 && I(x-1,y) == I(x,y) )
                k = N(x-1,y);
            if ( y > 0 && I(x,y-1) == I(x,y) && N(x,y-1) < k )
                k = N(x,y-1);
            if ( k == 0 )
            {
                k = next_region; next_region++;
            }
            if ( k >= MAX_LABELS )
            {
                fprintf(stderr, "Maximum number of labels reached. Increase MAX_LABELS and recompile.\n"); exit(1);
            }
            N(x,y) = k;
            if ( x > 0 && I(x-1,y) == I(x,y) && N(x-1,y) != k )
                uf_union( k, N(x-1,y), parent );
            if ( y > 0 && I(x,y-1) == I(x,y) && N(x,y-1) != k )
                uf_union( k, N(x,y-1), parent );
        }
    }
    for ( int i = 0; i < width*height; ++i )
        if ( nimage[i] != 0 )
            nimage[i] = uf_find( nimage[i], parent, labels );

    next_label = 1; // reset its value to its initial value
    return;
}


/**************************************************
 **************************************************
TIME TO WRITE CODE
**************************************************
**************************************************/


// store the co-occurence matrix
void CoMat(QImage image,int r, int c, int delta_r, int delta_c, int w, int h,int ***CoMat, int num_regions)
{

    if ((r<h-delta_r) && (c<w-delta_c))
    {
        int CurPx = (int) qRed(image.pixel(c, r));
        int NxPx = (int) qRed(image.pixel(c+delta_c, r+delta_r));
        CoMat[num_regions][CurPx][NxPx]++;
    }
    return;
}

// get centroid and boundary
void CoMatCenBd(int *nimg, double **cent, double **bd,int r, int c, int w)
{
    int i = nimg[r*w + c] - 1;
    cent[i][0] += r;  //row
    cent[i][1] += c; //column
    cent[i][2] ++;   //count the size of each region (for normalizing the r and c )

    bd[i][0] = (bd[i][0]>r) ? r:bd[i][0];
    bd[i][1] = (bd[i][1]>c) ? c:bd[i][1];
    bd[i][2] = (bd[i][0]<r) ? r:bd[i][2];
    bd[i][3] = (bd[i][0]<c) ? c:bd[i][3];

    return;
}


// clean the region from *img which area< regionThre
int CleanNoise(int w, int h, int *nimg ,int Thres, int num_regions )
{
    // store the region size
    int extra_RGNnum = num_regions+1;
    int* RGNsize = new int[extra_RGNnum];
    // initialize
    for (int i=0; i<extra_RGNnum; i++){
        RGNsize[i] =0;
    }
    for (int r=0; r<h; r++)
        for (int c=0; c<w; c++)
        {
            RGNsize[nimg[r*w+c]] ++;
        }
    // using RGNindexChange to store the region index after thresholding and start from 1 to end
    int RGNcounter = 0;
    int * RGNindexChange = new int [extra_RGNnum];
    for (int i=1; i< extra_RGNnum; i++)
    {
        if (RGNsize[i] > Thres)
        {
            RGNcounter++;
            RGNindexChange[i] = RGNcounter;
        }
        else
        {
            RGNindexChange[i] = 0;
        }
    }
    // store back to nimg
    for (int r=0; r<h; r++)
        for (int c=0; c<w; c++)
        {
            nimg[r*w + c] = RGNindexChange[nimg[r*w + c]];
        }
    // free the memory
    delete[] RGNsize;
    delete[] RGNindexChange;
    return RGNcounter;
}

// get random pixel value to be kind of global feature
void getRand(QImage image,int w, int h,double* getRand,int num_regions)
{
    int x=0;
    int y=0;
    for (int i = 0; i < num_regions; ++i) {
        x=rand()%w;
        y=rand()%h;
        getRand[i]  = (int) qRed(image.pixel(x,y));
        getRand[i] /= 255.0;
    }
    return;

}

// normailize the co-occurrence matrix (divided each value by summation of the values in the same region)
void NormalizeCoMat (int ***CoMat, double ***normCoMat, int num_regions )
{
    int pixelVal = 8;
    for (int i=0; i<num_regions; i++)
    {
        double Sum = 0.0;
        for (int r=0; r<pixelVal; r++)
        {
            for (int c=0; c<pixelVal; c++)
            {
                Sum += (double) CoMat[i][r][c];
            }
        }
        for (int r=0; r<pixelVal; r++)
        {
            for (int c=0; c<pixelVal; c++)
            {
                normCoMat[i][r][c] = (double) CoMat[i][r][c]/Sum;
            }
        }
    }
}

// get value of energy, entropy, contrast,
void CoMatGetEnergyEntropyContrast(double ***normCoMat, double *energy, double *entropy, double *contrast, double *corr,int num_regions)
{
    int kb = 1;
    float colorVal = 8;
    float denom = 5000;
    for (int i=0; i<num_regions; i++)
    {
        for (int r=0; r<colorVal; r++)
        {
            for (int c=0; c<colorVal; c++)
            {
                energy[i] += pow(normCoMat[i][r][c], 2);
                if (normCoMat[i][r][c] > 0)
                {
                    entropy[i] = entropy[i]-kb*(normCoMat[i][r][c]*log(normCoMat[i][r][c]));
                }
                contrast[i] += (r - c)*(r - c)*normCoMat[i][r][c];
                corr[i] += fabs(r - (colorVal-1)/2)*fabs(c-(colorVal-1)/2)/(denom)*normCoMat[i][r][c];
            }
        }

    }
}



// normalize the centroid and boundary
void NormalizeCentBd( int w, int h, double **cent, double **bd,int num_regions)
{
    for (int i = 0; i<num_regions; i++)
    {
        bd[i][4] = (bd[i][3]-bd[i][1])*(bd[i][2]-bd[i][0])  /(w*h) ;
        bd[i][0] /= h;
        bd[i][1] /= w;
        bd[i][2] /= h;
        bd[i][3] /= w;
        cent[i][0] /= (cent[i][2]*h);
        cent[i][1] /= (cent[i][2]*w);
    }
    return;
}
//*********************  debugging my code  *************************************
unsigned long HextoDec(const unsigned char *hex, int length)
{
    int i;
    unsigned long rslt = 0;
    for (i = 0; i < length; i++)
    {
        rslt += (unsigned long)(hex[i]) << (8 * (length - 1 - i));

    }
    return rslt;
}
std::string hexify(unsigned int n)
{
  std::string res;

  do
  {
    res += "0123456789ABCDEF"[n % 16];
    n >>= 4;
  } while(n);

  return std::string(res.rbegin(), res.rend());
}
int hex_char_value(char c)
{
    if(c >= '0' && c <= '9')
        return c - '0';
    else if(c >= 'a' && c <= 'f')
        return (c - 'a' + 10);
    else if(c >= 'A' && c <= 'F')
        return (c - 'A' + 10);
    assert(0);
    return 0;
}
int hex_to_decimal(const char* szHex, int len)
{
    int result = 0;
    for(int i = 0; i < len; i++)
    {
        result += (int)pow((float)16, (int)len-i-1) * hex_char_value(szHex[i]);
    }
    return result;
}
/**************************************************
Code to compute the features of a given image (both database images and query image)
**************************************************/

std::vector<double*> MainWindow::ExtractFeatureVector(QImage image)
{
    /********** STEP 1 **********/

    // Display the start of execution of this step in the progress box of the application window
    // You can use these 2 lines to display anything you want at any point of time while debugging

    ui->progressBox->append(QString::fromStdString("Clustering.."));
    QApplication::processEvents();

    // Perform K-means color clustering
    // This time the algorithm returns the cluster id for each pixel, not the rgb values of the corresponding cluster center
    // The code for random seed clustering is provided. You are free to use any clustering algorithm of your choice from HW 1
    // Experiment with the num_clusters and max_iterations values to get the best result

    int num_clusters = 5;
    int max_iterations = 50;
    QImage image_copy = image;
    Clustering(&image_copy,num_clusters,max_iterations);


    /********** STEP 2 **********/


    ui->progressBox->append(QString::fromStdString("Connecting components.."));
    QApplication::processEvents();

    // Find connected components in the labeled segmented image
    // Code is given, you don't need to change

    int r, c, w = image_copy.width(), h = image_copy.height();
    int *img = (int*)malloc(w*h * sizeof(int));
    memset( img, 0, w * h * sizeof( int ) );
    int *nimg = (int*)malloc(w*h *sizeof(int));
    memset( nimg, 0, w * h * sizeof( int ) );

    for (r=0; r<h; r++)
        for (c=0; c<w; c++)
            img[r*w + c] = qRed(image_copy.pixel(c,r));

    conrgn(img, nimg, w, h);

    int num_regions=0;
    for (r=0; r<h; r++)
        for (c=0; c<w; c++)
            num_regions = (nimg[r*w+c]>num_regions)? nimg[r*w+c]: num_regions;

//    ui->progressBox->append(QString::fromStdString("#regions = "+std::to_string(num_regions)));
//    QApplication::processEvents();

    // The resultant image of Step 2 is 'nimg', whose values range from 1 to num_regions

    // WRITE YOUR REGION THRESHOLDING AND REFINEMENT CODE HERE

    //noise cleaning

    int regionThre = 70;

    num_regions = CleanNoise(w, h, nimg,regionThre, num_regions);


    ui->progressBox->append(QString::fromStdString("#regions = "+std::to_string(num_regions)));
    QApplication::processEvents();

    /********** STEP 3 **********/


    ui->progressBox->append(QString::fromStdString("Extracting features.."));
    QApplication::processEvents();

    // Extract the feature vector of each region

    /*---------------------------------------------------------------------------------*/
    /*---------------------------------------------------------------------------------*/
    /*
    int colorScale = 256;
    int*** RGB_LCM = new int** [num_regions];
    for (int i=0;i<num_regions;i++)
    {
        RGB_LCM[i] = new int*[num_regions];
        for (int j=0;j<colorScale;j++)
        {
            RGB_LCM[i][j] = new int [colorScale];
            for (int k=0;k<colorScale;k++)
            {
                RGB_LCM[i][j][k] = 0;
            }
        }
    }
    */
    /*
    QImage GrayImage = image;
    int gray = 0;
    for (int i=0;i<GrayImage.height();i++)
    {
        for (int j=0;j<GrayImage.width();j++)
        {
            QRgb ind = GrayImage.pixel(j,i);
            double red = qRed(ind);
            double green = qGreen(ind);
            double blue = qBlue(ind);
            gray = (int)0.3*red+0.6*green+0.1*blue;
            gray=(int) (gray/32);
            GrayImage.setPixel(j,i,qRgb(int(gray),int(gray),int(gray)));
        }
        std::cout<<gray<<std::endl;
    }
    */




    //*************************************************************************************
    // pre define variables
    int*** CoOccurMat= new int** [num_regions];
    double*** normCoOccurMat = new double** [num_regions];
    double **boundary = new double*[num_regions];
    double **centroid = new double*[num_regions];
    double* energy = new double[num_regions];
    double* entropy = new double[num_regions];
    double* contrast = new double[num_regions];
    double* correlation = new double[num_regions];
    double* randomSelect = new double[num_regions];

    int pixelVal = 8;
    for (int i =0;i<num_regions;i++)
    {
        randomSelect[i]=0.0;
    }

    for (int i=0; i<num_regions; i++)
    {
        energy[i] = 0.0;
        entropy[i] = 0.0;
        contrast[i] = 0.0;
        correlation[i] = 0.0;
    }

    for (int i=0; i<num_regions; i++)
    {
        CoOccurMat[i] = new int*[pixelVal];
        normCoOccurMat[i] = new double*[pixelVal];
        for (int j=0; j<pixelVal; j++)
        {
            CoOccurMat[i][j] = new int[pixelVal];
            normCoOccurMat[i][j] = new double[pixelVal];
            for (int k=0; k<pixelVal; k++)
            {
                CoOccurMat[i][j][k] = 0;
                normCoOccurMat[i][j][k] = 0.0;
            }
        }
    }

    // initialize centroid
    for (int i=0; i<num_regions; i++)
    {
        centroid[i] = new double[3];    //[0]row [1]column [2] region size ( use for normalize the r and c )
        for (int j=0; j<3; j++){
            centroid[i][j] = 0;
        }
    }

    //initialize boundary box

    for (int i=0; i<num_regions; i++)
    {
        boundary[i] = new double[5];
        boundary[i][0] = h;
        boundary[i][1] = w;
        boundary[i][2] = 0;
        boundary[i][3] = 0;
        boundary[i][4] = 0;
    }

    // RGB to Gray 0 1 2 3 4 5 6 7 8
    QImage grayImgae = image;
    for(int r=0;r<grayImgae.height();r++){
        for(int c=0;c<grayImgae.width();c++)
        {
            QRgb pixel = grayImgae.pixel(c,r);
            double r = (double) qRed(pixel);
            double g = (double) qGreen(pixel);
            double b = (double) qBlue(pixel);
            int greyScale = (int) (0.3*r + 0.6*g + 0.1*b);
            greyScale=(int) (greyScale/32);
            grayImgae.setPixel(c, r, qRgb( (int) greyScale, (int) greyScale, (int) greyScale));
        }
    }
    // get co-occurence matrix
    int dr = 1, dc = 1;
    for (r=0; r<h; r++){
        for (c=0; c<w; c++)
        {
            int indexOfRegion = nimg[r*w+c] - 1;
            if (indexOfRegion>=0)
            {
                CoMat(grayImgae, r, c, dr, dc, w, h,CoOccurMat, indexOfRegion);
                CoMatCenBd(nimg, centroid, boundary, r, c, w);
                getRand(grayImgae,w,h,randomSelect,num_regions);
                getRand(grayImgae,w,h,randomSelect,num_regions);

            }
        }
    }
    // normize the centroid and boundary
    NormalizeCentBd( w, h, centroid, boundary,num_regions);
    // normize the co-occurance matrix
    NormalizeCoMat(CoOccurMat, normCoOccurMat, num_regions);
    // calculate the energy, entropy, contrast
    CoMatGetEnergyEntropyContrast(normCoOccurMat, energy, entropy, contrast, correlation, num_regions);

    /*---------------------------------------------------------------------------------*/
    /*---------------------------------------------------------------------------------*/


    // Set length of feature vector according to the number of features you plan to use.
    featurevectorlength = 16;

    // Initializations required to compute feature vector

    std::vector<double*> featurevector; // final feature vector of the image; to be returned
    double **features = new double* [num_regions]; // stores the feature vector for each connected component
    for(int i=0;i<num_regions; i++){
        features[i] = new double[featurevectorlength](); // initialize with zeros
    }

    // Sample code for computing the mean RGB values and size of each connected component
    for(int r=0; r<h; r++)
    {
        for (int c=0; c<w; c++)
        {
            if ((nimg[r*w+c] - 1) < 0) continue;

            features[nimg[r*w+c]-1][0] += 1; // stores the number of pixels for each connected component
            features[nimg[r*w+c]-1][1] += (double) qRed(image.pixel(c,r));
            features[nimg[r*w+c]-1][2] += (double) qGreen(image.pixel(c,r));
            features[nimg[r*w+c]-1][3] += (double) qBlue(image.pixel(c,r));
        }
    }
    // define the weights for each feature
    double *weights = new double [featurevectorlength];
    /*
    for (int i=0; i<featurevectorlength; i++){
        weights[i] = 1.0;
    }
    */
    weights[0] = 1.0;       // connected componets weighting
    weights[1] = 1.0;       // cluster Red value weighting
    weights[2] = 1.0;       // cluster Green value weighting
    weights[3] = 1.0;       // cluster Blue value weighting

    weights[4] = 0.2;       // boundary weighting
    weights[5] = 0.2;       // boundary  weighting
    weights[6] = 0.2;       // boundary  weighting
    weights[7] = 0.2;       // boundary  weighting
    weights[8] = 0.2;       // boundary weighting

    weights[9] = 1;       // centroid weighting top
    weights[10] = 1;      // centroid weighting left

    weights[11] = 60;      // energy weighting
    weights[12] = 0.2;      // entropy weighting
    weights[13] = 0.00001;      // contrast weighting

    weights[14] = 0.5;      // correlation weighting
    weights[15] = 0.0000000001;      // random selection

    // weighting all features
    for(int k=0; k<num_regions; k++)
    {

        features[k][1] = features[k][1]/(features[k][0]*255.0);
        features[k][2] = features[k][2]/(features[k][0]*255.0);
        features[k][3] = features[k][3]/(features[k][0]*255.0);
        features[k][0] /= (double) w*h;

        features[k][4] = boundary[k][0];
        features[k][5] = boundary[k][1];
        features[k][6] = boundary[k][2];
        features[k][7] = boundary[k][3];
        features[k][8] = boundary[k][4];

        features[k][9] = centroid[k][0];
        features[k][10] = centroid[k][1];
        features[k][11] = energy[k];
        features[k][12] = entropy[k];
        features[k][13] = contrast[k];
        features[k][14] = correlation[k];
        features[k][15] = randomSelect[k];


        // weighting my feature
        for (int i=0; i<featurevectorlength; i++){
            features[k][i] *= weights[i];
        }
        //char fea = features[1];
        //std::cout<<&fea<<std::endl;
        //int fea = (int)*features[1] & INT_MAX;
        //std::cout<<"feature value 1"<<fea<<std::endl;
        //std::cout<<"feature value 2"<<features[1]<<std::endl;
        //printf("%llx\n", &features[1]);
        /*
        std::cout<<"feature value"<<dec<<features[0]<<std::endl;
        std::cout<<"feature value"<<features[1]<<std::endl;
        std::cout<<"feature value"<<features[2]<<std::endl;
        std::cout<<"feature value"<<features[3]<<std::endl;
        std::cout<<"feature value"<<features[4]<<std::endl;
        std::cout<<"feature value"<<features[5]<<std::endl;
        std::cout<<"feature value"<<features[6]<<std::endl;
        std::cout<<"feature value"<<features[7]<<std::endl;
        std::cout<<"feature value"<<features[8]<<std::endl;
        std::cout<<"feature value"<<features[9]<<std::endl;
        std::cout<<"feature value"<<features[10]<<std::endl;
        std::cout<<"feature value"<<features[11]<<std::endl;
        std::cout<<"feature value"<<features[12]<<std::endl;
        std::cout<<"feature value"<<features[13]<<std::endl;
        std::cout<<"feature value"<<features[14]<<std::endl;
        */
        //int fea = HextoDec(hexify(features[15]),10);
        //std::cout<<"feature value"<<HextoDec(features[15],10)<<std::endl;
        //string feature1 = std::to_string(features[1]);
        // output the feature
        featurevector.push_back(features[k]);
    }

    // Return the created feature vector
    ui->progressBox->append(QString::fromStdString("***Done***"));
    QApplication::processEvents();

    // free the memory
    for (int i=0; i<num_regions; i++)
    {
        for (int j=0; j<pixelVal; j++)
        {
            delete[] CoOccurMat[i][j];
            delete[] normCoOccurMat[i][j];
            //delete[] RGB_LCM[i][j];
        }
    }
    for (int i=0; i<num_regions; i++)
    {
        delete[] CoOccurMat[i];
        delete[] normCoOccurMat[i];
        //delete[] RGB_LCM[i];
    }
    delete[] CoOccurMat;
    delete[] normCoOccurMat;
    delete[] energy;
    delete[] entropy;
    delete []contrast;
    for (int i=0; i<num_regions; i++)
    {
        delete[] centroid[i];
        delete[] boundary[i];
    }
    delete[] centroid;
    delete[] boundary;
    delete [] randomSelect;
    delete[] img;
    delete[] nimg;
    delete[] weights;
    //delete regionNum;
    return featurevector;
}


/***** Code to compute the distance between two images *****/

// Function that implements distance measure 1
// cosine distance (not good though)
/*
double distance1(double* vector1, double* vector2, int length)
{

    double nom = 1;
    double vector1sum = 0.0;
    double vector2sum = 0.0;
    for (int i=0 ;i< length; i++)
    {
        vector1sum += pow(vector1[i],2);
        vector2sum += pow(vector2[i],2);
        nom += fabs(vector1[i]*vector2[i]);
    }
    nom = nom / sqrt(vector1sum*vector2sum);
    return nom;
}
*/
// reference distance weighting

double distance1(double* vector1, double* vector2, int length)
{
    double sum=0.0;
    for (int i =0;i<length;i++)
    {
        if (vector1[i] != 0){
            sum+=abs((vector1[i]-vector2[i])/vector1[i]);
        }
        else {
            // this term will influence the accuracy if I don't give a better weight, so I just drop it off
            // and if vector1 is the target image then I elliminate this term won't influence the final result
            //sum+=abs((vector1[i]-vector2[i]));
        }

    }
    sum=sum/length;
    return sum;
}


// Function that implements distance measure 2
// L1 norm
double distance2(double* vector1, double* vector2, int length)
{

    double disSum = 0.0;
    for (int i=0 ;i< length; i++)
    {
        disSum += abs(vector1[i] - vector2[i]);
    }

    return disSum;

}

// Function to calculate the distance between two images
// Input argument isOne takes true for distance measure 1 and takes false for distance measure 2

void MainWindow::CalculateDistances(bool isOne)
{
    for(int n=0; n<num_images; n++) // for each image in the database
    {
        distances[n] = 0.0; // initialize to 0 the distance from query image to a database image

        for (int i = 0; i < queryfeature.size(); i++) // for each region in the query image
        {
            double current_distance = (double) RAND_MAX, new_distance;

            for (int j = 0; j < databasefeatures[n].size(); j++) // for each region in the current database image
            {
                if (isOne)
                    new_distance = distance1(queryfeature[i], databasefeatures[n][j], featurevectorlength);
                else
                    new_distance = distance2(queryfeature[i], databasefeatures[n][j], featurevectorlength);

                current_distance = std::min(current_distance, new_distance); // distance between the best matching regions
            }

            distances[n] = distances[n] + current_distance; // sum of distances between each matching pair of regions
        }

        distances[n] = distances[n] / (double) queryfeature.size(); // normalize by number of matching pairs

        // Display the distance values
        ui->progressBox->append(QString::fromStdString("Distance to image "+std::to_string(n+1)+" = "+std::to_string(distances[n])));
        QApplication::processEvents();
    }
}
