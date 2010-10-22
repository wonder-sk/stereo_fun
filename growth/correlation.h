#ifndef CORRELATION_H
#define CORRELATION_H

#include <QImage>
#include "now.h"

// various correlation methods:
// http://siddhantahuja.wordpress.com/tag/normalized-cross-correlation/

class Correlation
{
public:
  Correlation(): mNumCorrelations(0) {}
  virtual ~Correlation() {}

  long numCorrelations() { return mNumCorrelations; }
  void resetNumCorrelations() { mNumCorrelations = 0; }

  double getThreshold() { return threshold; }
  void setThreshold(double value) { threshold= value; }

  virtual int init(QImage imgL, QImage imgR) = 0;
  virtual double correlation(int x_l, int x_r, int y) = 0;

protected:
  long mNumCorrelations;
  double threshold;
};

////////////////////////

class CorrelationNCC : public Correlation
{
public:
  CorrelationNCC(int w) : win_size(w) {}

  virtual int init(QImage imgL, QImage imgR)
  {
	uint64_t start= Now::now();

	// convert images to grayscale, calc mean and variance
	IMG_l = qimage2img(imgL, win_size);
	IMG_r = qimage2img(imgR, win_size);

	return (Now::now()-start)/1000;
  }

  virtual double correlation(int x_l, int x_r, int y)
  {
	double sum_orig = 0;

	mNumCorrelations++;

	double var_l = IMG_l.data_var[x_l + y*IMG_l.width];
	double var_r = IMG_r.data_var[x_r + y*IMG_r.width];

	if (var_l == 0 || var_r == 0) return 0;

	double mean_l = IMG_l.data_mean[ x_l + y*IMG_l.width ];
	double mean_r = IMG_r.data_mean[ x_r + y*IMG_r.width ];

	for (int j = -win_size; j <= win_size; j++)
	{
		double* lineL = IMG_l.data + (y+j)*IMG_l.width + x_l - win_size;
		double* lineR = IMG_r.data + (y+j)*IMG_r.width + x_r - win_size;

		for (int i = -win_size; i <= win_size; ++i, ++lineL, ++lineR)
		{
			sum_orig += (*lineL - mean_l)*(*lineR - mean_r);
			//sum += (*lineL) * (*lineR);
		}
	}

	int N = ((win_size*2+1)*(win_size*2+1));

	//double cov2 = ( sum - mean_l*mean_r );
	double cov = sum_orig / N;
	double norm = sqrt( var_l * var_r );
	//double norm_m = ( var_l + var_r ) / 2;
	double ncc = cov / norm;

	//qDebug("[%d,%d] cov %f vs %f", x_l, y, cov, cov2);
	//double ncc2 = sum*sum / ( var_l * var_r );
	//qDebug("ncc %f  ( %f / %f )", ncc, cov, norm); //, ncc_orig);
	return ncc;
  }

  typedef struct
  {
	int width, height;
	double* data;
	double* data_mean;
	double* data_var;
  } IMG;

protected:

  IMG qimage2img(QImage img, int w)
  {
	int N = ((w*2+1)*(w*2+1));

	IMG i2;
	i2.width = img.width();
	i2.height = img.height();
	i2.data = new double[i2.width*i2.height];
	for (int x=0; x < i2.width; x++)
	  for (int y=0; y < i2.height; y++)
		i2.data[x+y*i2.width] = qGray(img.pixel(x,y));

	i2.data_mean = new double[i2.width*i2.height];

	double* data = i2.data;
	for (int x=0; x < i2.width; x++)
	  for (int y=0; y < i2.height; y++)
	  {
		double sum=0;
		for (int j = -w; j <= w; j++)
		  for (int i = -w; i <= w; i++)
			sum += data[x+i + (y+j)*i2.width];
		i2.data_mean[x+y*i2.width] = sum / N;
	  }

	i2.data_var = new double[i2.width*i2.height];

	for (int x=0; x < i2.width; x++)
	  for (int y=0; y < i2.height; y++)
	  {
		double data_mean = i2.data_mean[x + y*i2.width];
		double sum_sqr=0;
		for (int j = -w; j <= w; j++)
		  for (int i = -w; i <= w; i++)
		  {
			int dd = data[x+i + (y+j)*i2.width] - data_mean;
			sum_sqr += dd*dd;
		  }
		//i2.data_var[x+y*i2.width] = sum_sqr;
		i2.data_var[x+y*i2.width] = sum_sqr / N; // - data_mean*data_mean;
		//qDebug("mean %f var %f", data_mean, i2.data_var[x+y*i2.width]);
	  }

	return i2;
  }


  int win_size;
  IMG IMG_l, IMG_r;
};

////////////////////////

#include "censuscorrelation.h"


class CorrelationCensus : public Correlation
{
public:
  CorrelationCensus(int transformRadius, int corrRadius) : cc(transformRadius, corrRadius) {}

  virtual int init(QImage imgL, QImage imgR)
  {
	uint64_t start= Now::now();
	cc.transform(imgL, imgR);
	return (Now::now()-start) / 1000;
  }

  virtual double correlation(int x_l, int x_r, int y)
  {
	mNumCorrelations++;
	return 1.0 / ( cc.cost(x_l,x_r,y) + 1 );
  }
protected:
  CensusCorrelation cc;
};

#endif // CORRELATION_H
