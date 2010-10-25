#ifndef DISPARITY_H
#define DISPARITY_H

#include <QtGui>

class Disparity
{
public:
	Disparity(QImage leftImage, QImage rightImage);
	~Disparity();

	void transform();
	void calculate();
	QImage getDisparity();

	float getLastTime() { return lastTime; }
	
private:
	QSize size;
	int colorBytes;
	int bytes;
	
	QImage leftImage;
	void* leftColor;
	void* left;
	void* leftCensus5;
	void* leftCensus7;

	QImage rightImage;	
	void* rightColor;
	void* right;
	void* rightCensus5;
	void* rightCensus7;

	void* disparityColor;
	void* disparity;

	// Last execution time in ms
	float lastTime;
};

#endif