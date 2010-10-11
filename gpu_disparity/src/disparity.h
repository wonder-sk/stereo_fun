#ifndef DISPARITY_H
#define DISPARITY_H

#include <QtGui>

class Disparity
{
public:
	Disparity(QImage leftImage, QImage rightImage);
	~Disparity();

	void calculate();
	QImage getDisparity();

	float getLastTime() { return lastTime; }
	
private:
	QSize size;
	QImage leftImage;
	QImage rightImage;
	void* left;
	void* right;

	void* disparity;

	// Last execution time in ms
	float lastTime;
};

#endif