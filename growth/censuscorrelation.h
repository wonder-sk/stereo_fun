#ifndef CENSUSCORRELATION_H
#define CENSUSCORRELATION_H

#include <QtGui>
#include "censusimage.h"

class CensusCorrelation
{
public:
	CensusCorrelation(int transformRadius= 7, int correlationRadius= 4);

	void transform(QImage leftImage, QImage rightImage);

	uint cost(int x1, int x2, int y);

	void setTransformRadius(int value) { transformRadius= value; }
	int getTransformRadius() { return transformRadius; }
	
	void setCorrelationRadius(int value) { correlationRadius= value; }
	int getCorrelationRadius() { return correlationRadius; }
	
private:
	int transformRadius;
	int correlationRadius;
	
	CensusImage left;
	CensusImage right;
};

#endif