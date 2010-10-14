#include "censuscorrelation.h"

CensusCorrelation::CensusCorrelation(int transformRadius, int correlationRadius)
{
	this->transformRadius= transformRadius;
	this->correlationRadius= correlationRadius;
}

void CensusCorrelation::transform(QImage leftImage, QImage rightImage)
{
	left.transform(leftImage, transformRadius);
	right.transform(rightImage, transformRadius);
}

uint CensusCorrelation::cost(int x1, int x2, int y)
{	
	uint cost= 0;

	for(int dy= -correlationRadius; dy <= correlationRadius; dy++) {
		for(int dx= -correlationRadius; dx <= correlationRadius; dx++) {

			QBitArray& l= left.getElement( x1+dx, y+dy);
			QBitArray& r= right.getElement(x2+dx, y+dy);

			// Accumulate hamming distance
			cost += (l ^ r).count(true);
		}
	}

	return cost;
}
