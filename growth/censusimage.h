#ifndef CENSUSIMAGE_H
#define CENSUSIMAGE_H

#include <QtGui>

class CensusImage
{
public:
	CensusImage();

	void transform(QImage image, int radius= 7);

	QBitArray& getElement(int x, int y);

	QSize getSize() { return image.size(); }
	QImage getImage() { return image; }
	
private:
	QImage image;
	int radius;

	QVector<QBitArray> data;

	QBitArray getCensus(int x, int y);
};

#endif