#include "censusimage.h"

CensusImage::CensusImage()
{
}

void CensusImage::transform(QImage image, int radius)
{
	this->image= image;
	this->radius= radius;
	
	data.resize(image.width() * image.height());
	
	for(int y=0; y<image.height(); y++)
		for(int x=0; x<image.width(); x++)
			data[x + y * image.width()]= getCensus(x, y);
}

QBitArray CensusImage::getCensus(int x, int y)
{
	QBitArray ret((2*radius + 1) * (2*radius + 1));

	uchar refPixel= qGray(image.pixel(x, y));
	
	int bitCount= 0;
	
	for(int dy= -radius; dy <= radius; dy++) {
		for(int dx= -radius; dx <= radius; dx++) {

			QPoint pos(x+dx, y+dy);
			// Clamp coords
			pos.rx() = qBound(0, pos.x(), image.width() -1);
			pos.ry() = qBound(0, pos.y(), image.height()-1);

			const uchar pixel= qGray(image.pixel(pos));
			
			ret.setBit(bitCount, pixel < refPixel);
			
			bitCount++;
		}
	}

	return ret;
}

QBitArray& CensusImage::getElement(int x, int y)
{
	// Clamp coords
	x= qBound(0, x, image.width() -1);
	y= qBound(0, y, image.height()-1);

	return data[x + y * image.width()];
}