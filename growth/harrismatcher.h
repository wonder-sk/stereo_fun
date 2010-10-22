#ifndef HARRISMATCHER_H
#define HARRISMATCHER_H

#include <QtGui>

class HarrisMatcher
{
public:
	HarrisMatcher();

	void setImages(QImage left, QImage right) { leftImage=left; rightImage= right; }

	QList<QVector3D> getMatches();

	// Last calculation time in ms
	int getLastTime() { return lastTime; }

	// Save debugging PNG's
	bool saveImages;

	/// Harris points parameters
	// K
	float harrisK;
	// Radius of the block to find the local max
	int localMaxRadius;
	// Don't consider local max elements with cornerness below this value
	float localMaxThreshold;

	/// Matching parameters
	// Tolerance in pixels to matching Harris points that don't lie on the same scanline
	int matchingYTolerance;
	// When two Harris points are matched, the SAD of small horizontal block around each one is used
	int matchBlockRadius;
	// Threshold of the average SAD. Values above this value will we rejected as matches
	float diffThreshold;
	// Max allowed disparity, help filtering the final matches
	int maxDisparity;
	bool allowNegativeDisparity;

private:
	QImage leftImage;
	QImage rightImage;

	// 2D harris surfaces for each image
	QVector<float> harrisLeft;
	QVector<float> harrisRight;

	int lastTime;

	/// Aux operations

	// Computes the harris (eigen) surface of an image
	QVector<float> getHarrisSurface(QImage image, int blockSize= 3, int apertureSize= 3);

	// Given a harris surface it returns a list for each scanline containing the best local corners
	// Elements with cornerness below the threshold are ignored alltogheter
	QVector< QList<int> > getLocalMax(QVector<float> surface, QSize size);

	// Return the matches of the (leftXs[i], y) points with the rightPoints
	// The points of rightPoints can have different y (see matchingYTolerance)
	QList<QVector3D> matchLine(int y, QList<int> leftXs, QList<QPoint> rightPoints);

	QImage getHarrisImage(QVector<float> surface, QSize size);
	QImage drawPoints(QImage image, QVector< QList<int> > points);
	QImage matchedImage(QList<QVector3D> matches);

};

#endif // HARRISMATCHER_H
