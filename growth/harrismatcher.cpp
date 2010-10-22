#include "harrismatcher.h"
#include "util.h"
#include "now.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>

using namespace std;
using namespace cv;

HarrisMatcher::HarrisMatcher()
{
	saveImages= false;
	lastTime= -1;

	// Matching parameters
	matchingYTolerance= 2;

	localMaxRadius= 2;
	diffThreshold= 0.001f;

	maxDisparity= 150;
	allowNegativeDisparity= false;

	// Harris parameters
	harrisK= 0.04f;

	localMaxRadius= 4;
	localMaxThreshold= 0.0002f;
}

QList<QVector3D> HarrisMatcher::getMatches()
{
	uint64_t start= Now::now();

	// List of detected Harris points per scanline
	QVector< QList<int> > listLeft;
	QVector< QList<int> > listRight;

	// Each is image is proccessed on a different thread
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			// Get the Harris surface
			harrisLeft= getHarrisSurface(leftImage);
			// Get the points
			listLeft= getLocalMax(harrisLeft, leftImage.size());
			if(saveImages) {
				getHarrisImage(harrisLeft, leftImage.size()).save("surfaceLeft.png");
				drawPoints(leftImage, listLeft).save("pointsLeft.png");
			}
		}
		#pragma omp section
		{
			harrisRight= getHarrisSurface(rightImage);
			listRight= getLocalMax(harrisRight, rightImage.size());
			if(saveImages) {
				getHarrisImage(harrisRight, rightImage.size()).save("surfaceRight.png");
				drawPoints(rightImage, listRight).save("pointsRight.png");
			}
		}
	}

	QList<QVector3D> matches;

	const int height= qMin(leftImage.height(), rightImage.height());

	// For each line...
	for(int y=0; y<height; y++) {

		// The list of points for this line of the left image is on listLeft[y]

		// The list of points for this line of the right image will
		// include points from other lines if matchingYTolerance > 0
		QList<QPoint> right;
		for(int y2= y-matchingYTolerance; y2 <= y+matchingYTolerance; y2++) {
			if(y2 < 0 or y2 >= height) continue;

			foreach(int x, listRight[y2])
				right << QPoint(x, y2);
		}

		// Get the matches for this line
		matches << matchLine(y, listLeft[y], right);
	}

	// Store calculation time
	lastTime= Now::msElapsed(start);

	if(saveImages)
		matchedImage(matches).save("matches.png");

	return matches;
}

QList<QVector3D> HarrisMatcher::matchLine(int y, QList<int> leftXs, QList<QPoint> rightPoints)
{
	QList<QVector3D> matches;

	// For each point on this line of the left image...
	foreach(int leftX, leftXs) {
		QPoint leftPoint(leftX, y);

		// ...find the best match (if any) on rightPoints
		QPoint bestMatch;
		float bestDiff= 10000.0f;

		foreach(QPoint rightPoint, rightPoints) {

			// The similarity between the two points is calculated as the normalized
			// SAD of an horizontal block centered on each point
			float diff= 0;
			for(int dx= -localMaxRadius; dx <= localMaxRadius; dx++) {
				int leftIndex= qBound(0, leftPoint.x()+dx, leftImage.width()-1) +
							   leftPoint.y() * leftImage.width();
				int rightIndex= qBound(0, rightPoint.x()+dx, rightImage.width()-1) +
								rightPoint.y() * rightImage.width();

				diff += fabsf(harrisLeft[leftIndex] - harrisRight[rightIndex]);
			}
			diff /= localMaxRadius*2 + 1;

			if(diff < bestDiff) {
				bestDiff= diff;
				bestMatch= rightPoint;
			}
		}

		bool reject= false;

		// Reject match if the similarity between the blocks is not good enough
		reject |= bestDiff > diffThreshold;
		// Reject match if the disparity is too big
		reject |= abs(leftPoint.x() - bestMatch.x()) > maxDisparity;
		// Reject match if the disparity is negative and this is not allowed
		reject |= !allowNegativeDisparity and leftPoint.x() < bestMatch.x();
		// Reject match similarities with the borders (good for images with black borders)
		//reject |= !leftPoint.x() or !bestMatch.x() or !leftPoint.y();

		if(!reject)
			matches << QVector3D(leftPoint.x(), bestMatch.x(), leftPoint.y());
	}

	return matches;
}

QVector<float> HarrisMatcher::getHarrisSurface(QImage image, int blockSize, int apertureSize)
{
	// Make sure the image has 32 bpp
	image= image.convertToFormat(QImage::Format_RGB32);

	// Convert the QImage to a OpenCV Mat
	// <http://opencv-users.1802565.n2.nabble.com/QImage-to-OpenCV-td5121181.html>
	cv::Mat tempMat= cv::Mat(image.height(), image.width(), CV_8UC4, (uchar*)image.bits(), image.bytesPerLine());
	cv::Mat mat= cv::Mat(tempMat.rows, tempMat.cols, CV_8UC3 );
	int from_to[] = { 0,0,  1,1,  2,2 };
	cv::mixChannels(&tempMat, 1, &mat, 1, from_to, 3);

	// Convert the image to grayscale (required by the harris detector)
	Mat grayMat;
	cvtColor(mat, grayMat, CV_RGB2GRAY);

	// Get the CV_32FC1 harris surface
	Mat harrisMat;
	cornerHarris(grayMat, harrisMat, blockSize, apertureSize, harrisK);

	// Create a QVector with the harris info
	const int elements= harrisMat.size().width * harrisMat.size().height;
	QVector<float> harrisSurface(elements);
	memcpy(harrisSurface.data(), harrisMat.ptr(), elements * sizeof(float));

	return harrisSurface;
}

QVector< QList<int> > HarrisMatcher::getLocalMax(QVector<float> surface, QSize size)
{
	QVector< QList<int> > points(size.height());

	// For each element of the Harris surface...
	for(int y= 0; y<size.height(); y++) {
	for(int x= 0; x<size.width(); x++) {

		float element= surface[x + y * size.width()];

		// If the cornerness is below the threshold ignore this element as a local max
		if(element < localMaxThreshold)
			continue;

		bool isMax= true;

		// Check if this element is a local max around a block
		for(int dy=-localMaxRadius; isMax and dy<=localMaxRadius; dy++) {
		for(int dx=-localMaxRadius; isMax and dx<=localMaxRadius; dx++) {
			if(!dx or !dy) continue;
			const int x2= qBound(0, x+dx, size.width()-1);
			const int y2= qBound(0, y+dy, size.height()-1);
			isMax= element > surface[x2 + y2 * size.width()];
		}
		}

		// If it is a local max store it in the list for this scanline
		if(isMax)
			points[y] << x;
	}
	}

	return points;
}

QImage HarrisMatcher::getHarrisImage(QVector<float> surface, QSize size)
{
	const int elements= surface.count();

	// Calculate max and min values
	float minValue= 0;
	float maxValue= 0;
	for(int i=0; i<elements; i++) {
		const float value= qMax(0.0f, surface[i]);
		if(!i or value < minValue) minValue= value;
		if(!i or value > maxValue) maxValue= value;
	}
	float range= fabsf(maxValue - minValue);

	// Calculate surface histogram
	const int buckets= 2048;
	int histogram[buckets];

	for(int i=0; i<buckets; i++)
		histogram[i]= 0;

	for(int i=0; i<elements; i++) {
		float value= qMax(0.0f, surface[i]);
		// Normalize value from 0 to 1
		value= (value - minValue) / range;

		int bucket= qBound(0, (int)(value * buckets), buckets-1);
		histogram[bucket]++;
	}

	// Get the min and max values for the rendering (ignore 5% small values, 10% of big values)
	float maxGraph;
	float minGraph;

	int count= 0;
	int i= 0;
	while(count < elements * 0.005) {
		count += histogram[i];
		i++;
	}
	minGraph= i / (float)buckets;

	count= 0;
	i= buckets-1;
	while(count < elements * 0.01) {
		count += histogram[i];
		i--;
	}
	maxGraph= i / (float)buckets;

	maxGraph= qMax(maxGraph, minGraph);

	// Render image
	QImage image(size, QImage::Format_RGB32);

	for(int y= 0; y<size.height(); y++) {
		for(int x= 0; x<size.width(); x++) {
			float elem= qMax(surface[x + y * size.width()], 0.0f);
			float value= (elem - minValue) / range;

			image.setPixel(x, y, getColour(value, minGraph, maxGraph));
		}
	}

	return image;
}

QImage HarrisMatcher::drawPoints(QImage image, QVector< QList<int> > points)
{
	image= image.convertToFormat(QImage::Format_RGB32);
	QPainter painter(&image);
	painter.setPen(Qt::red);

	for(int y=0; y<points.count(); y++) {
		foreach(int x, points[y])
			painter.drawEllipse(x, y, 4, 4);
	}
	painter.end();

	return image;
}

QImage HarrisMatcher::matchedImage(QList<QVector3D> matches)
{
	QImage image(leftImage.width(), leftImage.height()+rightImage.height(), QImage::Format_RGB32);
	QPainter painter(&image);

	painter.drawImage(0,0, leftImage);
	painter.drawImage(0, leftImage.height(), rightImage);
	foreach(QVector3D match, matches) {
		QPoint point1(match.x(), match.z());
		QPoint point2(match.y(), match.z()+leftImage.height());
		QColor color= QColor::fromHsv(rand() % 360, 255, 255);

		painter.setPen(color);
		painter.drawEllipse(point1, 3, 3);
		painter.drawEllipse(point2, 3, 3);

		color.setAlpha(100);
		painter.setPen(color);
		painter.drawLine(point1, point2);
	}
	painter.end();

	return image;
}
