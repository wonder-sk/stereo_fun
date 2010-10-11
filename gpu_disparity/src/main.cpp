#include <QtGui>
#include "disparity.h"

int main(int argc, char* argv[])
{
	qDebug() << "Loading images...";
	
	QImage left("left.png");
	QImage right("right.png");

	if(left.isNull() or right.isNull()) {
		qDebug() << "Couldn't load left.png or right.png.";
		return EXIT_FAILURE;
	}

	if(left.size() != right.size()) {
		qDebug() << "Both images should have the same size, quitting.";
		return EXIT_FAILURE;
	}

	qDebug() << "Initializing";
	Disparity disparity(left, right);

	qDebug() << "Calculating disparity map";
	
	disparity.calculate();
	qDebug() << "Done. Time:" << disparity.getLastTime() << "msecs.";
	
	disparity.getDisparity().save("disparity.png");

	qDebug() << "Bye.";

	return EXIT_SUCCESS;
}
