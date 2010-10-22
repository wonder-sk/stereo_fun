#include <QtGui>

#include "stereo.h"
#include "harrismatcher.h"
#include "util.h"
#include "now.h"


// visualization
const int STEPS_AT_ONCE = 50000;
double IMG_SCALE = 0.4;


class Gui : public QWidget
{
public:

	Gui(StereoMatching* stereoMatcher) : QWidget()
	{
		stereo = stereoMatcher;
		resize(stereo->size() * IMG_SCALE);
	}

	~Gui() {}

	int do_stuff()
	{
		uint64_t start= Now::now();
		stereo->runInit();

		while (!stereo->hasFinished())
		{
			stereo->step();

			if (stereo->numSteps() % STEPS_AT_ONCE == 0) {
				update();
				qApp->processEvents();
			}
		}
		update();
		stereo->runExit();

		return Now::msElapsed(start);
	}

protected:
	void paintEvent(QPaintEvent* )
	{
		QPainter p(this);
		p.scale(IMG_SCALE, IMG_SCALE);
		p.drawImage(0,0, stereo->leftImage());

		p.drawImage(0,0, stereo->disparityImage(false));

		p.setPen(QPen(Qt::red, 2));

		/*
		// show seeds
		int d = 6;
		foreach (const Seed& s, init_seeds)
		{
		  p.drawLine(s.x1-d, s.y, s.x1+d, s.y);
		  p.drawLine(s.x1, s.y-d, s.x1, s.y+d);
		}*/

		QFont f = p.font();
		f.setPointSize(25);
		p.setFont(f);
		p.drawText(20,40, QString("STEPS %1 ... #CORR: %2").arg(stereo->numSteps())
				   .arg(stereo->correlation()->numCorrelations()));
	}

	StereoMatching* stereo;
};

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	// Parse arguments
	QString leftPath= "Rotunda_0.png";
	QString rightPath= "Rotunda_1.png";
	QString outputPath= "disparity.png";
	bool gui= true;
	bool proposedAlg= false;
	bool saveImages= false;
	bool censusCorrelation= false;

	// ./grow left.png right.png output.png [options]
	if(argc >= 4) {
		leftPath= QString(argv[1]);
		rightPath= QString(argv[2]);
		outputPath= QString(argv[3]);
	}
	for(int i=4; i<argc; i++) {
		QString param(argv[i]);
		if(param == "-gui") gui= true; else
		if(param == "-nogui") gui= false; else
		if(param == "-proposed") proposedAlg= true; else
		if(param == "-baseline") proposedAlg= false; else
		if(param == "-saveimages") saveImages= true; else
		if(param == "-census") censusCorrelation= true; else
		if(param == "-ncc") censusCorrelation= false;
	}

	qDebug() << "Loading images...";
	QImage leftImage(leftPath);
	QImage rightImage(rightPath);

	if(leftImage.isNull() or rightImage.isNull()) {
		qDebug() << "Couldn't open" << leftPath << "or" << rightPath;
		return EXIT_FAILURE;
	}

	qDebug() << "Matching images...";
	HarrisMatcher matcher;
	matcher.setImages(leftImage, rightImage);
	matcher.saveImages= saveImages;
	matcher.maxDisparity= 100;
	QList<QVector3D> seeds= matcher.getMatches();
	qDebug() << " - Found" << seeds.count() << "matches in" << matcher.getLastTime() << "ms.";

	qDebug() << "Initializing the" << QString(censusCorrelation ? "Census" : "NCC") << "correlator...";
	const int cRadius= 2;
	Correlation* correlator;
	if(censusCorrelation)
		correlator= new CorrelationCensus(cRadius, cRadius);
	else
		correlator= new CorrelationNCC(cRadius);
	correlator->setThreshold(censusCorrelation ? 0.005 : 0.6);
	int initTime= correlator->init(leftImage, rightImage);
	qDebug() << " - Done in" << initTime << "ms.";

	qDebug() << "Stereo matching using the" << QString(proposedAlg ? "Proposed" : "Baseline") << "algorithm...";
	StereoMatching stereoMatcher(leftImage, rightImage, seeds, correlator);
	//stereoMatcher.loadSeedsFromFile("Rotunda_seeds.txt");
	stereoMatcher.setUsingProposedAlg(proposedAlg);

	int elapsedTime;
	if(gui) {

		qDebug() << "Starting the GUI...";
		Gui gui(&stereoMatcher);
		gui.show();
		elapsedTime= gui.do_stuff();
		gui.close();

	} else {

		stereoMatcher.run();
		elapsedTime= stereoMatcher.getLastTime();
	}

	qDebug() << "Finishing...";
	stereoMatcher.disparityImage(true).save(outputPath);

	qDebug() << "--";
	qDebug() << "Disparity map calculated in" << elapsedTime << "ms.";
	qDebug() << correlator->numCorrelations()/1000 << "K correlations.";


	delete correlator;

	QTimer::singleShot(100, &app, SLOT(quit()));
	return app.exec();
}
