
#include <iostream>

#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QTime>

#include "util.h"

#include "stereo.h"

// visualization
const int STEPS_AT_ONCE = 50000;
double IMG_SCALE = 0.4;


class Gui : public QWidget
{
public:
  Gui(QString nameLeft, QString nameRight, QString nameSeeds)
  {
    c = new CorrelationNCC(2);
    double cThr = 0.6;
    //c = new CorrelationCensus(2,2);
    //double cThr = 0.0002;

    stereo = new StereoMatching(nameLeft, nameRight, nameSeeds, c, cThr);

    resize(stereo->size()*IMG_SCALE);

    stereo->runInit();
  }

  ~Gui() { delete stereo; }

  void do_stuff()
  {
    stereo->runInit();

    QTime t;
    t.start();

    steps = 0;
    while (!stereo->hasFinished())
    {
        steps++;
        stereo->stepBaseline();

        if (steps % STEPS_AT_ONCE == 0)
        {
          update();
          qApp->processEvents();
        }
    }
    qDebug("total time %d ms", t.elapsed());

    update();

    stereo->runExit();

    stereo->disparityImage(true).save("disparity.png");
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
    p.drawText(20,40, QString("STEPS %1 ... #CORR: %2").arg(steps).arg(c->numCorrelations()));

  }

  Correlation* c;
  StereoMatching* stereo;
  int steps;
};


int main(int argc, char* argv[])
{
#if 0
  StereoMatching sm("Rotunda_0.png", "Rotunda_1.png", "Rotunda_seeds.txt");
  //StereoMatching sm("Larch_0.png", "Larch_1.png", "Larch_seeds.txt");
  //StereoMatching sm("car_left.png", "car_right.png", "car_seeds.txt");

  sm.setUsingProposedAlg(true);
  sm.run();

  sm.disparityImage(true).save("disparity.png", "png");
  // sm.correlationImage();
  return 0;

#else
  QApplication a(argc, argv);

  Gui g("Rotunda_0.png", "Rotunda_1.png", "Rotunda_seeds.txt");
  //Gui g("Larch_0.png", "Larch_1.png", "Larch_seeds.txt");
  //Gui g("car_left.png", "car_right.png", "car_seeds.txt");
  g.show();

  g.do_stuff();
    
  return a.exec();
#endif
}
