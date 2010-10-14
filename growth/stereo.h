#ifndef STEREO_H
#define STEREO_H

#include <QImage>
#include <QFile>

#include <cmath>
#include <queue>

#include "util.h"
#include "correlation.h"

class MatchingTable;
class MatchingTable2;

typedef struct Seed {
    int x1,x2,y;
    double c;

    Seed(): x1(0), x2(0), y(0), c(-1) {}
    Seed(int _x1, int _x2, int _y, Correlation* calc): x1(_x1), x2(_x2), y(_y)
    {
       c = calc->correlation(x1, x2, y);
    }

    bool operator<(const Seed& q) const { return c < q.c; }
    bool operator>(const Seed& q) const { return c > q.c; }
} Seed;

typedef std::priority_queue<Seed> SeedQueue;


class StereoMatching
{
public:
  StereoMatching(QString n_left, QString n_right, QString n_seeds, Correlation* c = NULL, double cThreshold = 0.6);
  ~StereoMatching();

  void setUsingProposedAlg(bool use) { mUseProposed = use; }

  void run();

  void runInit();
  void runExit();
  bool hasFinished() { return mQueue.empty(); }
  void stepBaseline();
  void stepProposed();

  int numSteps() const { return mNumSteps; }
  QSize size() const { return mImgL.size(); }
  QImage leftImage() const { return mImgL; }
  QImage rightImage() const { return mImgR; }

  QImage disparityImage(bool with_background);
  QImage correlationImage();

protected:

  void loadSeeds();

  Seed best_neighbor_2(const Seed& s, const int N_i[][3], int size);
  Seed best_neighbor(const Seed& s, int i);

  void proposed_matching_line(int y);
  void proposed_do_matching();

  QString mNameLeft, mNameRight, mNameSeeds;
  QImage mImgL, mImgR;
  SeedQueue mQueue;
  QImage mImgD;
  //QImage mImgC;
  int mNumSteps;

  Correlation* mC;

  double mTreshold;

  bool mUseProposed;

  MatchingTable* mTableBaseline;

  MatchingTable2* mTableProposed;
  double mMargin;
  double* C_best_L;
  double* C_best_R;
};

#endif // STEREO_H
