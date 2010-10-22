
#include "stereo.h"

#include "matchingtable.h"
#include "now.h"

#include <QPainter>


const int DISP_MIN = -100;
const int DISP_MAX = +100;


//////////////////////////////////

StereoMatching::StereoMatching(QImage leftImage, QImage rightImage, QList<QVector3D> seeds, Correlation* c)
	: mImgL(leftImage), mImgR(rightImage), mUseProposed(false), mTableBaseline(NULL), mTableProposed(NULL)
{
	Q_ASSERT(!mImgL.isNull() and !mImgR.isNull());
	Q_ASSERT(mImgL.height() == mImgR.height());

	// cut right image for the same size
	// (census correlation requires this)
	mImgR = mImgR.copy(mImgL.rect());

	// initialize correlation calculator
	if(c == NULL) {
		mC= new CorrelationNCC(2);
		mC->init(leftImage, rightImage);
	} else {
		mC= c;
	}

	// Load seeds into the queue
	mQueue = SeedQueue();

	int ignoredSeeds= 0;
	foreach(QVector3D match, seeds) {
		Seed seed(match.x(), match.y(), match.z(), mC);

		if(seed.c >= mC->getThreshold())
			mQueue.push(seed);
		else
			ignoredSeeds++;
	}
	if(!seeds.isEmpty())
		qDebug() << "Using" << seeds.count()-ignoredSeeds << "of" << seeds.count() << "seeds.";

	// prepare disparity image
	mImgD = QImage(mImgL.size(), QImage::Format_ARGB32_Premultiplied);
	mImgD.fill(0);

	lastTime= -1;
}

StereoMatching::~StereoMatching()
{
}

void StereoMatching::run()
{
  uint64_t start= Now::now();
  // load seeds, init matching tables, correlation calculators
  runInit();

  // growth
  mNumSteps = 0;
  while (!mQueue.empty())
  {
	mNumSteps++;
	step();
  }

  // finalization and matching (only proposed)
  runExit();

  lastTime= (Now::now() - start) / 1000;
}

double* _init_C_array(QImage& img)
{
  int size = img.width()*img.height();
  double* C_best = new double[size];
  for (int i=0; i < size; i++)
	C_best[i] = -1;
  return C_best;
}

void StereoMatching::runInit()
{
  mC->resetNumCorrelations();

  if (mUseProposed)
  {
	mTableProposed = new MatchingTable2(mImgL.width(), mImgR.width(), mImgL.height());
	mMargin = 0.1;

	C_best_L = _init_C_array(mImgL);
	C_best_R = _init_C_array(mImgR);
  }
  else
  {
	mTableBaseline = new MatchingTable(mImgL.width(), mImgR.width(), mImgL.height());
  }
}

void StereoMatching::runExit()
{
  if (mUseProposed)
  {
	proposed_do_matching();

	delete mTableProposed;
	mTableProposed = NULL;

	delete C_best_L;
	delete C_best_R;
  }
  else
  {
	delete mTableBaseline;
	mTableBaseline = NULL;
  }
}

void StereoMatching::stepBaseline()
{
	Seed s = mQueue.top();
	//qDebug("[%d] seed: %d %d %d -- d %d -- %f", queue.size(), s.x1, s.x2, s.y, s.x2-s.x1, s.c);
	mQueue.pop();

	int disp = s.x2-s.x1;
	mImgD.setPixel(s.x1, s.y, getColour(disp, DISP_MIN, DISP_MAX)); // disparity
	//mImgC.setPixel(s.x1, s.y, getColour(s.c, tau, 1)); // correlation

	for (int i = 1; i <= 4; i++)
	{
		Seed q_i = best_neighbor(s, i);

		if (q_i.x2-q_i.x1 < DISP_MIN || q_i.x2-q_i.x1 > DISP_MAX) return;

		if (q_i.c >= mC->getThreshold() && !mTableBaseline->isSeedForbidden(q_i))
		{
			mTableBaseline->addSeed(q_i);
			mQueue.push(q_i);
		}
	}
}


void StereoMatching::stepProposed()
{
  Seed s = mQueue.top();
  //qDebug("[%d] seed: %d %d %d -- d %d -- %f", queue.size(), s.x1, s.x2, s.y, s.x2-s.x1, s.c);
  mQueue.pop();

  //mImgD.setPixel( s.x1, s.y, ::getColour(s.x2-s.x1, DISP_MIN, DISP_MAX) );
  int imgLw = mImgL.width(), imgRw = mImgR.width();

  for (int i = 1; i <= 4; i++)
  {
	  Seed q_i = best_neighbor(s, i);

	  //if (q_i.x2-q_i.x1 < DISP_MIN || q_i.x2-q_i.x1 > DISP_MAX) return;
	  bool forbidden = mTableProposed->isSeedForbidden(q_i);

	  if (q_i.c >= mC->getThreshold() && !forbidden &&
		  q_i.c + mMargin >= qMin(C_best_L[q_i.x1+q_i.y*imgLw], C_best_R[q_i.x2+q_i.y*imgRw]) )
	  {
		  mTableProposed->addSeed(q_i);
		  mQueue.push(q_i);

		  C_best_L[q_i.x1+q_i.y*imgLw] = qMax(q_i.c, C_best_L[q_i.x1+q_i.y*imgLw]);
		  C_best_R[q_i.x2+q_i.y*imgRw] = qMax(q_i.c, C_best_R[q_i.x2+q_i.y*imgRw]);
	  }
  }

}

QImage StereoMatching::disparityImage(bool with_background)
{
  if (with_background)
  {
	QImage img_final(mImgL.size(), QImage::Format_ARGB32_Premultiplied);
	QPainter p;
	p.begin(&img_final);
	p.drawImage(0,0, mImgL);
	p.drawImage(0,0, mImgD);
	p.end();
	return img_final;
  }
  else
	return mImgD;
}

void StereoMatching::loadSeedsFromFile(QString path)
{
  // Start a new queue
  mQueue = SeedQueue();

  QFile fseeds(path);
  bool res = fseeds.open(QIODevice::ReadOnly);
  Q_ASSERT(res);

  // add seeds
  for (;;)
  {
	  QByteArray line = fseeds.readLine();
	  if (line.isNull()) break;
	  QList<QByteArray> fields = line.split(',');
	  Q_ASSERT(fields.count() == 3);
	  int x1 = fields[0].toInt(), x2 = fields[1].toInt(), y = fields[2].trimmed().toInt();

	  Seed s_tmp(x1, x2, y, mC);
	  if (s_tmp.c >= mC->getThreshold())
		mQueue.push( s_tmp );
	  else
		qDebug("seed ignored: [%d,%d,%d] (c=%f)", x1,x2,y, s_tmp.c);
  }
}



const int N_1[][3] = { { -1, -1,  0 }, { -2, -1,  0 }, { -1, -2,  0 } };
const int N_2[][3] = { { +1, +1,  0 }, { +2, +1,  0 }, { +1, +2,  0 } };
const int N_3[][3] = { {  0,  0, -1 }, { +1,  0, -1 }, { -1,  0, -1 }, { 0, +1, -1 }, { 0, -1, -1 } };
const int N_4[][3] = { {  0,  0, +1 }, { +1,  0, +1 }, { -1,  0, +1 }, { 0, +1, +1 }, { 0, -1, +1 } };

Seed StereoMatching::best_neighbor_2(const Seed& s, const int N_i[][3], int size)
{
	// find best one q_i =(u,u',v) from the group N_i
	Seed best_candidate;
	for (int j = 0; j < size; j++)
	{
	  int x1 = s.x1 + N_i[j][0], x2 = s.x2 + N_i[j][1], y = s.y + N_i[j][2];

	  if (x1 < 0 || x2 < 0 || y < 0 ||
		  x1 >= mImgL.width() || x2 >= mImgR.width() || y >= mImgL.height()) continue;

	  Seed candidate(x1, x2, y, mC);
		if (candidate > best_candidate)
			best_candidate = candidate;
	}

	return best_candidate;
}

Seed StereoMatching::best_neighbor(const Seed& s, int i)
{
	switch (i)
	{
		case 1: return best_neighbor_2(s, N_1, 3); //3);
		case 2: return best_neighbor_2(s, N_2, 3); //3);
		case 3: return best_neighbor_2(s, N_3, 5); //5);
		case 4: return best_neighbor_2(s, N_4, 5); //5);
	}
	Q_ASSERT(false);
	return Seed();
}


void StereoMatching::proposed_matching_line(int y)
{
  WD_set* WD1 = mTableProposed->matchingAtLine(y);
  WD_set* WD2 = new WD_set[mTableProposed->w2()];

  // 0; construct matching table from right image for current line
  for (int x1 = 0; x1 < mTableProposed->w1(); x1++)
  {
	WD_set& WD1_x1 = WD1[x1];
	for (WD_set::iterator it = WD1_x1.begin(); it != WD1_x1.end(); it++)
	{
	  int x2 = x1 + it->disparity;
	  WD2[x2].insert( *it );
	}
  }

  // for each line of T:
  // - find dominant element
  for (;;)
  {
	bool has_dominant = false;
	int x1,x2;

	// try various x_1
	for (x1 = 0; x1 < mTableProposed->w1(); x1++)
	{
	  WD_set& WD1_x1 = WD1[x1];
	  if (WD1_x1.size() == 0)
		continue;

	  double best_c = -1;
	  int best_d = 0;
	  bool best_is_dominant = true;

	  // find the one with best correlation
	  for (WD_set::iterator it = WD1_x1.begin(); it != WD1_x1.end(); it++)
	  {
		if (it->c > best_c)
		{
		  best_c = it->c;
		  best_d = it->disparity;
		}
	  }

	  x2 = x1+best_d;
	  WD_set& WD2_x2 = WD2[x2];

	  // check whether it's dominant
	  for (WD_set::iterator it = WD1_x1.begin(); it != WD1_x1.end(); it++) // (x1,:,y)
	  {
		if (it->disparity == best_d) continue; // ignore the same item
		if (best_c <= it->c + mMargin) { best_is_dominant = false; break; }
	  }
	  for (WD_set::iterator it = WD2_x2.begin(); it != WD2_x2.end(); it++) // (:,x2,y)
	  {
		if (it->disparity == best_d) continue; // ignore the same item
		if (best_c <= it->c + mMargin) { best_is_dominant = false; break; }
	  }

	  if (best_is_dominant)
	  {
		has_dominant = true;
		break;
	  }
	} // (for each x1)

	// no more dominant elements on this line
	if (!has_dominant) break;

	// remove its inhibition zone
	WD1[x1].clear(); // (x1,:,y)
	WD2[x2].clear(); // (:,x2,y)

	// mark the point as finished
	mImgD.setPixel(x1,y, getColour(x2-x1, DISP_MIN, DISP_MAX));
  }

  delete [] WD2;
}

void StereoMatching::proposed_do_matching()
{
  QTime t;
  t.start();
  for (int y = 0; y < mTableProposed->h(); y++)
	proposed_matching_line(y);
  qDebug("Matching time: %d ms", t.elapsed());
}
