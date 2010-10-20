#ifndef MATCHINGTABLE_H
#define MATCHINGTABLE_H

// matching table for baseline algorithm

class MatchingTable
{
public:
	MatchingTable(int w1, int w2, int h) : width1(w1), width2(w2), height(h)
	{
		// Set the tables' size. They are initialized to 0
		table1.resize(width1 * height);
		table2.resize(width2 * height);
	}

	bool isSeedForbidden(const Seed& s)
	{
		// return whether it's in the inhibition zone
		const bool forbidden1= table1.at(s.x1 + s.y * width1);
		const bool forbidden2= table2.at(s.x2 + s.y * width2);

		return forbidden1 or forbidden2;
	}

	void addSeed(const Seed& s)
	{
		table1.setBit(s.x1 + s.y * width1);
		table2.setBit(s.x2 + s.y * width1);
	}

protected:
	int width1, width2, height;

	QBitArray table1; // match indexes to the right image
	QBitArray table2;
};

// matching table for proposed algorithm

#include <set>

typedef struct WD
{
	WD(int _d, double _c): disparity(_d), c(_c) {}
	int disparity;
	double c;

	bool operator<(const WD& other) const { return disparity < other.disparity; }
} WD;

typedef std::set<WD> WD_set;

class MatchingTable2
{
public:
  MatchingTable2(int w1, int w2, int h) : width1(w1), width2(w2), height(h)
  {
	T = new WD_set[width1 * height];
	matches = 0;
  }

  bool isSeedForbidden(const Seed& s)
  {
	// must have different disparity to allow
	return T[s.x1+s.y*width1].count( WD(s.x2-s.x1, s.c) ) > 0;
  }

  void addSeed(const Seed& s)
  {
	T[s.x1+s.y*width1].insert( WD(s.x2-s.x1, s.c) );
	matches++;
  }

  int w1() { return width1; }
  int w2() { return width2; }
  int h() { return height; }
  WD_set* matchingAtLine(int y) { return T+(y*width1); }

  void save(QString filename)
  {
	qDebug("matches: %d", matches);
	QFile f(filename);
	bool openSucess= f.open(QIODevice::WriteOnly);
	Q_ASSERT(openSucess);
	f.write((const char*)&width1, sizeof(int));
	f.write((const char*)&width2, sizeof(int));
	f.write((const char*)&height, sizeof(int));

	for (int i = 0; i < width1*height; i++)
	{
	  WD_set& Ti = T[i];
	  int count = Ti.size();
	  f.write((const char*) &count, sizeof(int));
	  for (WD_set::iterator it = Ti.begin(); it != Ti.end(); it++ )
	  {
		const WD& tmp_wd = *it;
		f.write((const char*) &tmp_wd, sizeof(WD));
	  }
	}
	f.close();
  }

  static MatchingTable2* load(QString filename)
  {
	QFile f(filename);
	bool openSuccess= f.open(QIODevice::ReadOnly);
	Q_ASSERT(openSuccess);
	// format: width1, width2, height
	// for each pixel (width1*height)-times: number of entries, entry1, entry2, ...  (entry = disparity, correlation)
	int w1, w2, h;
	f.read((char*) &w1, sizeof(int));
	f.read((char*) &w2, sizeof(int));
	f.read((char*) &h, sizeof(int));
	MatchingTable2* MT = new MatchingTable2(w1, w2, h);

	WD tmp_wd(0,-1);
	for (int i = 0; i < w1*h; i++)
	{
	  int count;
	  f.read((char*) &count, sizeof(int));
	  for (int j = 0; j < count; j++)
	  {
		f.read((char*) &tmp_wd, sizeof(WD));
		MT->T[i].insert( tmp_wd );
	  }
	  MT->matches += count;
	}
	qDebug("loaded matches: %d", MT->matches);
	Q_ASSERT(f.atEnd());
	return MT;
  }

protected:
  int width1, width2, height;
  WD_set* T;
  int matches;
};


#endif // MATCHINGTABLE_H
