######################################################################
# Automatically generated by qmake (2.01a) Fri Oct 8 09:53:40 2010
######################################################################

TEMPLATE = app
TARGET =
DEPENDPATH += .
INCLUDEPATH += .

LIBS += -lcv -lhighgui

# Input
SOURCES += censusimage.cpp \
	censuscorrelation.cpp \
	stereo.cpp \
	main.cpp \
	harrismatcher.cpp

HEADERS += \
	util.h \
	censusimage.h \
	censuscorrelation.h \
	stereo.h \
	correlation.h \
	matchingtable.h \
	harrismatcher.h \
    now.h \
    now.h

QMAKE_CXXFLAGS_RELEASE = -march=native \
	-O3 -pipe -fomit-frame-pointer \
	-mssse3 -msse2 \
	-fopenmp

QMAKE_LFLAGS_RELEASE = -fopenmp
