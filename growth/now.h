#ifndef NOW_H
#define NOW_H

#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

namespace Now
{
	inline
	uint64_t now()
	{
	  /* time value container */
	  struct timeval tv;
	  /* get unix time */
	  gettimeofday( &tv, NULL );
	  /* return double value of unix time */
	  return (tv.tv_sec * 1000000) + tv.tv_usec;
	}
}

#endif // NOW_H
