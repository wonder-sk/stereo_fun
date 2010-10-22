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

	inline
	uint32_t msElapsed(uint64_t start)
	{
		return (now() - start) / 1000;
	}
}

#endif // NOW_H
