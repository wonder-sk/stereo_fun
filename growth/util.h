#ifndef UTIL_H
#define UTIL_H

inline QRgb getColour(double v,double vmin,double vmax)
{
   // http://local.wasp.uwa.edu.au/~pbourke/texture_colour/colourramp/
   double r = 1.0, g = 1.0, b = 1.0; // white
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      r = 0;
      g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      r = 0;
      b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      r = 4 * (v - vmin - 0.5 * dv) / dv;
      b = 0;
   } else {
      g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      b = 0;
   }

    return qRgb(r*255, g*255, b*255);
    //return qRgb(best_dis*4, best_dis*4, best_dis*4);
}

#endif // UTIL_H
