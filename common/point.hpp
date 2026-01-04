#pragma once

#include <cmath>
#include <vector>

using namespace std;

class Point {
public:
  vector<double> coords;
  int id;

  Point(const vector<double> &c = {}, int i = -1) : coords(c), id(i) {}

  double distance(const Point &other) const {
    double dist = 0.0;
    for (size_t i = 0; i < coords.size(); ++i) {
      double diff = coords[i] - other.coords[i];
      dist += diff * diff;
    }
    return sqrt(dist);
  }

  double operator[](size_t i) const { return coords[i]; }
  size_t size() const { return coords.size(); }
};
