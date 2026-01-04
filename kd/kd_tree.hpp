#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

#include "point.hpp"

using namespace std;
using namespace std::chrono;

struct KDNode {
  Point point;
  int axis;
  unique_ptr<KDNode> left;
  unique_ptr<KDNode> right;

  KDNode(const Point &p, int a)
      : point(p), axis(a), left(nullptr), right(nullptr) {}
};

class KDTree {
private:
  unique_ptr<KDNode> root;
  int dimensions;
  int treeSize;

  struct AxisComparator {
    int axis;
    AxisComparator(int a) : axis(a) {}
    bool operator()(const Point &a, const Point &b) const {
      return a.coords[axis] < b.coords[axis];
    }
  };

  unique_ptr<KDNode> buildTree(vector<Point> &points, int depth, int start,
                               int end) {
    if (start >= end)
      return nullptr;

    int axis = depth % dimensions;
    int mid = start + (end - start) / 2;

    nth_element(points.begin() + start, points.begin() + mid,
                points.begin() + end, AxisComparator(axis));

    auto node = make_unique<KDNode>(points[mid], axis);
    node->left = buildTree(points, depth + 1, start, mid);
    node->right = buildTree(points, depth + 1, mid + 1, end);

    return node;
  }

  void nearestNeighbor(const KDNode *node, const Point &target,
                       const KDNode *&best, double &bestDist) const {
    if (!node)
      return;

    double dist = node->point.distance(target);
    if (dist < bestDist) {
      bestDist = dist;
      best = node;
    }

    int axis = node->axis;
    double diff = target[axis] - node->point[axis];

    const KDNode *first = diff < 0 ? node->left.get() : node->right.get();
    const KDNode *second = diff < 0 ? node->right.get() : node->left.get();

    nearestNeighbor(first, target, best, bestDist);

    if (fabs(diff) < bestDist) {
      nearestNeighbor(second, target, best, bestDist);
    }
  }

  void
  kNearestNeighbors(const KDNode *node, const Point &target, int k,
                    priority_queue<pair<double, const Point *>> &heap) const {
    if (!node)
      return;

    double dist = node->point.distance(target);

    if (heap.size() < k) {
      heap.push({dist, &node->point});
    } else if (dist < heap.top().first) {
      heap.pop();
      heap.push({dist, &node->point});
    }

    int axis = node->axis;
    double diff = target[axis] - node->point[axis];

    const KDNode *first = diff < 0 ? node->left.get() : node->right.get();
    const KDNode *second = diff < 0 ? node->right.get() : node->left.get();

    kNearestNeighbors(first, target, k, heap);

    if (heap.size() < k || fabs(diff) < heap.top().first) {
      kNearestNeighbors(second, target, k, heap);
    }
  }

  void insert(unique_ptr<KDNode> &node, const Point &point, int depth) {
    if (!node) {
      node = make_unique<KDNode>(point, depth % dimensions);
      treeSize++;
      return;
    }

    int axis = node->axis;
    if (point[axis] < node->point[axis]) {
      insert(node->left, point, depth + 1);
    } else {
      insert(node->right, point, depth + 1);
    }
  }

  int calculateDepth(const KDNode *node) const {
    if (!node)
      return 0;
    return 1 + max(calculateDepth(node->left.get()),
                   calculateDepth(node->right.get()));
  }

public:
  double buildTimeMs;
  double totalInsertionTimeMs;
  double totalSearchTimeMs;
  size_t estimatedMemoryBytes;

  KDTree()
      : root(nullptr), dimensions(0), treeSize(0), buildTimeMs(0),
        totalInsertionTimeMs(0), totalSearchTimeMs(0), estimatedMemoryBytes(0) {
  }

  void build(vector<Point> &points) {
    if (points.empty())
      return;

    auto start = high_resolution_clock::now();

    dimensions = points[0].size();
    treeSize = points.size();

    root = buildTree(points, 0, 0, points.size());

    auto end = high_resolution_clock::now();
    buildTimeMs = duration_cast<microseconds>(end - start).count() / 1000.0;

    estimatedMemoryBytes =
        treeSize * (sizeof(KDNode) + dimensions * sizeof(double));
  }

  void insertPoint(const Point &point) {
    auto start = high_resolution_clock::now();

    if (dimensions == 0) {
      dimensions = point.size();
    }

    insert(root, point, 0);

    auto end = high_resolution_clock::now();
    double insertionTime =
        duration_cast<microseconds>(end - start).count() / 1000.0;
    totalInsertionTimeMs += insertionTime;
  }

  Point nearestNeighbor(const Point &target, double &searchTime) {
    auto start = high_resolution_clock::now();

    const KDNode *best = nullptr;
    double bestDist = numeric_limits<double>::max();

    nearestNeighbor(root.get(), target, best, bestDist);

    auto end = high_resolution_clock::now();
    searchTime = duration_cast<microseconds>(end - start).count() / 1000.0;
    totalSearchTimeMs += searchTime;

    return best ? best->point : Point();
  }

  vector<Point> kNearestNeighbors(const Point &target, int k,
                                  double &searchTime) {
    auto start = high_resolution_clock::now();

    priority_queue<pair<double, const Point *>> heap;

    kNearestNeighbors(root.get(), target, k, heap);

    vector<Point> result;
    while (!heap.empty()) {
      result.push_back(*heap.top().second);
      heap.pop();
    }
    reverse(result.begin(), result.end());

    auto end = high_resolution_clock::now();
    searchTime = duration_cast<microseconds>(end - start).count() / 1000.0;
    totalSearchTimeMs += searchTime;

    return result;
  }

  int getDepth() const { return calculateDepth(root.get()); }
  double getBalanceFactor() const {
    int depth = getDepth();
    if (depth == 0 || treeSize == 0)
      return 0.0;
    double idealDepth = log2(treeSize + 1);
    return depth / idealDepth;
  }

  double getBuildTime() const { return buildTimeMs; }
  double getTotalInsertionTime() const { return totalInsertionTimeMs; }
  double getAverageInsertionTime() const {
    return treeSize > 0 ? totalInsertionTimeMs / treeSize : 0.0;
  }

  int size() const { return treeSize; }
  int getDimensions() const { return dimensions; }
};

inline vector<Point> readCSV(const string &filename, int maxRows = -1,
                             int numFeatures = -1) {
  ifstream file(filename);
  vector<Point> points;
  string line;
  int rowCount = 0;

  if (!file.is_open()) {
    cerr << "Error: No se pudo abrir el archivo " << filename << endl;
    return points;
  }

  while (getline(file, line) && (maxRows == -1 || rowCount < maxRows)) {
    stringstream ss(line);
    string value;

    if (!getline(ss, value, ','))
      continue;

    int id;
    try {
      id = stoi(value);
    } catch (...) {
      cerr << "Error al leer ID: " << value << endl;
      continue;
    }

    vector<double> coords;
    int featureCount = 0;

    while (getline(ss, value, ',')) {
      if (numFeatures != -1 && featureCount >= numFeatures)
        break;

      try {
        coords.push_back(stod(value));
        featureCount++;
      } catch (...) {
        cerr << "Error al convertir valor: " << value << endl;
        coords.push_back(0.0);
        featureCount++;
      }
    }

    if (!coords.empty()) {
      points.emplace_back(coords, id);
      rowCount++;
    }
  }

  file.close();
  return points;
}
