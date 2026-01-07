#pragma once

#include "vp_defs.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

#include "point.hpp"

class VP_tree {
  std::random_device rd;
  std::mt19937 eng{rd()};

  inline double euclidsq_dist(size_t i, size_t j) const;

  typedef std::priority_queue<VPNeig, std::vector<VPNeig>,
                              decltype([](const VPNeig &lhs,
                                          const VPNeig &rhs) {
                                return lhs.d < rhs.d;
                              })>
      NodeMaxHeap;

  size_t nobjs{};
  std::unique_ptr<VPNode> root;

  std::vector<std::vector<double>> feat_vecs;
  std::vector<int> points;

  std::unique_ptr<VPNode> _build(std::vector<int> &objs, size_t i, size_t j);

  void print_tree(VPNode *node);

  void _radial_search(VPNode *node, size_t id, double r,
                      std::vector<int> &objs);

  void _nn(VPNode *node, size_t ref_id, size_t &best_id, double &best_dist);
  void _knn(VPNode *node, size_t id, double &u, NodeMaxHeap &heap, size_t n);

public:
  size_t estimatedMemoryBytes{};

  struct Metrics {
    size_t radius_sum{};
    size_t totalDistanceCalls{};
    size_t lastVisitedNodes{};
    size_t totalNodesVisited{};
    size_t totalNodesPruned{};
  } metrics;

  void build();
  bool puntal_search(size_t id);

  std::vector<int> radial_search(size_t id, double r);

  void print_tree();
  void reset_metrics();
  void reset_search_metrics();

  double get_last_prunning_rate();
  size_t get_last_visited_nodes();
  size_t get_total_distance_calls();
  double get_average_partition_radius();
  size_t get_depth() const;
  size_t get_depth(VPNode *node) const;

  int nn(size_t id);
  std::vector<int> knn(size_t id, size_t n);

  VP_tree(std::vector<Point> &data) {
    nobjs = data.size();
    feat_vecs.resize(5000); // TODO: menor?
    points.reserve(nobjs);

    for (auto &p : data) {
      feat_vecs[p.id] = p.coords;
      points.push_back(p.id);
    }
  }
};
