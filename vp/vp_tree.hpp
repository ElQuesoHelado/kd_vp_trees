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

class VP_tree {
  std::random_device rd;
  std::mt19937 eng{rd()};

  inline double dist(size_t i, size_t j);

  typedef std::priority_queue<VPNeig, std::vector<VPNeig>,
                              decltype([](const VPNeig &lhs, const VPNeig &rhs) {
                                return lhs.d <= rhs.d;
                              })>
      NodeMaxHeap;

  size_t nobjs{};
  std::vector<double> distances;
  std::unique_ptr<VPNode> root;

  void init_distances(std::string &dist_path);

  std::unique_ptr<VPNode> _build(std::vector<int> &objs, size_t i, size_t j);
  void print_tree(VPNode *node);

  void _radial_search(VPNode *node, size_t id, double r, std::vector<int> &objs);

  void _knn(VPNode *node, size_t id, double &u, NodeMaxHeap &heap, size_t n);

public:
  void build();
  bool puntal_search(size_t id);

  std::vector<int> radial_search(size_t id, double r);

  void print_tree();

  std::vector<int> knn(size_t id, size_t n);

  VP_tree(std::string dist_path) {
    init_distances(dist_path);
  }
};
