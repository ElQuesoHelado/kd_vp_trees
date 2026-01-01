#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

class VP_tree {
  std::random_device rd;
  std::mt19937 eng{rd()};

  struct Node {
    size_t id{};
    double median{};
    std::unique_ptr<Node> near{}, far{};
    Node(size_t id, double median, std::unique_ptr<Node> &&near, std::unique_ptr<Node> &&far)
        : id(id), median(median), near(std::move(near)), far(std::move(far)) {};
  };

  typedef std::priority_queue<std::reference_wrapper<Node>, std::vector<std::reference_wrapper<Node>>,
                              decltype([](const Node &lhs, const Node &rhs) {
                                return lhs.median <= rhs.median;
                              })>
      NodeMaxHeap;

  size_t nobjs{};
  std::vector<double> distances;
  std::unique_ptr<Node> root;

  inline double dist(size_t i, size_t j);

  void init_distances(std::string &dist_path);

  std::unique_ptr<Node> _build(std::vector<int> &objs, size_t i, size_t j);
  void print_tree(Node *node);

  void _radial_search(Node *node, size_t id, double r, std::vector<int> &objs);

  void _knn(Node *node, size_t id, double u, NodeMaxHeap &heap, size_t n);

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
