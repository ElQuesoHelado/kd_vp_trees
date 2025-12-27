#pragma once

#include <cstddef>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

class VP_tree {
  std::random_device rd;
  std::mt19937 eng{rd()};

  struct Node {
    size_t id{};
    size_t median{};
    std::unique_ptr<Node> near{}, far{};
    Node(size_t id, size_t median, std::unique_ptr<Node> &&near, std::unique_ptr<Node> &&far)
        : id(id), median(median), near(std::move(near)), far(std::move(far)) {};
  };

  size_t nobjs{};
  std::vector<double> distances;
  std::unique_ptr<Node> root;

  inline double dist(size_t i, size_t j);

  void init_distances(std::string &dist_path);

  std::unique_ptr<Node> _build(std::vector<int> &objs, size_t i, size_t j);

public:
  void build();
  bool puntal_search(size_t id);

  void print_tree(Node *node);
  void print_tree();

  VP_tree(std::string dist_path) {
    init_distances(dist_path);
  }
};
