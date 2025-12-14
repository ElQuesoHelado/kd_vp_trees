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

  class Node {
    size_t id;
    std::unique_ptr<Node> near, far;
  };

  size_t nobjs{};
  std::vector<int> distances;

  void init_distances(std::string &dist_path);
  void build();
  void _build(std::vector<int> &objs, size_t i, size_t j);

public:
  VP_tree(std::string dist_path) {
    init_distances(dist_path);
  }
};
