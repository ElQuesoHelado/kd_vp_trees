#pragma once

#include <cstddef>
#include <memory>

struct VPNode {
  size_t id{};
  double r{};
  std::unique_ptr<VPNode> near{}, far{};
  VPNode(size_t id, double median, std::unique_ptr<VPNode> &&near, std::unique_ptr<VPNode> &&far)
      : id(id), r(median), near(std::move(near)), far(std::move(far)) {};
};

struct VPNeig {
  size_t id{};
  double d{}; // Distancia a objeto referencia
};
