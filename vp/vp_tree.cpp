#include "vp_tree.hpp"
#include "rapidcsv.h"
#include "vp_defs.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <print>

// inline size_t idx(size_t i, size_t j) {
//   if (i > j)
//     std::swap(i, j);
//   return j * (j + 1) / 2 + i;
// }

// inline double VP_tree::ssim_dist(size_t i, size_t j) {
//   return distances[idx(i, j)];
// }

inline double VP_tree::euclidsq_dist(size_t i, size_t j) const {
  auto &a = feat_vecs[i],
       &b = feat_vecs[j];
  double sum = 0.0;
  for (size_t k = 0; k < a.size(); k++) {
    double diff = a[k] - b[k];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

// void VP_tree::init_distances(std::string &dist_path) {
//   std::println("Cargando distancias");
//
//   try {
//     rapidcsv::Document csv_file(
//         dist_path,
//         rapidcsv::LabelParams(-1, -1),
//         rapidcsv::SeparatorParams(),
//         rapidcsv::ConverterParams(),
//         rapidcsv::LineReaderParams(true, '#', true));
//
//     auto nrows = csv_file.GetRowCount(), ncols = csv_file.GetColumnCount();
//
//     if (nrows == 0 || nrows != ncols)
//       return;
//
//     distances.resize(nrows * (nrows + 1) / 2);
//
//     for (size_t i{0}; i < nrows; ++i) {
//       auto row = csv_file.GetRow<double>(i);
//       for (size_t j{i}; j < ncols; ++j) {
//         distances[idx(i, j)] = row[j];
//       }
//     }
//
//     nobjs = nrows;
//
//   } catch (std::exception &e) {
//     std::println("Exception: {}", e.what());
//   }
//
//   std::println("Distancias cargadas");
// }

void VP_tree::build() {
  root = _build(points, 0, nobjs);

  estimatedMemoryBytes =
      nobjs * (sizeof(VPNode) + 2 * sizeof(std::unique_ptr<VPNode>));
}

std::unique_ptr<VPNode> VP_tree::_build(std::vector<int> &objs,
                                        size_t i, size_t j) {
  if (i >= j)
    return {};

  if (i + 1 == j)
    return std::make_unique<VPNode>(objs[i], 0, nullptr, nullptr);

  std::uniform_int_distribution<> int_dist(i, j - 1);

  auto piv = int_dist(eng);
  std::swap(objs[piv], objs[j - 1]);

  auto piv_obj = objs[j - 1];
  size_t median = (j + i - 1) / 2;

  std::nth_element(objs.begin() + i, objs.begin() + median, objs.begin() + j - 1,
                   [&](int a, int b) { return euclidsq_dist(a, piv_obj) <
                                              euclidsq_dist(b, piv_obj); });

  auto distance = euclidsq_dist(piv_obj, objs[median]);

  metrics.radius_sum += distance;

  // Aparentemente la posicion del pivot se puede ignorar/descarta
  // std::swap(objs[piv], objs[median]);

  return std::make_unique<VPNode>(piv_obj,
                                  distance,
                                  _build(objs, i, median),
                                  _build(objs, median, j - 1));
}

bool VP_tree::puntal_search(size_t id) {
  if (id > nobjs)
    return false;

  VPNode *node = root.get();
  while (node) {
    std::println("{}", node->id);
    if (node->id == id)
      return true;

    if (euclidsq_dist(id, node->id) < node->r)
      node = node->near.get();
    else
      node = node->far.get();
  }

  return false;
}

void VP_tree::print_tree(VPNode *node) {
  if (!node)
    return;

  std::print("{} median: {} ", node->id, node->r);
  if (!node->near && !node->far)
    std::print("(l) ");

  std::println();

  print_tree(node->near.get());
  print_tree(node->far.get());
}

void VP_tree::print_tree() {
  print_tree(root.get());
}

void VP_tree::reset_metrics() {
  metrics = {};
}

void VP_tree::reset_search_metrics() {
  metrics.lastVisitedNodes = 0;
  // metrics.totalDistanceCalls
}

double VP_tree::get_last_prunning_rate() {
  if (nobjs == 0)
    return 0;
  double nodesNotVisited = nobjs - metrics.lastVisitedNodes;
  return nodesNotVisited / nobjs;
}

size_t VP_tree::get_last_visited_nodes() {
  return metrics.lastVisitedNodes;
}

size_t VP_tree::get_total_distance_calls() {
  return metrics.totalDistanceCalls;
}

double VP_tree::get_average_partition_radius() {
  if (nobjs == 0)
    return 0;

  return static_cast<double>(metrics.radius_sum) / nobjs;
}

size_t VP_tree::get_depth() const {
  return get_depth(root.get());
}

size_t VP_tree::get_depth(VPNode *node) const {
  if (!node)
    return 0;

  return 1 + std::max(get_depth(node->near.get()),
                      get_depth(node->far.get()));
}

void VP_tree::_radial_search(VPNode *node, size_t id, double r, std::vector<int> &objs) {
  if (!node)
    return;

  if (euclidsq_dist(node->id, id) <= r)
    objs.push_back(node->id);

  if (euclidsq_dist(node->id, id) <= node->r + r)
    _radial_search(node->near.get(), id, r, objs);
  else
    _radial_search(node->far.get(), id, r, objs);
}

std::vector<int> VP_tree::radial_search(size_t id, double r) {
  std::vector<int> objs{};
  _radial_search(root.get(), id, r, objs);

  return objs;
}

void VP_tree::_knn(VPNode *node, size_t ref_id, double &u, NodeMaxHeap &heap, size_t n) {
  if (!node)
    return;

  metrics.lastVisitedNodes++;
  metrics.totalNodesVisited++;
  metrics.totalDistanceCalls++;

  auto d = euclidsq_dist(node->id, ref_id);

  if (d < u) {
    if (heap.size() == n)
      heap.pop();
    heap.push({node->id, d});

    u = heap.top().d;
  }

  if (d < node->r) {
    _knn(node->near.get(), ref_id, u, heap, n);
    if (d + u >= node->r)
      _knn(node->far.get(), ref_id, u, heap, n);
  } else {
    _knn(node->far.get(), ref_id, u, heap, n);
    if (d - u <= node->r)
      _knn(node->near.get(), ref_id, u, heap, n);
  }
}

std::vector<int> VP_tree::knn(size_t ref_id, size_t n) {
  NodeMaxHeap heap;
  auto u = std::numeric_limits<double>::max();

  _knn(root.get(), ref_id, u, heap, n);

  std::vector<int> objs;
  objs.reserve(n);

  while (!heap.empty()) {
    objs.push_back(heap.top().id);
    heap.pop();
  }

  return objs;
}

int VP_tree::nn(size_t ref_id) {
  size_t best_id = ref_id;
  double best_dist = std::numeric_limits<double>::max();

  _nn(root.get(), ref_id, best_id, best_dist);
  return best_id;
}

void VP_tree::_nn(VPNode *node, size_t ref_id,
                  size_t &best_id, double &best_dist) {
  if (!node)
    return;

  metrics.lastVisitedNodes++;
  metrics.totalNodesVisited++;
  metrics.totalDistanceCalls++;

  double d = euclidsq_dist(node->id, ref_id);

  if (d < best_dist) {
    best_dist = d;
    best_id = node->id;
  }

  double r = node->r;

  if (d < r) {
    _nn(node->near.get(), ref_id, best_id, best_dist);
    if (d + best_dist >= r)
      _nn(node->far.get(), ref_id, best_id, best_dist);
  } else {
    _nn(node->far.get(), ref_id, best_id, best_dist);
    if (d - best_dist <= r)
      _nn(node->near.get(), ref_id, best_id, best_dist);
  }
}
