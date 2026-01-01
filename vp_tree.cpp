#include "vp_tree.hpp"
#include "rapidcsv.h"
#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <print>
#include <queue>

inline size_t idx(size_t i, size_t j) {
  if (i > j)
    std::swap(i, j);
  return j * (j + 1) / 2 + i;
}

inline double VP_tree::dist(size_t i, size_t j) {
  return distances[idx(i, j)];
}

void VP_tree::init_distances(std::string &dist_path) {
  std::println("Cargando distancias");

  try {
    rapidcsv::Document csv_file(
        dist_path,
        rapidcsv::LabelParams(-1, -1),
        rapidcsv::SeparatorParams(),
        rapidcsv::ConverterParams(),
        rapidcsv::LineReaderParams(true, '#', true));

    auto nrows = csv_file.GetRowCount(), ncols = csv_file.GetColumnCount();

    if (nrows == 0 || nrows != ncols)
      return;

    distances.resize(nrows * (nrows + 1) / 2);

    for (size_t i{0}; i < nrows; ++i) {
      auto row = csv_file.GetRow<double>(i);
      for (size_t j{i}; j < ncols; ++j) {
        distances[idx(i, j)] = row[j];
      }
    }

    nobjs = nrows;

  } catch (std::exception &e) {
    std::println("Exception: {}", e.what());
  }

  std::println("Distancias cargadas");
}

void VP_tree::build() {
  std::println("Construyendo arbol");

  std::vector<int> objs(nobjs);

  std::iota(objs.begin(), objs.end(), 0);

  root = _build(objs, 0, nobjs);
}

std::unique_ptr<VP_tree::Node> VP_tree::_build(std::vector<int> &objs, size_t i, size_t j) {
  if (i >= j)
    return {};

  if (i + 1 == j)
    return std::make_unique<VP_tree::Node>(objs[i], 0, nullptr, nullptr);

  std::uniform_int_distribution<> dist(i, j - 1);

  auto piv = dist(eng);
  std::swap(objs[piv], objs[j - 1]);

  auto piv_obj = objs[j - 1];
  size_t median = (j + i - 1) / 2;

  std::nth_element(objs.begin() + i, objs.begin() + median, objs.begin() + j - 1,
                   [&](int a, int b) { return VP_tree::dist(a, piv_obj) <
                                              VP_tree::dist(b, piv_obj); });

  auto distance = VP_tree::dist(piv_obj, objs[median]);

  // Aparentemente la posicion del pivot se puede ignorar/descarta
  // std::swap(objs[piv], objs[median]);

  return std::make_unique<VP_tree::Node>(piv_obj,
                                         distance,
                                         _build(objs, i, median),
                                         _build(objs, median, j - 1));
}

bool VP_tree::puntal_search(size_t id) {
  if (id > nobjs)
    return false;

  Node *node = root.get();
  while (node) {
    std::println("{}", node->id);
    if (node->id == id)
      return true;

    if (dist(id, node->id) < node->median)
      node = node->near.get();
    else
      node = node->far.get();
  }

  return false;
}

void VP_tree::print_tree(Node *node) {
  if (!node)
    return;

  std::print("{} median: {} ", node->id, node->median);
  if (!node->near && !node->far)
    std::print("(l) ");

  std::println();

  print_tree(node->near.get());
  print_tree(node->far.get());
}

void VP_tree::print_tree() {
  print_tree(root.get());
}

void VP_tree::_radial_search(Node *node, size_t id, double r, std::vector<int> &objs) {
  if (!node)
    return;

  if (dist(node->id, id) <= r)
    objs.push_back(node->id);

  if (dist(node->id, id) <= node->median + r)
    _radial_search(node->near.get(), id, r, objs);
  else
    _radial_search(node->far.get(), id, r, objs);
}

std::vector<int> VP_tree::radial_search(size_t id, double r) {
  std::vector<int> objs{};
  _radial_search(root.get(), id, r, objs);

  return objs;
}

void VP_tree::_knn(Node *node, size_t id, double r, NodeMaxHeap &heap, size_t n) {
  if (!node)
    return;

  // if (dist(node->id, id) <= r)
  //   objs.push_back(node->id);
  //
  // if (dist(node->id, id) <= node->median + r)
  //   _radial_search(node->near.get(), id, r, objs);
  // else
  //   _radial_search(node->far.get(), id, r, objs);
  //
  // return;
}

std::vector<int> VP_tree::knn(size_t id, size_t n) {
  NodeMaxHeap heap;
  auto u = std::numeric_limits<double>::max();

  _knn(root.get(), id, u, heap, n);

  std::vector<int> objs{};
  return objs;
}
