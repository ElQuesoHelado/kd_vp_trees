#include "vp_tree.hpp"
#include "rapidcsv.h"
#include <cstddef>
#include <exception>
#include <memory>
#include <numeric>
#include <print>

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
        rapidcsv::LineReaderParams(true, '#', true) // El último 'true' es para skipEmptyLines
    );

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
  // if (i == j)
  //   return std::make_unique<VP_tree::Node>(objs[i], 0, nullptr, nullptr);

  std::uniform_int_distribution<> dist(i, j - 1);

  auto piv = dist(eng);

  double median{};
  for (size_t k{i}; k < j - 1; ++k) {
    median += VP_tree::dist(piv, k);
  }

  median /= ((double)j - (double)i + 1.f);

  std::swap(objs[piv], objs[j - 1]);

  int b_piv = (int)i - 1;

  for (int p = i; p < j - 1; p++) {
    if (VP_tree::dist(piv, p) < median) {
      b_piv++;
      std::swap(objs[p], objs[b_piv]);
    }
  }

  b_piv++;
  std::swap(objs[j - 1], objs[b_piv]);

  return std::make_unique<VP_tree::Node>(piv, median,
                                         _build(objs, i, b_piv),
                                         _build(objs, b_piv + 1, j));
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

  std::println("{} ", node->id);
  if (!node->near && !node->far)
    std::print("(l) ");

  std::print("Near: ");
  print_tree(node->near.get());
  std::print("Far: ");
  print_tree(node->far.get());
}

void VP_tree::print_tree() {
  print_tree(root.get());
}
