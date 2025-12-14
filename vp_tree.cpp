#include "vp_tree.hpp"
#include "rapidcsv.h"
#include <cstddef>
#include <exception>
#include <print>

inline size_t idx(size_t i, size_t j) {
  if (i > j)
    std::swap(i, j);
  return j * (j + 1) / 2 + i;
}

void VP_tree::init_distances(std::string &dist_path) {
  try {
    rapidcsv::Document csv_file(
        dist_path,
        rapidcsv::LabelParams(-1, -1));

    auto nrows = csv_file.GetRowCount(), ncols = csv_file.GetColumnCount();

    if (nrows == 0 || nrows != ncols)
      return;

    distances.resize(nrows * (nrows + 1) / 2);

    for (size_t i{0}; i < nrows; ++i) {
      auto row = csv_file.GetRow<int>(i);
      for (size_t j{i}; j < ncols; ++j) {
        distances[idx(i, j)] = row[j];
      }
    }

    nobjs = nrows;

  } catch (std::exception &e) {
    std::println("{}", e.what());
  }
}

void VP_tree::build() {
  std::vector<int> objs(nobjs);

  _build(objs, 0, nobjs);
}

void VP_tree::_build(std::vector<int> &objs, size_t i, size_t j) {
  std::uniform_int_distribution<> dist(i, j);

  auto piv = dist(eng);

  double median{};
  for (size_t k{i}; i < j; ++k) {
    median += distances[idx(piv, k)];
  }

  median /= (j - i + 1);

  std::swap(objs[0], objs[piv]);
}
