#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

class Point {
public:
  vector<double> coords;
  int id;

  Point(const vector<double> &c = {}, int i = -1) : coords(c), id(i) {}

  double distance(const Point &other) const {
    double dist = 0.0;
    for (size_t i = 0; i < coords.size(); ++i) {
      double diff = coords[i] - other.coords[i];
      dist += diff * diff;
    }
    return sqrt(dist);
  }

  double operator[](size_t i) const { return coords[i]; }
  size_t size() const { return coords.size(); }
};

struct KDNode {
  Point point;
  int axis;
  unique_ptr<KDNode> left;
  unique_ptr<KDNode> right;

  KDNode(const Point &p, int a)
      : point(p), axis(a), left(nullptr), right(nullptr) {}
};

class KDTree {
private:
  unique_ptr<KDNode> root;
  int dimensions;
  int treeSize;

  struct AxisComparator {
    int axis;
    AxisComparator(int a) : axis(a) {}
    bool operator()(const Point &a, const Point &b) const {
      return a.coords[axis] < b.coords[axis];
    }
  };

  unique_ptr<KDNode> buildTree(vector<Point> &points, int depth, int start,
                               int end) {
    if (start >= end)
      return nullptr;

    int axis = depth % dimensions;
    int mid = start + (end - start) / 2;

    nth_element(points.begin() + start, points.begin() + mid,
                points.begin() + end, AxisComparator(axis));

    auto node = make_unique<KDNode>(points[mid], axis);
    node->left = buildTree(points, depth + 1, start, mid);
    node->right = buildTree(points, depth + 1, mid + 1, end);

    return node;
  }

  void nearestNeighbor(const KDNode *node, const Point &target,
                       const KDNode *&best, double &bestDist) const {
    if (!node)
      return;

    double dist = node->point.distance(target);
    if (dist < bestDist) {
      bestDist = dist;
      best = node;
    }

    int axis = node->axis;
    double diff = target[axis] - node->point[axis];

    const KDNode *first = diff < 0 ? node->left.get() : node->right.get();
    const KDNode *second = diff < 0 ? node->right.get() : node->left.get();

    nearestNeighbor(first, target, best, bestDist);

    if (fabs(diff) < bestDist) {
      nearestNeighbor(second, target, best, bestDist);
    }
  }

  void
  kNearestNeighbors(const KDNode *node, const Point &target, int k,
                    priority_queue<pair<double, const Point *>> &heap) const {
    if (!node)
      return;

    double dist = node->point.distance(target);

    if (heap.size() < k) {
      heap.push({dist, &node->point});
    } else if (dist < heap.top().first) {
      heap.pop();
      heap.push({dist, &node->point});
    }

    int axis = node->axis;
    double diff = target[axis] - node->point[axis];

    const KDNode *first = diff < 0 ? node->left.get() : node->right.get();
    const KDNode *second = diff < 0 ? node->right.get() : node->left.get();

    kNearestNeighbors(first, target, k, heap);

    if (heap.size() < k || fabs(diff) < heap.top().first) {
      kNearestNeighbors(second, target, k, heap);
    }
  }

  void insert(unique_ptr<KDNode> &node, const Point &point, int depth) {
    if (!node) {
      node = make_unique<KDNode>(point, depth % dimensions);
      treeSize++;
      return;
    }

    int axis = node->axis;
    if (point[axis] < node->point[axis]) {
      insert(node->left, point, depth + 1);
    } else {
      insert(node->right, point, depth + 1);
    }
  }

  int calculateDepth(const KDNode *node) const {
    if (!node)
      return 0;
    return 1 + max(calculateDepth(node->left.get()),
                   calculateDepth(node->right.get()));
  }

public:
  double buildTimeMs;
  double totalInsertionTimeMs;
  double totalSearchTimeMs;
  size_t estimatedMemoryBytes;

  KDTree()
      : root(nullptr), dimensions(0), treeSize(0), buildTimeMs(0),
        totalInsertionTimeMs(0), totalSearchTimeMs(0), estimatedMemoryBytes(0) {
  }

  void build(vector<Point> &points) {
    if (points.empty())
      return;

    auto start = high_resolution_clock::now();

    dimensions = points[0].size();
    treeSize = points.size();

    root = buildTree(points, 0, 0, points.size());

    auto end = high_resolution_clock::now();
    buildTimeMs = duration_cast<microseconds>(end - start).count() / 1000.0;

    estimatedMemoryBytes =
        treeSize * (sizeof(KDNode) + dimensions * sizeof(double));
  }

  void insertPoint(const Point &point) {
    auto start = high_resolution_clock::now();

    if (dimensions == 0) {
      dimensions = point.size();
    }

    insert(root, point, 0);

    auto end = high_resolution_clock::now();
    double insertionTime =
        duration_cast<microseconds>(end - start).count() / 1000.0;
    totalInsertionTimeMs += insertionTime;
  }

  Point nearestNeighbor(const Point &target, double &searchTime) {
    auto start = high_resolution_clock::now();

    const KDNode *best = nullptr;
    double bestDist = numeric_limits<double>::max();

    nearestNeighbor(root.get(), target, best, bestDist);

    auto end = high_resolution_clock::now();
    searchTime = duration_cast<microseconds>(end - start).count() / 1000.0;
    totalSearchTimeMs += searchTime;

    return best ? best->point : Point();
  }

  vector<Point> kNearestNeighbors(const Point &target, int k,
                                  double &searchTime) {
    auto start = high_resolution_clock::now();

    priority_queue<pair<double, const Point *>> heap;

    kNearestNeighbors(root.get(), target, k, heap);

    vector<Point> result;
    while (!heap.empty()) {
      result.push_back(*heap.top().second);
      heap.pop();
    }
    reverse(result.begin(), result.end());

    auto end = high_resolution_clock::now();
    searchTime = duration_cast<microseconds>(end - start).count() / 1000.0;
    totalSearchTimeMs += searchTime;

    return result;
  }

  int getDepth() const { return calculateDepth(root.get()); }
  double getBalanceFactor() const {
    int depth = getDepth();
    if (depth == 0 || treeSize == 0)
      return 0.0;
    double idealDepth = log2(treeSize + 1);
    return depth / idealDepth;
  }

  double getBuildTime() const { return buildTimeMs; }
  double getTotalInsertionTime() const { return totalInsertionTimeMs; }
  double getAverageInsertionTime() const {
    return treeSize > 0 ? totalInsertionTimeMs / treeSize : 0.0;
  }

  int size() const { return treeSize; }
  int getDimensions() const { return dimensions; }
};

vector<Point> readCSV(const string &filename, int maxRows = -1,
                      int numFeatures = -1) {
  ifstream file(filename);
  vector<Point> points;
  string line;
  int rowCount = 0;

  if (!file.is_open()) {
    cerr << "Error: No se pudo abrir el archivo " << filename << endl;
    return points;
  }

  while (getline(file, line) && (maxRows == -1 || rowCount < maxRows)) {
    stringstream ss(line);
    string value;

    if (!getline(ss, value, ','))
      continue;

    int id;
    try {
      id = stoi(value);
    } catch (...) {
      cerr << "Error al leer ID: " << value << endl;
      continue;
    }

    vector<double> coords;
    int featureCount = 0;

    while (getline(ss, value, ',')) {
      if (numFeatures != -1 && featureCount >= numFeatures)
        break;

      try {
        coords.push_back(stod(value));
        featureCount++;
      } catch (...) {
        cerr << "Error al convertir valor: " << value << endl;
        coords.push_back(0.0);
        featureCount++;
      }
    }

    if (!coords.empty()) {
      points.emplace_back(coords, id);
      rowCount++;
    }
  }

  file.close();
  return points;
}

void saveMetricsToCSV(const string &filename,
                      const vector<vector<string>> &metrics,
                      const vector<string> &headers) {
  ofstream file(filename);

  if (!file.is_open()) {
    cerr << "Error: No se pudo crear el archivo " << filename << endl;
    return;
  }

  for (size_t i = 0; i < headers.size(); ++i) {
    file << headers[i];
    if (i < headers.size() - 1)
      file << ",";
  }
  file << endl;

  for (const auto &row : metrics) {
    for (size_t i = 0; i < row.size(); ++i) {
      file << row[i];
      if (i < row.size() - 1)
        file << ",";
    }
    file << endl;
  }

  file.close();
  cout << "Métricas guardadas en " << filename << endl;
}

void generateStatisticalSummary(const vector<vector<string>> &allResults) {
  ofstream summary("resumen_estadistico.txt");

  summary << "=== RESUMEN ESTADÍSTICO EXPERIMENTOS KD-TREE ===\n\n";

  // Análisis por dimensiones
  map<int, vector<double>> nnBalancedByDim, nnUnbalancedByDim;
  map<int, vector<double>> knnBalancedByDim, knnUnbalancedByDim;

  for (const auto &row : allResults) {
    if (row.size() < 14)
      continue;

    int dims = stoi(row[0]);
    string tipo = row[4];
    double avgNN = stod(row[8]);
    double avgKNN = stod(row[10]);

    if (tipo == "balanceado") {
      nnBalancedByDim[dims].push_back(avgNN);
      knnBalancedByDim[dims].push_back(avgKNN);
    } else {
      nnUnbalancedByDim[dims].push_back(avgNN);
      knnUnbalancedByDim[dims].push_back(avgKNN);
    }
  }

  summary << "TIEMPO PROMEDIO BÚSQUEDA NN POR DIMENSIÓN (ms):\n";
  summary << "Dimensión | Balanceado | Desbalanceado | Mejora %\n";
  summary << string(55, '-') << "\n";

  for (const auto &[dims, times] : nnBalancedByDim) {
    if (nnUnbalancedByDim[dims].empty())
      continue;

    double avgBal = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avgUnb = accumulate(nnUnbalancedByDim[dims].begin(),
                               nnUnbalancedByDim[dims].end(), 0.0) /
                    nnUnbalancedByDim[dims].size();
    double mejora = ((avgUnb - avgBal) / avgUnb) * 100.0;

    summary << fixed << setprecision(2);
    summary << setw(9) << dims << " | " << setw(10) << avgBal << " | "
            << setw(13) << avgUnb << " | " << setw(8) << setprecision(1)
            << mejora << "%\n";
  }

  summary << "\nTIEMPO PROMEDIO BÚSQUEDA kNN POR DIMENSIÓN (ms):\n";
  summary << "Dimensión | Balanceado | Desbalanceado | Mejora %\n";
  summary << string(55, '-') << "\n";

  for (const auto &[dims, times] : knnBalancedByDim) {
    if (knnUnbalancedByDim[dims].empty())
      continue;

    double avgBal = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avgUnb = accumulate(knnUnbalancedByDim[dims].begin(),
                               knnUnbalancedByDim[dims].end(), 0.0) /
                    knnUnbalancedByDim[dims].size();
    double mejora = ((avgUnb - avgBal) / avgUnb) * 100.0;

    summary << fixed << setprecision(2);
    summary << setw(9) << dims << " | " << setw(10) << avgBal << " | "
            << setw(13) << avgUnb << " | " << setw(8) << setprecision(1)
            << mejora << "%\n";
  }

  summary.close();
  cout << "Resumen estadístico guardado en resumen_estadistico.txt\n";
}

int main() {
  string inputFile = "images_dataset.csv";

  cout << "=== EXPERIMENTOS MULTIDIMENSIONALES KD-TREE ===\n";

  vector<int> dimensionsToTest = {2, 6, 10, 14};
  vector<int> dataSizes = {200, 500, 800, 1100, 1400, 1600, 1800, 2000, 2200};
  vector<int> searchCounts = {10, 50, 100, 200};
  vector<int> kValues = {1, 5, 10, 20};

  // Leer dataset base
  cout << "\nCargando dataset base..." << endl;
  vector<Point> baseData = readCSV(inputFile, 20000, -1); // Máximo 20k puntos

  if (baseData.empty()) {
    cerr << "Error: No se pudieron cargar datos del archivo" << endl;
    return 1;
  }

  cout << "Dataset base cargado: " << baseData.size() << " puntos con "
       << baseData[0].size() << " dimensiones\n";

  // Cabeceras para el CSV de resultados
  vector<string> headers = {"dimensiones",
                            "datos_entrenamiento",
                            "datos_busqueda",
                            "k_vecinos",
                            "tipo_arbol",
                            "tiempo_construccion_ms",
                            "tiempo_insercion_total_ms",
                            "tiempo_busqueda_nn_total_ms",
                            "tiempo_busqueda_nn_promedio_ms",
                            "tiempo_busqueda_knn_total_ms",
                            "tiempo_busqueda_knn_promedio_ms",
                            "profundidad_arbol",
                            "factor_balance",
                            "memoria_estimada_kb"};

  vector<vector<string>> allResults;

  int totalExperiments = 0;

  for (int dims : dimensionsToTest) {
    if (dims > (int)baseData[0].size())
      continue;

    for (int dataSize : dataSizes) {
      if (dataSize > (int)baseData.size())
        continue;

      // Crear subconjunto con dimensiones reducidas
      vector<Point> dataset;
      for (int i = 0; i < dataSize; i++) {
        vector<double> coords(baseData[i].coords.begin(),
                              baseData[i].coords.begin() + dims);
        dataset.push_back(Point(coords, baseData[i].id));
      }

      cout << "\n[Experimento] Dims: " << dims << ", Datos: " << dataSize
           << endl;

      for (int searchCount : searchCounts) {
        if (searchCount > dataSize / 2)
          continue;

        vector<Point> queryPoints;
        int startIdx = dataSize / 2;
        int endIdx = min(startIdx + searchCount, dataSize);

        for (int i = startIdx; i < endIdx; i++) {
          queryPoints.push_back(dataset[i]);
        }

        for (int k : kValues) {
          if (k > dataSize)
            continue;

          totalExperiments++;

          KDTree balancedTree;
          auto startBuild = high_resolution_clock::now();
          balancedTree.build(dataset);
          auto endBuild = high_resolution_clock::now();
          double buildTime =
              duration_cast<microseconds>(endBuild - startBuild).count() /
              1000.0;

          double totalNNTimeBal = 0, totalKNNTimeBal = 0;
          for (const auto &query : queryPoints) {
            double searchTime;
            balancedTree.nearestNeighbor(query, searchTime);
            totalNNTimeBal += searchTime;

            balancedTree.kNearestNeighbors(query, k, searchTime);
            totalKNNTimeBal += searchTime;
          }

          double avgNNBal = totalNNTimeBal / queryPoints.size();
          double avgKNNBal = totalKNNTimeBal / queryPoints.size();

          allResults.push_back(
              {to_string(dims), to_string(dataSize),
               to_string(queryPoints.size()), to_string(k), "balanceado",
               to_string(buildTime), "0", to_string(totalNNTimeBal),
               to_string(avgNNBal), to_string(totalKNNTimeBal),
               to_string(avgKNNBal), to_string(balancedTree.getDepth()),
               to_string(balancedTree.getBalanceFactor()),
               to_string(balancedTree.estimatedMemoryBytes / 1024.0)});

          // ===== ÁRBOL DESBALANCEADO =====
          if (dataSize >= 10) {
            KDTree unbalancedTree;

            // Construir con primer punto
            vector<Point> firstPoint = {dataset[0]};
            unbalancedTree.build(firstPoint);

            // Insertar puntos restantes
            double totalInsertTime = 0;
            int insertions = min(500, dataSize - 1);
            for (int i = 1; i <= insertions; i++) {
              auto startInsert = high_resolution_clock::now();
              unbalancedTree.insertPoint(dataset[i]);
              auto endInsert = high_resolution_clock::now();
              totalInsertTime +=
                  duration_cast<microseconds>(endInsert - startInsert).count() /
                  1000.0;
            }

            double totalNNTimeUnb = 0, totalKNNTimeUnb = 0;
            for (const auto &query : queryPoints) {
              double searchTime;
              unbalancedTree.nearestNeighbor(query, searchTime);
              totalNNTimeUnb += searchTime;

              unbalancedTree.kNearestNeighbors(query, k, searchTime);
              totalKNNTimeUnb += searchTime;
            }

            double avgNNUnb = totalNNTimeUnb / queryPoints.size();
            double avgKNNUnb = totalKNNTimeUnb / queryPoints.size();

            allResults.push_back(
                {to_string(dims), to_string(dataSize),
                 to_string(queryPoints.size()), to_string(k), "desbalanceado",
                 "0", to_string(totalInsertTime), to_string(totalNNTimeUnb),
                 to_string(avgNNUnb), to_string(totalKNNTimeUnb),
                 to_string(avgKNNUnb), to_string(unbalancedTree.getDepth()),
                 to_string(unbalancedTree.getBalanceFactor()),
                 to_string(unbalancedTree.estimatedMemoryBytes / 1024.0)});
          }
        }
      }
    }
  }

  // Guardar resultados
  string timestamp = to_string(
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count());
  string resultsFile = "resultados_experimentos_" + timestamp + ".csv";

  saveMetricsToCSV(resultsFile, allResults, headers);

  generateStatisticalSummary(allResults);

  // Guardar archivo de configuración
  ofstream config("configuracion_experimentos.txt");
  config << "=== CONFIGURACIÓN EXPERIMENTOS ===\n\n";
  config << "Dataset: " << inputFile << "\n";
  config << "Fecha: " << __DATE__ << " " << __TIME__ << "\n\n";

  config << "Dimensiones probadas: ";
  for (size_t i = 0; i < dimensionsToTest.size(); i++) {
    config << dimensionsToTest[i];
    if (i < dimensionsToTest.size() - 1)
      config << ", ";
  }
  config << "\n";

  config << "Tamaños de datos: ";
  for (size_t i = 0; i < dataSizes.size(); i++) {
    config << dataSizes[i];
    if (i < dataSizes.size() - 1)
      config << ", ";
  }
  config << "\n";

  config << "Búsquedas por prueba: ";
  for (size_t i = 0; i < searchCounts.size(); i++) {
    config << searchCounts[i];
    if (i < searchCounts.size() - 1)
      config << ", ";
  }
  config << "\n";

  config << "Valores de k: ";
  for (size_t i = 0; i < kValues.size(); i++) {
    config << kValues[i];
    if (i < kValues.size() - 1)
      config << ", ";
  }
  config << "\n\n";

  config << "Total experimentos: " << totalExperiments << "\n";
  config << "Resultados guardados en: " << resultsFile << "\n";
  config.close();

  cout << "\n=== EXPERIMENTOS COMPLETADOS ===" << endl;
  cout << "Total experimentos realizados: " << totalExperiments << endl;
  cout << "Resultados principales: " << resultsFile << endl;
  cout << "Resumen estadístico: resumen_estadistico.txt" << endl;
  cout << "Configuración: configuracion_experimentos.txt" << endl;

  return 0;
}