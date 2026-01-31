/**
 * @file bindings.cpp
 * @brief pybind11 Python bindings for VectorSearch C++ core
 *
 * Exports:
 *   - HNSWIndex with numpy array interop
 *   - SIMD distance functions
 *   - Quantizers (SQ, PQ, OPQ)
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "common.hpp"
#include "hnsw.hpp"
#include "quantization.hpp"
#include "simd_ops.hpp"


namespace py = pybind11;

namespace vectorsearch {

// =============================================================================
// NUMPY HELPERS
// =============================================================================

/// Extract contiguous float32 data from numpy array
const scalar_t *get_vector_data(const py::array_t<float> &arr) {
  auto buf = arr.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Expected 1D array");
  }
  return static_cast<const scalar_t *>(buf.ptr);
}

/// Extract contiguous 2D float32 data
const scalar_t *get_vectors_data(const py::array_t<float> &arr, size_t &n,
                                 size_t &dim) {
  auto buf = arr.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Expected 2D array");
  }
  n = buf.shape[0];
  dim = buf.shape[1];
  return static_cast<const scalar_t *>(buf.ptr);
}

// =============================================================================
// HNSW INDEX WRAPPER
// =============================================================================

class PyHNSWIndex {
public:
  PyHNSWIndex(size_t dimension, size_t max_elements = 1000000, size_t M = 16,
              size_t ef_construction = 200,
              const std::string &metric = "cosine") {
    HNSWConfig config;
    config.dimension = static_cast<dim_t>(dimension);
    config.max_elements = max_elements;
    config.M = M;
    config.M_max = M * 2;
    config.ef_construction = ef_construction;

    if (metric == "cosine") {
      config.metric = MetricType::COSINE;
    } else if (metric == "l2" || metric == "euclidean") {
      config.metric = MetricType::L2;
    } else if (metric == "ip" || metric == "inner_product") {
      config.metric = MetricType::INNER_PRODUCT;
    } else {
      throw std::runtime_error("Unknown metric: " + metric);
    }

    index_ = std::make_unique<HNSWIndex>(config);
  }

  void insert(uint64_t id, const py::array_t<float> &vector) {
    py::gil_scoped_release release;
    auto result = index_->insert(id, get_vector_data(vector));
    if (result.is_err()) {
      throw std::runtime_error(result.error().message);
    }
  }

  size_t insert_batch(const py::array_t<uint64_t> &ids,
                      const py::array_t<float> &vectors) {
    auto id_buf = ids.request();
    size_t n_vectors, dim;
    const scalar_t *vec_data = get_vectors_data(vectors, n_vectors, dim);
    const vid_t *id_data = static_cast<const vid_t *>(id_buf.ptr);

    py::gil_scoped_release release;
    auto result = index_->insert_batch(id_data, vec_data, n_vectors);
    return result.unwrap();
  }

  py::list search(const py::array_t<float> &query, size_t k, size_t ef = 0) {
    std::vector<SearchResult> results;
    {
      py::gil_scoped_release release;
      results = index_->search(get_vector_data(query), k, ef);
    }

    py::list output;
    for (const auto &r : results) {
      output.append(py::make_tuple(r.id, r.score));
    }
    return output;
  }

  py::list search_batch(const py::array_t<float> &queries, size_t k,
                        size_t ef = 0) {
    size_t n_queries, dim;
    const scalar_t *query_data = get_vectors_data(queries, n_queries, dim);

    std::vector<std::vector<SearchResult>> all_results;
    {
      py::gil_scoped_release release;
      all_results = index_->search_batch(query_data, n_queries, k, ef);
    }

    py::list output;
    for (const auto &results : all_results) {
      py::list query_results;
      for (const auto &r : results) {
        query_results.append(py::make_tuple(r.id, r.score));
      }
      output.append(query_results);
    }
    return output;
  }

  bool remove(uint64_t id) {
    py::gil_scoped_release release;
    return index_->remove(id);
  }

  py::array_t<float> get_vector(uint64_t id) {
    const scalar_t *data = index_->get_vector(id);
    if (!data) {
      throw std::runtime_error("Vector not found");
    }

    size_t dim = index_->dimension();
    auto result = py::array_t<float>(dim);
    auto buf = result.request();
    std::memcpy(buf.ptr, data, dim * sizeof(float));
    return result;
  }

  bool contains(uint64_t id) const { return index_->contains(id); }

  size_t size() const { return index_->size(); }
  size_t dimension() const { return index_->dimension(); }
  size_t capacity() const { return index_->capacity(); }

  py::dict get_stats() const {
    auto stats = index_->get_stats();
    py::dict d;
    d["total_vectors"] = stats.total_vectors;
    d["deleted_vectors"] = stats.deleted_vectors;
    d["max_level"] = stats.max_level;
    d["memory_bytes"] = stats.memory_bytes;
    d["avg_connections"] = stats.avg_connections;
    return d;
  }

  void set_ef_search(size_t ef) {
    // TODO: Add setter to HNSWIndex
  }

private:
  std::unique_ptr<HNSWIndex> index_;
};

// =============================================================================
// SIMD FUNCTION WRAPPERS
// =============================================================================

float py_l2_distance(const py::array_t<float> &a, const py::array_t<float> &b) {
  auto buf_a = a.request();
  auto buf_b = b.request();
  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Dimension mismatch");
  }
  return simd::l2_distance(static_cast<const scalar_t *>(buf_a.ptr),
                           static_cast<const scalar_t *>(buf_b.ptr),
                           static_cast<dim_t>(buf_a.size));
}

float py_inner_product(const py::array_t<float> &a,
                       const py::array_t<float> &b) {
  auto buf_a = a.request();
  auto buf_b = b.request();
  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Dimension mismatch");
  }
  return simd::inner_product(static_cast<const scalar_t *>(buf_a.ptr),
                             static_cast<const scalar_t *>(buf_b.ptr),
                             static_cast<dim_t>(buf_a.size));
}

float py_cosine_similarity(const py::array_t<float> &a,
                           const py::array_t<float> &b) {
  auto buf_a = a.request();
  auto buf_b = b.request();
  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Dimension mismatch");
  }
  return simd::cosine_similarity_unnorm(
      static_cast<const scalar_t *>(buf_a.ptr),
      static_cast<const scalar_t *>(buf_b.ptr), static_cast<dim_t>(buf_a.size));
}

py::array_t<float> py_normalize(const py::array_t<float> &v) {
  auto buf = v.request();
  auto result = py::array_t<float>(buf.size);
  simd::normalize(static_cast<const scalar_t *>(buf.ptr),
                  static_cast<scalar_t *>(result.request().ptr),
                  static_cast<dim_t>(buf.size));
  return result;
}

// =============================================================================
// MODULE DEFINITION
// =============================================================================

PYBIND11_MODULE(_vectorsearch_core, m) {
  m.doc() = "VectorSearch C++ Core: High-performance SIMD-optimized vector "
            "operations";

  // Version info
  m.attr("__version__") = "1.0.0";
  m.attr("simd_support") = py::dict("avx2"_a = simd::cpu_features.avx2,
                                    "avx512"_a = simd::cpu_features.avx512f,
                                    "fma"_a = simd::cpu_features.fma);

  // SIMD distance functions
  m.def("l2_distance", &py_l2_distance,
        "Compute L2 (Euclidean) distance between two vectors", py::arg("a"),
        py::arg("b"));

  m.def("inner_product", &py_inner_product,
        "Compute inner (dot) product between two vectors", py::arg("a"),
        py::arg("b"));

  m.def("cosine_similarity", &py_cosine_similarity,
        "Compute cosine similarity between two vectors", py::arg("a"),
        py::arg("b"));

  m.def("normalize", &py_normalize, "L2-normalize a vector", py::arg("v"));

  // HNSW Index
  py::class_<PyHNSWIndex>(m, "HNSWIndex")
      .def(py::init<size_t, size_t, size_t, size_t, const std::string &>(),
           "Create HNSW index", py::arg("dimension"),
           py::arg("max_elements") = 1000000, py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("metric") = "cosine")
      .def("insert", &PyHNSWIndex::insert, "Insert single vector",
           py::arg("id"), py::arg("vector"))
      .def("insert_batch", &PyHNSWIndex::insert_batch,
           "Insert batch of vectors", py::arg("ids"), py::arg("vectors"))
      .def("search", &PyHNSWIndex::search, "Search for k nearest neighbors",
           py::arg("query"), py::arg("k"), py::arg("ef") = 0)
      .def("search_batch", &PyHNSWIndex::search_batch,
           "Batch search for k nearest neighbors", py::arg("queries"),
           py::arg("k"), py::arg("ef") = 0)
      .def("remove", &PyHNSWIndex::remove, "Remove vector by ID", py::arg("id"))
      .def("get_vector", &PyHNSWIndex::get_vector, "Get vector by ID",
           py::arg("id"))
      .def("contains", &PyHNSWIndex::contains, "Check if ID exists",
           py::arg("id"))
      .def_property_readonly("size", &PyHNSWIndex::size)
      .def_property_readonly("dimension", &PyHNSWIndex::dimension)
      .def_property_readonly("capacity", &PyHNSWIndex::capacity)
      .def("get_stats", &PyHNSWIndex::get_stats);

  // Metric type enum
  py::enum_<MetricType>(m, "MetricType")
      .value("L2", MetricType::L2)
      .value("COSINE", MetricType::COSINE)
      .value("INNER_PRODUCT", MetricType::INNER_PRODUCT);
}

} // namespace vectorsearch
