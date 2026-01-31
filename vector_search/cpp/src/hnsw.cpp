/**
 * @file hnsw.cpp
 * @brief HNSW index implementation
 */

#include "hnsw.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace vectorsearch {

// =============================================================================
// NEIGHBOR LIST IMPLEMENTATION
// =============================================================================

NeighborList::NeighborList(level_t max_level, size_t M, size_t M_max)
    : M_(M), M_max_(M_max) {
  // Calculate offsets for each level
  offsets_.resize(max_level + 1);
  counts_.resize(max_level + 1, 0);

  size_t total = 0;
  for (level_t l = 0; l <= max_level; ++l) {
    offsets_[l] = total;
    total += (l == 0) ? M_max : M;
  }

  data_.resize(total, std::numeric_limits<idx_t>::max());
}

idx_t *NeighborList::neighbors(level_t level) {
  return data_.data() + offsets_[level];
}

const idx_t *NeighborList::neighbors(level_t level) const {
  return data_.data() + offsets_[level];
}

size_t NeighborList::count(level_t level) const { return counts_[level]; }

void NeighborList::set_count(level_t level, size_t count) {
  counts_[level] = count;
}

size_t NeighborList::capacity(level_t level) const {
  return (level == 0) ? M_max_ : M_;
}

bool NeighborList::add(level_t level, idx_t neighbor) {
  if (counts_[level] >= capacity(level))
    return false;
  neighbors(level)[counts_[level]++] = neighbor;
  return true;
}

bool NeighborList::remove(level_t level, idx_t neighbor) {
  idx_t *nbrs = neighbors(level);
  for (size_t i = 0; i < counts_[level]; ++i) {
    if (nbrs[i] == neighbor) {
      nbrs[i] = nbrs[--counts_[level]];
      return true;
    }
  }
  return false;
}

void NeighborList::clear() { std::fill(counts_.begin(), counts_.end(), 0); }

// =============================================================================
// RESULT HEAP IMPLEMENTATION
// =============================================================================

void ResultHeap::push(vid_t id, score_t score, idx_t idx) {
  SearchResult result{id, score, idx};

  if (heap_.size() < k_) {
    heap_.push_back(result);
    std::push_heap(heap_.begin(), heap_.end(),
                   [](const SearchResult &a, const SearchResult &b) {
                     return a.score > b.score; // Min-heap
                   });
  } else if (score > heap_[0].score) {
    std::pop_heap(heap_.begin(), heap_.end(),
                  [](const SearchResult &a, const SearchResult &b) {
                    return a.score > b.score;
                  });
    heap_.back() = result;
    std::push_heap(heap_.begin(), heap_.end(),
                   [](const SearchResult &a, const SearchResult &b) {
                     return a.score > b.score;
                   });
  }
}

bool ResultHeap::should_add(score_t score) const {
  return heap_.size() < k_ || score > heap_[0].score;
}

std::vector<SearchResult> ResultHeap::extract_sorted() {
  std::sort(heap_.begin(), heap_.end(),
            [](const SearchResult &a, const SearchResult &b) {
              return a.score > b.score; // Descending by score
            });
  return std::move(heap_);
}

score_t ResultHeap::worst_score() const {
  return heap_.empty() ? std::numeric_limits<score_t>::lowest()
                       : heap_[0].score;
}

// =============================================================================
// HNSW INDEX IMPLEMENTATION
// =============================================================================

HNSWIndex::HNSWIndex(const HNSWConfig &config)
    : config_(config), rng_(config.seed),
      distance_func_(simd::get_distance_func(config.metric)),
      is_similarity_metric_(is_similarity_metric(config.metric)) {
  // Reserve capacity
  vectors_.reserve(config_.max_elements * config_.dimension);
  nodes_.reserve(config_.max_elements);
  neighbors_.reserve(config_.max_elements);
}

HNSWIndex::~HNSWIndex() = default;

HNSWIndex::HNSWIndex(HNSWIndex &&) noexcept = default;
HNSWIndex &HNSWIndex::operator=(HNSWIndex &&) noexcept = default;

// =============================================================================
// RANDOM LEVEL GENERATION
// =============================================================================

level_t HNSWIndex::random_level() {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  level_t level = 0;
  while (dist(rng_) < std::exp(-static_cast<float>(level) * config_.ml) &&
         level < 16) {
    ++level;
  }
  return level;
}

// =============================================================================
// DISTANCE COMPUTATION
// =============================================================================

score_t HNSWIndex::distance(const scalar_t *a, const scalar_t *b) const {
  return distance_func_(a, b, config_.dimension);
}

const scalar_t *HNSWIndex::get_vector_by_idx(idx_t idx) const {
  return vectors_.data() + static_cast<size_t>(idx) * config_.dimension;
}

// =============================================================================
// INSERT
// =============================================================================

Result<void> HNSWIndex::insert(vid_t id, const scalar_t *vector) {
  std::unique_lock lock(rw_mutex_);

  // Check if ID exists
  if (id_to_idx_.count(id)) {
    // Update existing vector
    idx_t idx = id_to_idx_[id];
    scalar_t *dst =
        vectors_.data() + static_cast<size_t>(idx) * config_.dimension;

    // Normalize if using cosine
    if (config_.metric == MetricType::COSINE) {
      simd::normalize(vector, dst, config_.dimension);
    } else {
      std::memcpy(dst, vector, config_.dimension * sizeof(scalar_t));
    }
    return Result<void>::Ok({});
  }

  // Check capacity
  if (count_.load() >= config_.max_elements) {
    return Result<void>::Err(Error("Index at capacity"));
  }

  // Allocate new node
  idx_t new_idx = static_cast<idx_t>(nodes_.size());
  level_t level = random_level();

  // Store vector (normalized if cosine)
  size_t vec_offset = vectors_.size();
  vectors_.resize(vectors_.size() + config_.dimension);
  scalar_t *dst = vectors_.data() + vec_offset;

  if (config_.metric == MetricType::COSINE) {
    simd::normalize(vector, dst, config_.dimension);
  } else {
    std::memcpy(dst, vector, config_.dimension * sizeof(scalar_t));
  }

  // Create node
  HNSWNode node{};
  node.id = id;
  node.level = level;
  node.deleted = 0;
  node.vector_offset = static_cast<idx_t>(vec_offset / config_.dimension);
  nodes_.push_back(node);

  // Create neighbor list
  neighbors_.push_back(
      std::make_unique<NeighborList>(level, config_.M, config_.M_max));

  // Update ID mapping
  id_to_idx_[id] = new_idx;

  // Handle first node
  idx_t current_entry = entry_point_.load(std::memory_order_relaxed);
  if (current_entry == std::numeric_limits<idx_t>::max()) {
    entry_point_.store(new_idx, std::memory_order_release);
    max_level_.store(level, std::memory_order_release);
    count_.fetch_add(1, std::memory_order_release);
    return Result<void>::Ok({});
  }

  // Find neighbors and connect
  const scalar_t *query = get_vector_by_idx(new_idx);
  level_t current_max_level = max_level_.load(std::memory_order_acquire);
  std::vector<std::pair<score_t, idx_t>> candidates;

  // Traverse from top to node's level
  for (level_t l = current_max_level; l > level; --l) {
    candidates.clear();
    search_layer(query, current_entry, 1, l, candidates);
    if (!candidates.empty()) {
      current_entry = candidates[0].second;
    }
  }

  // Insert at each layer
  for (level_t l = std::min(level, current_max_level);; --l) {
    candidates.clear();
    search_layer(query, current_entry, config_.ef_construction, l, candidates);

    size_t M = (l == 0) ? config_.M_max : config_.M;
    auto neighbors = select_neighbors(query, candidates, M);

    connect(new_idx, neighbors, l);

    if (!candidates.empty()) {
      current_entry = candidates[0].second;
    }

    if (l == 0)
      break;
  }

  // Update entry point if new node has higher level
  if (level > current_max_level) {
    entry_point_.store(new_idx, std::memory_order_release);
    max_level_.store(level, std::memory_order_release);
  }

  count_.fetch_add(1, std::memory_order_release);
  return Result<void>::Ok({});
}

Result<size_t> HNSWIndex::insert_batch(const vid_t *ids,
                                       const scalar_t *vectors, size_t count) {
  size_t inserted = 0;
  for (size_t i = 0; i < count; ++i) {
    auto result = insert(ids[i], vectors + i * config_.dimension);
    if (result.is_ok()) {
      ++inserted;
    }
  }
  return Result<size_t>::Ok(inserted);
}

// =============================================================================
// SEARCH LAYER
// =============================================================================

void HNSWIndex::search_layer(
    const scalar_t *query, idx_t entry_point, size_t ef, level_t level,
    std::vector<std::pair<score_t, idx_t>> &candidates) const {
  std::vector<bool> visited(nodes_.size(), false);

  // Priority queues for candidates and results
  // For similarity metrics: max-heap for candidates, min-heap for results
  auto cmp_candidates = [this](const std::pair<score_t, idx_t> &a,
                               const std::pair<score_t, idx_t> &b) {
    return is_similarity_metric_ ? a.first < b.first : a.first > b.first;
  };
  auto cmp_results = [this](const std::pair<score_t, idx_t> &a,
                            const std::pair<score_t, idx_t> &b) {
    return is_similarity_metric_ ? a.first > b.first : a.first < b.first;
  };

  std::priority_queue<std::pair<score_t, idx_t>,
                      std::vector<std::pair<score_t, idx_t>>,
                      decltype(cmp_candidates)>
      candidate_queue(cmp_candidates);
  std::priority_queue<std::pair<score_t, idx_t>,
                      std::vector<std::pair<score_t, idx_t>>,
                      decltype(cmp_results)>
      result_queue(cmp_results);

  // Initialize with entry point
  score_t entry_dist = distance(query, get_vector_by_idx(entry_point));
  visited[entry_point] = true;
  candidate_queue.push({entry_dist, entry_point});
  result_queue.push({entry_dist, entry_point});

  while (!candidate_queue.empty()) {
    auto [current_dist, current_idx] = candidate_queue.top();
    candidate_queue.pop();

    // Check stopping condition
    score_t worst_result = result_queue.top().first;
    bool should_stop = is_similarity_metric_ ? current_dist < worst_result
                                             : current_dist > worst_result;
    if (result_queue.size() >= ef && should_stop) {
      break;
    }

    // Explore neighbors
    const auto &node = nodes_[current_idx];
    if (level <= node.level && neighbors_[current_idx]) {
      const idx_t *nbrs = neighbors_[current_idx]->neighbors(level);
      size_t n_nbrs = neighbors_[current_idx]->count(level);

      // Prefetch neighbor vectors
      for (size_t i = 0; i < std::min(n_nbrs, size_t(4)); ++i) {
        if (nbrs[i] < nodes_.size()) {
          simd::prefetch_vector(get_vector_by_idx(nbrs[i]), config_.dimension);
        }
      }

      for (size_t i = 0; i < n_nbrs; ++i) {
        idx_t nbr_idx = nbrs[i];
        if (nbr_idx >= nodes_.size() || visited[nbr_idx])
          continue;

        visited[nbr_idx] = true;

        // Prefetch next
        if (i + 4 < n_nbrs && nbrs[i + 4] < nodes_.size()) {
          simd::prefetch_vector(get_vector_by_idx(nbrs[i + 4]),
                                config_.dimension);
        }

        score_t nbr_dist = distance(query, get_vector_by_idx(nbr_idx));

        // Check if should add
        bool should_add = result_queue.size() < ef;
        if (!should_add) {
          worst_result = result_queue.top().first;
          should_add = is_similarity_metric_ ? nbr_dist > worst_result
                                             : nbr_dist < worst_result;
        }

        if (should_add) {
          candidate_queue.push({nbr_dist, nbr_idx});
          result_queue.push({nbr_dist, nbr_idx});

          if (result_queue.size() > ef) {
            result_queue.pop();
          }
        }
      }
    }
  }

  // Extract results
  candidates.clear();
  candidates.reserve(result_queue.size());
  while (!result_queue.empty()) {
    candidates.push_back(result_queue.top());
    result_queue.pop();
  }

  // Sort by score (best first)
  std::sort(candidates.begin(), candidates.end(),
            [this](const auto &a, const auto &b) {
              return is_similarity_metric_ ? a.first > b.first
                                           : a.first < b.first;
            });
}

// =============================================================================
// SELECT NEIGHBORS
// =============================================================================

std::vector<idx_t> HNSWIndex::select_neighbors(
    const scalar_t *query,
    const std::vector<std::pair<score_t, idx_t>> &candidates, size_t M) const {
  if (candidates.size() <= M) {
    std::vector<idx_t> result;
    result.reserve(candidates.size());
    for (const auto &[score, idx] : candidates) {
      result.push_back(idx);
    }
    return result;
  }

  // Simple heuristic: take top M by score
  std::vector<idx_t> result;
  result.reserve(M);
  for (size_t i = 0; i < M && i < candidates.size(); ++i) {
    result.push_back(candidates[i].second);
  }
  return result;
}

// =============================================================================
// CONNECT NODES
// =============================================================================

void HNSWIndex::connect(idx_t node_idx, const std::vector<idx_t> &neighbors,
                        level_t level) {
  // Connect node to neighbors
  for (idx_t nbr_idx : neighbors) {
    neighbors_[node_idx]->add(level, nbr_idx);

    // Bidirectional connection
    if (level <= nodes_[nbr_idx].level) {
      neighbors_[nbr_idx]->add(level, node_idx);

      // Prune if over capacity
      size_t M = (level == 0) ? config_.M_max : config_.M;
      if (neighbors_[nbr_idx]->count(level) > M) {
        // Simple pruning: keep M closest
        const scalar_t *nbr_vec = get_vector_by_idx(nbr_idx);
        std::vector<std::pair<score_t, idx_t>> nbr_neighbors;

        const idx_t *nbrs = neighbors_[nbr_idx]->neighbors(level);
        size_t n = neighbors_[nbr_idx]->count(level);

        for (size_t i = 0; i < n; ++i) {
          score_t d = distance(nbr_vec, get_vector_by_idx(nbrs[i]));
          nbr_neighbors.push_back({d, nbrs[i]});
        }

        std::sort(nbr_neighbors.begin(), nbr_neighbors.end(),
                  [this](const auto &a, const auto &b) {
                    return is_similarity_metric_ ? a.first > b.first
                                                 : a.first < b.first;
                  });

        neighbors_[nbr_idx]->set_count(level, 0);
        for (size_t i = 0; i < M && i < nbr_neighbors.size(); ++i) {
          neighbors_[nbr_idx]->add(level, nbr_neighbors[i].second);
        }
      }
    }
  }
}

// =============================================================================
// SEARCH
// =============================================================================

std::vector<SearchResult> HNSWIndex::search(const scalar_t *query, size_t k,
                                            size_t ef) const {
  std::shared_lock lock(rw_mutex_);

  if (count_.load() == 0) {
    return {};
  }

  if (ef == 0)
    ef = config_.ef_search;
  ef = std::max(ef, k);

  // Prepare query (normalize if cosine)
  aligned_vector<scalar_t> query_normalized;
  const scalar_t *query_ptr = query;

  if (config_.metric == MetricType::COSINE) {
    query_normalized.resize(config_.dimension);
    simd::normalize(query, query_normalized.data(), config_.dimension);
    query_ptr = query_normalized.data();
  }

  idx_t current_entry = entry_point_.load(std::memory_order_acquire);
  level_t current_max_level = max_level_.load(std::memory_order_acquire);
  std::vector<std::pair<score_t, idx_t>> candidates;

  // Traverse from top to layer 1
  for (level_t l = current_max_level; l > 0; --l) {
    candidates.clear();
    search_layer(query_ptr, current_entry, 1, l, candidates);
    if (!candidates.empty()) {
      current_entry = candidates[0].second;
    }
  }

  // Search layer 0 with full ef
  candidates.clear();
  search_layer(query_ptr, current_entry, ef, 0, candidates);

  // Build results
  std::vector<SearchResult> results;
  results.reserve(std::min(k, candidates.size()));

  for (size_t i = 0; i < k && i < candidates.size(); ++i) {
    const auto &[score, idx] = candidates[i];
    const auto &node = nodes_[idx];
    if (!node.deleted) {
      results.push_back({node.id, score, idx});
    }
  }

  return results;
}

std::vector<std::vector<SearchResult>>
HNSWIndex::search_batch(const scalar_t *queries, size_t n_queries, size_t k,
                        size_t ef) const {
  std::vector<std::vector<SearchResult>> results(n_queries);

  // TODO: Parallel search with thread pool
  for (size_t i = 0; i < n_queries; ++i) {
    results[i] = search(queries + i * config_.dimension, k, ef);
  }

  return results;
}

// =============================================================================
// DELETE
// =============================================================================

bool HNSWIndex::remove(vid_t id) {
  std::unique_lock lock(rw_mutex_);

  auto it = id_to_idx_.find(id);
  if (it == id_to_idx_.end()) {
    return false;
  }

  nodes_[it->second].deleted = 1;
  id_to_idx_.erase(it);
  return true;
}

// =============================================================================
// RETRIEVAL
// =============================================================================

const scalar_t *HNSWIndex::get_vector(vid_t id) const {
  std::shared_lock lock(rw_mutex_);

  auto it = id_to_idx_.find(id);
  if (it == id_to_idx_.end()) {
    return nullptr;
  }

  return get_vector_by_idx(it->second);
}

bool HNSWIndex::contains(vid_t id) const {
  std::shared_lock lock(rw_mutex_);
  return id_to_idx_.count(id) > 0;
}

// =============================================================================
// STATISTICS
// =============================================================================

HNSWIndex::Stats HNSWIndex::get_stats() const {
  std::shared_lock lock(rw_mutex_);

  Stats stats{};
  stats.total_vectors = nodes_.size();
  stats.deleted_vectors = 0;
  stats.max_level = max_level_.load();

  size_t total_connections = 0;
  for (const auto &node : nodes_) {
    if (node.deleted) {
      ++stats.deleted_vectors;
    }
  }

  for (const auto &nbr_list : neighbors_) {
    if (nbr_list) {
      for (level_t l = 0; l <= max_level_.load(); ++l) {
        total_connections += nbr_list->count(l);
      }
    }
  }

  stats.avg_connections =
      stats.total_vectors > 0
          ? static_cast<double>(total_connections) / stats.total_vectors
          : 0.0;

  // Memory estimate
  stats.memory_bytes = vectors_.size() * sizeof(scalar_t) +
                       nodes_.size() * sizeof(HNSWNode) +
                       total_connections * sizeof(idx_t);

  return stats;
}

// =============================================================================
// PERSISTENCE (TODO: Full implementation)
// =============================================================================

Result<void> HNSWIndex::save(const std::string &path) const {
  // TODO: Implement binary serialization
  return Result<void>::Err(Error("Not implemented"));
}

Result<HNSWIndex> HNSWIndex::load(const std::string &path) {
  // TODO: Implement binary deserialization
  return Result<HNSWIndex>::Err(Error("Not implemented"));
}

} // namespace vectorsearch
