/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder_init.h"
#include "sparse_dem_from_stim_text.h"
#include "cudaq/qec/extended_dem.h"
#include <stdexcept>
#include <utility>

namespace cudaq::qec {

struct decoder_init::impl {
  decoder_model_source source = decoder_model_source::matrices;
  std::size_t num_detectors = 0;
  std::size_t num_error_mechanisms = 0;
  std::size_t num_observables = 0;
  sparse_binary_matrix H;
  /// Absent when the model supplies no observable mapping. A present but
  /// zero-row O is a supplied model, not an absent one.
  std::optional<sparse_binary_matrix> O;
  std::vector<double> rates;
  std::optional<std::vector<std::size_t>> ids;
  std::optional<sparse_binary_matrix> D;
  std::optional<std::string> raw_stim_dem;
  std::optional<dem_chunks_spec> chunk_spec;
  /// The expansion the projection was built from. Shared because the state is
  /// immutable, so an impl copy need not duplicate every round's matrices.
  std::shared_ptr<const std::vector<extended_dem>> chunks;
};

namespace {

void validate_model(const sparse_binary_matrix &H,
                    const std::optional<sparse_binary_matrix> &O,
                    const std::vector<double> &rates,
                    const std::optional<std::vector<std::size_t>> &ids,
                    const std::optional<sparse_binary_matrix> &D) {
  if (O && O->num_cols() != H.num_cols())
    throw std::invalid_argument(
        "decoder_init: O column count must match H column count");
  if (!rates.empty() && rates.size() != H.num_cols())
    throw std::invalid_argument(
        "decoder_init: error_rates size must match H column count");
  if (ids && ids->size() != H.num_cols())
    throw std::invalid_argument(
        "decoder_init: error_ids size must match H column count");
  if (D && D->num_rows() != H.num_rows())
    throw std::invalid_argument(
        "decoder_init: D row count must match H row count");
}

/// A repeating phase emits one copy per round, so a streaming spec has no
/// definite sequence. Rejected rather than deferred: without an expansion there
/// is no matrix projection, and the decoder base sizes its buffers from that.
/// Checked ahead of phase_sequence(), which would fail less clearly.
void require_resolvable_round_count(const dem_chunks_spec &spec) {
  if (spec.has_repeating_phase() && !spec.num_rounds.has_value())
    throw std::invalid_argument(
        "decoder_init: cannot expand dem_chunks with a repeating phase and no "
        "num_rounds; the round count has to be known to build a decoder");
}

} // namespace

std::shared_ptr<decoder_init::impl> decoder_init::make_matrix_state(
    decoder_model_source source, sparse_binary_matrix H,
    std::optional<sparse_binary_matrix> O, std::vector<double> rates,
    std::optional<std::vector<std::size_t>> ids,
    std::optional<sparse_binary_matrix> D,
    std::optional<std::string> raw_stim_dem) {
  H = H.to_csc();
  if (O)
    *O = O->to_csr();
  if (D)
    *D = D->to_csr();
  validate_model(H, O, rates, ids, D);

  auto state = std::make_shared<decoder_init::impl>();
  state->source = source;
  state->num_detectors = H.num_rows();
  state->num_error_mechanisms = H.num_cols();
  state->num_observables = O ? O->num_rows() : 0;
  state->H = std::move(H);
  state->O = std::move(O);
  state->rates = std::move(rates);
  state->ids = std::move(ids);
  state->D = std::move(D);
  state->raw_stim_dem = std::move(raw_stim_dem);
  return state;
}

std::shared_ptr<decoder_init::impl> decoder_init::make_chunk_state(
    dem_chunks_spec spec, std::vector<extended_dem> chunks,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  if (chunks.empty())
    throw std::invalid_argument("decoder_init: dem_chunks expanded to no "
                                "chunks, so it describes no model");

  // A supplied expansion has to be the one the spec describes, or the retained
  // spec would name a different model than the projection. Comparing lengths
  // catches the likely misuse: an expansion of a different round count.
  // For a streaming spec (repeating phase, no num_rounds) the round count is
  // open-ended by design; phase_sequence() would throw, so the check is skipped
  // and the caller-supplied chunk count is accepted as-is.
  if (!spec.has_repeating_phase() || spec.num_rounds.has_value()) {
    const auto sequence = spec.phase_sequence();
    if (sequence.size() != chunks.size())
      throw std::invalid_argument(
          "decoder_init: dem_chunks spec describes " +
          std::to_string(sequence.size()) + " chunks but " +
          std::to_string(chunks.size()) + " were supplied");
  } // end - if(spec resolvable)

  // Project straight to sparse, as from_stim_dem() does: closing the chunks
  // would allocate a dense detector x fault tensor for a model whose sparse
  // form is a small fraction of it.
  auto H = dem_chunks_to_pcm(chunks, spec.seam.from_seam, spec.seam.to_seam);
  const auto observable_rows = dem_chunks_to_o_sparse(chunks);
  // Phases with no observable rows supply no mapping, like an H-only matrix
  // input. A synthesized zero-row O would report one, letting a decoder built
  // for observable output construct and then return an empty frame.
  std::optional<sparse_binary_matrix> O;
  if (!observable_rows.empty())
    O = sparse_binary_matrix::from_nested_csr(
        static_cast<sparse_binary_matrix::index_type>(observable_rows.size()),
        H.num_cols(), observable_rows);

  // Both projections lay fault columns out in chunk order, so the priors
  // concatenate in the same basis.
  std::size_t num_rates = 0;
  for (const auto &chunk : chunks)
    num_rates += chunk.error_rates.size();
  std::vector<double> error_rates;
  error_rates.reserve(num_rates);
  for (const auto &chunk : chunks)
    error_rates.insert(error_rates.end(), chunk.error_rates.begin(),
                       chunk.error_rates.end());

  auto state =
      make_matrix_state(decoder_model_source::dem_chunks, std::move(H),
                        std::move(O), std::move(error_rates), std::nullopt,
                        std::move(measurement_to_detectors));
  state->chunk_spec = std::move(spec);
  state->chunks =
      std::make_shared<const std::vector<extended_dem>>(std::move(chunks));
  return state;
}

decoder_init::decoder_init(sparse_binary_matrix H)
    : decoder_init(make_matrix_state(decoder_model_source::matrices,
                                     std::move(H), std::nullopt, {},
                                     std::nullopt, std::nullopt)) {}

decoder_init::decoder_init(
    sparse_binary_matrix H, std::optional<sparse_binary_matrix> O,
    std::vector<double> error_rates,
    std::optional<sparse_binary_matrix> measurement_to_detectors,
    std::optional<std::vector<std::size_t>> error_ids)
    : decoder_init(make_matrix_state(
          decoder_model_source::matrices, std::move(H), std::move(O),
          std::move(error_rates), std::move(error_ids),
          std::move(measurement_to_detectors))) {}

decoder_init::decoder_init(
    detector_error_model model,
    std::optional<sparse_binary_matrix> measurement_to_detectors)
    : decoder_init(make_matrix_state(
          decoder_model_source::matrices,
          sparse_binary_matrix(model.detector_error_matrix),
          sparse_binary_matrix(model.observables_flips_matrix),
          std::move(model.error_rates), std::move(model.error_ids),
          std::move(measurement_to_detectors))) {}

decoder_init decoder_init::from_stim_dem(
    std::string stim_dem_text,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  // Project straight to sparse. Going through the materialized
  // detector_error_model would allocate a dense detectors x mechanisms tensor
  // only to scan it back out again: ~98 MiB for a distance-13 model whose
  // sparse form is under 1 MiB, and wasted entirely for a DEM-native decoder.
  auto [H, O, error_rates] = details::sparse_dem_from_stim_text(stim_dem_text);
  return decoder_init(make_matrix_state(
      decoder_model_source::stim_dem, std::move(H), std::move(O),
      std::move(error_rates), std::nullopt, std::move(measurement_to_detectors),
      std::move(stim_dem_text)));
}

decoder_init decoder_init::from_dem_chunks(
    dem_chunks_spec spec,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  require_resolvable_round_count(spec);
  auto chunks = dem_chunks_from_spec(spec);
  return from_dem_chunks(std::move(spec), std::move(chunks),
                         std::move(measurement_to_detectors));
}

decoder_init decoder_init::from_dem_chunks(
    dem_chunks_spec spec, std::vector<extended_dem> chunks,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  return decoder_init(make_chunk_state(std::move(spec), std::move(chunks),
                                       std::move(measurement_to_detectors)));
}

decoder_init::decoder_init(std::shared_ptr<const impl> state)
    : state_(std::move(state)) {}

decoder_init::decoder_init(const decoder_init &) noexcept = default;
decoder_init::decoder_init(decoder_init &&) noexcept = default;
decoder_init &decoder_init::operator=(const decoder_init &) noexcept = default;
decoder_init &decoder_init::operator=(decoder_init &&) noexcept = default;
decoder_init::~decoder_init() = default;

decoder_model_source decoder_init::source() const noexcept {
  return state_->source;
}

const sparse_binary_matrix &decoder_init::detector_error_matrix() const {
  return state_->H;
}

bool decoder_init::has_observable_model() const noexcept {
  return state_->O.has_value();
}

const sparse_binary_matrix &decoder_init::observable_flips_matrix() const {
  if (!state_->O)
    throw std::logic_error("decoder_init: no observable mapping was supplied");
  return *state_->O;
}

const std::vector<double> &decoder_init::error_rates() const {
  return state_->rates;
}

const std::optional<std::vector<std::size_t>> &decoder_init::error_ids() const {
  return state_->ids;
}

const sparse_binary_matrix *
decoder_init::measurement_to_detectors() const noexcept {
  return state_->D ? &*state_->D : nullptr;
}

decoder_init decoder_init::canonicalize_H() const {
  // Copy and replace H rather than rebuild: canonicalization leaves every
  // dimension and source-specific field untouched, so a later source cannot
  // be dropped here by omission.
  auto state = std::make_shared<impl>(*state_);
  state->H = state_->H.canonicalize().to_csc();
  return decoder_init(std::move(state));
}

decoder_init decoder_init::decoder_init_without_d() const {
  auto state = std::make_shared<impl>(*state_);
  state->D.reset();
  return decoder_init(std::move(state));
}

bool decoder_init::has_stim_dem() const noexcept {
  return state_->raw_stim_dem.has_value();
}

const std::string &decoder_init::stim_dem() const {
  if (!state_->raw_stim_dem)
    throw std::logic_error(
        "decoder_init: authoritative source is not a Stim DEM");
  return *state_->raw_stim_dem;
}

bool decoder_init::has_dem_chunks() const noexcept {
  return state_->chunk_spec.has_value();
}

const dem_chunks_spec &decoder_init::dem_chunks() const {
  if (!state_->chunk_spec)
    throw std::logic_error(
        "decoder_init: authoritative source is not a chunked DEM");
  return *state_->chunk_spec;
}

const std::vector<extended_dem> *
decoder_init::dem_chunk_sequence() const noexcept {
  return state_->chunks.get();
}

std::size_t decoder_init::num_detectors() const noexcept {
  return state_->num_detectors;
}

std::size_t decoder_init::num_error_mechanisms() const noexcept {
  return state_->num_error_mechanisms;
}

std::size_t decoder_init::num_observables() const noexcept {
  return state_->num_observables;
}

} // namespace cudaq::qec
