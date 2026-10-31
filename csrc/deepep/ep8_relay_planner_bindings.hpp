#pragma once
#include "ep8_relay_planner.hpp"
#include "ep8_fast_selector.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace deep_ep::ep8 {
// Return immutable tuples, avoiding a second list-to-tuple conversion in Python.
template <class Sequence> pybind11::tuple Scalars(const Sequence& values) {
    pybind11::tuple out(values.size());
    size_t i = 0;
    for (auto value : values) out[i++] = pybind11::int_(value);
    return out;
}
template <class Sequence> pybind11::tuple Rows(const Sequence& values) {
    pybind11::tuple out(values.size());
    size_t i = 0;
    for (const auto& value : values) out[i++] = Scalars(value);
    return out;
}
// Strict host integers; do not implicitly synchronize or accept device tensors.
inline Matrix HostCounts(const pybind11::object& obj) {
    namespace py = pybind11;
    if (!py::isinstance<py::tuple>(obj) && !py::isinstance<py::list>(obj))
        throw py::value_error("counts must be a host list/tuple");
    auto values = py::reinterpret_borrow<py::sequence>(obj);
    Matrix result{};
    if (values.size() != 64 && values.size() != 8) throw py::value_error("counts must be flat64 or 8x8");
    for (size_t i = 0; i < 64; ++i) {
        py::object value;
        if (values.size() == 64) value = values[i];
        else {
            auto row = values[i / 8];
            if ((!py::isinstance<py::tuple>(row) && !py::isinstance<py::list>(row)) || py::len(row) != 8)
                throw py::value_error("counts must be 8x8");
            value = py::reinterpret_borrow<py::sequence>(row)[i % 8];
        }
        if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr())) throw py::value_error("count must be an integer");
        try { result[i] = value.cast<U>(); }
        catch (const py::cast_error&) { throw py::value_error("count out of uint64 range"); }
    }
    return result;
}
inline void BindPlanner(pybind11::module_& m) {
    namespace py = pybind11;
    m.attr("ep8_relay_planner_abi") = 2;
    m.attr("ep8_dense_ready_planner_abi") = 2;
    m.attr("ep8_fast_selector_abi") = 1;
    py::class_<FastDecision>(m, "Ep8FastDecision")
        .def_property_readonly("uses_relay", &FastDecision::UsesRelay)
        .def_readonly("reason", &FastDecision::reason)
        .def_readonly("flow", &FastDecision::flow)
        .def_readonly("relay", &FastDecision::relay)
        .def_readonly("rows", &FastDecision::rows)
        .def_readonly("chunk_bytes", &FastDecision::chunkBytes)
        .def_readonly("candidate_evaluations", &FastDecision::evaluations)
        .def_readonly("predicted_direct_us", &FastDecision::directUs)
        .def_readonly("predicted_relay_us", &FastDecision::relayUs)
        .def_readonly("predicted_saving_us", &FastDecision::savingUs);
    py::class_<FastSelector>(m, "Ep8FastSelector")
        .def(py::init<const std::vector<std::vector<U>>&, U, const std::vector<U>&, U, double, double, double>())
        .def("select", [](const FastSelector& s, const py::object& c) { return s.Select(HostCounts(c)); })
        .def("materialize", [](const FastSelector& s, const FastDecision& d) {
            auto p = s.Materialize(d);
            return py::make_tuple(Scalars(p.counts), Scalars(p.relays), Scalars(p.moved),
                                 Scalars(p.loads), Rows(p.actions), p.evaluations, d.reason);
        });
    py::class_<CopySpans>(m, "Ep8CopySpans")
        .def(py::init<std::vector<Span>>())
        .def("__len__", [](const CopySpans& c) { return c.spans.size(); })
        .def("__iter__", [](const CopySpans& c) {
            return py::make_iterator(c.spans.begin(), c.spans.end());
        }, py::keep_alive<0, 1>());
    py::class_<Planner>(m, "Ep8RelayPlanner")
        .def(py::init<const std::vector<std::vector<U>>&, U, U, U, U, const std::vector<U>&>())
        .def("with_dense_ready", &Planner::WithDenseReady)
        .def("plan", [](const Planner& planner, const Matrix& counts) {
            auto p = planner.Make(counts);
            return py::make_tuple(Scalars(p.counts), Scalars(p.relays), Scalars(p.moved),
                                 Scalars(p.loads), Rows(p.actions), p.evaluations,
                                 planner.DenseDivisor() ? (p.exhausted ? "no_volume_candidate_within_peak" : "volume_excess_selected") :
                                     (p.exhausted ? "no_profitable_candidate" : "action_limit"));
        })
        .def("layout", [](const Planner& planner, const Matrix& counts, const Relays& relays,
                          const Matrix& moved, U chunkBytes, U maxChunks, int rank) {
            auto p = planner.Build(counts, relays, moved, chunkBytes, maxChunks, rank);
            py::tuple chunks(p.chunks.size());
            for (size_t i = 0; i < p.chunks.size(); ++i) {
                auto& c = p.chunks[i];
                py::object copies = py::none();
                if (rank >= 0) copies = py::make_tuple(std::move(c.localCopies[0]),
                                                     std::move(c.localCopies[1]), std::move(c.localCopies[2]));
                chunks[i] = py::make_tuple(Scalars(c.first), Scalars(c.second), Rows(c.pieces), Rows(c.sizes), copies);
            }
            return py::make_tuple(p.chunkRows, Scalars(p.direct), Rows(p.directPieces), chunks, Rows(p.pools));
        }, py::arg("counts"), py::arg("relays"), py::arg("moved"), py::arg("chunk_bytes"),
           py::arg("max_chunks"), py::arg("rank") = -1);
}
} // namespace deep_ep::ep8
