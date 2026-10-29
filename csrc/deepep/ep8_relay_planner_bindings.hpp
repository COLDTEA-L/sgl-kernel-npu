#pragma once
#include "ep8_relay_planner.hpp"
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
inline void BindPlanner(pybind11::module_& m) {
    namespace py = pybind11;
    m.attr("ep8_relay_planner_abi") = 2;
    py::class_<CopySpans>(m, "Ep8CopySpans")
        .def(py::init<std::vector<Span>>())
        .def("__len__", [](const CopySpans& c) { return c.spans.size(); })
        .def("__iter__", [](const CopySpans& c) {
            return py::make_iterator(c.spans.begin(), c.spans.end());
        }, py::keep_alive<0, 1>());
    py::class_<Planner>(m, "Ep8RelayPlanner")
        .def(py::init<const std::vector<std::vector<U>>&, U, U, U, U, const std::vector<U>&>())
        .def("plan", [](const Planner& planner, const Matrix& counts) {
            auto p = planner.Make(counts);
            return py::make_tuple(Scalars(p.counts), Scalars(p.relays), Scalars(p.moved),
                                 Scalars(p.loads), Rows(p.actions), p.evaluations,
                                 p.exhausted ? "no_profitable_candidate" : "action_limit");
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
