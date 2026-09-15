#!/usr/bin/env python3
"""Fail-closed capability gate for explicit A5 IO Die relay routing."""

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path


CAPABILITIES = {
    "install_source_route": 1 << 0,
    "query_source_route": 1 << 1,
    "remove_source_route": 1 << 2,
    "io_die_forwarding": 1 << 3,
    "no_relay_hbm": 1 << 4,
}
REQUIRED_MASK = sum(CAPABILITIES.values())
REQUIRED_SYMBOLS = (
    "A5UvsBackendGetCapabilities",
    "A5UvsBackendInstallRoute",
    "A5UvsBackendQueryRoute",
    "A5UvsBackendRemoveRoute",
)


class BackendCaps(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("capability_flags", ctypes.c_uint32),
        ("max_routes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def inspect_umdk_source(root: Path) -> dict:
    header = root / "src/urma/lib/uvs/core/include/uvs_api.h"
    result = {
        "root": str(root),
        "header": str(header),
        "found": header.is_file(),
        "has_get_route_list": False,
        "has_route_mutation_api": False,
        "direct_only_comment": False,
    }
    if not header.is_file():
        return result
    text = header.read_text(errors="replace")
    result["has_get_route_list"] = "uvs_get_route_list" in text
    result["has_route_mutation_api"] = any(
        name in text
        for name in (
            "uvs_add_route",
            "uvs_create_route",
            "uvs_install_route",
            "uvs_delete_route",
            "uvs_remove_route",
        )
    )
    result["direct_only_comment"] = "Only supports direct routes" in text
    return result


def inspect_backend(path: str) -> dict:
    result = {
        "path": path,
        "loaded": False,
        "symbols": {},
        "abi_version": 0,
        "capability_flags": 0,
        "max_routes": 0,
        "required_mask": REQUIRED_MASK,
        "usable": False,
        "error": "",
    }
    if not path:
        result["error"] = "A5_UVS_EXPLICIT_ROUTE_BACKEND is not set"
        return result
    try:
        library = ctypes.CDLL(path)
        result["loaded"] = True
        for symbol in REQUIRED_SYMBOLS:
            result["symbols"][symbol] = hasattr(library, symbol)
        if not all(result["symbols"].values()):
            result["error"] = "backend is missing required ABI symbols"
            return result
        caps_fn = library.A5UvsBackendGetCapabilities
        caps_fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(BackendCaps)]
        caps_fn.restype = ctypes.c_int
        caps = BackendCaps()
        status = caps_fn(1, ctypes.byref(caps))
        result["abi_version"] = caps.abi_version
        result["capability_flags"] = caps.capability_flags
        result["max_routes"] = caps.max_routes
        result["capabilities"] = {
            name: bool(caps.capability_flags & flag)
            for name, flag in CAPABILITIES.items()
        }
        result["usable"] = (
            status == 0
            and caps.abi_version == 1
            and (caps.capability_flags & REQUIRED_MASK) == REQUIRED_MASK
            and caps.max_routes > 0
        )
        if not result["usable"]:
            result["error"] = f"capability call/status mismatch: status={status}"
    except OSError as exc:
        result["error"] = str(exc)
    return result


def inspect_runtime_uvs() -> dict:
    candidates = []
    for name in ("uvs", "urma"):
        path = ctypes.util.find_library(name)
        if path:
            candidates.append(path)
    exported = {}
    for path in candidates:
        try:
            library = ctypes.CDLL(path)
            exported[path] = {
                "uvs_get_route_list": hasattr(library, "uvs_get_route_list"),
                "mutation_symbols": [
                    symbol
                    for symbol in (
                        "uvs_add_route",
                        "uvs_install_route",
                        "uvs_remove_route",
                        "uvs_delete_route",
                    )
                    if hasattr(library, symbol)
                ],
            }
        except OSError as exc:
            exported[path] = {"error": str(exc)}
    return {"candidates": candidates, "exports": exported}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umdk-root", default=os.environ.get("UMDK_REPO", "/home/liuyuanwen/umdk")
    )
    parser.add_argument(
        "--backend", default=os.environ.get("A5_UVS_EXPLICIT_ROUTE_BACKEND", "")
    )
    parser.add_argument("--output", help="also write the JSON report to this file")
    parser.add_argument(
        "--allow-missing-backend",
        action="store_true",
        help="report capability without failing; intended for the no-card build host",
    )
    args = parser.parse_args()

    report = {
        "umdk_source": inspect_umdk_source(Path(args.umdk_root)),
        "runtime_uvs": inspect_runtime_uvs(),
        "explicit_route_backend": inspect_backend(args.backend),
    }
    report["result"] = (
        "PASS_EXPLICIT_RELAY_BACKEND"
        if report["explicit_route_backend"]["usable"]
        else "BLOCKED_NO_ROUTE_PROGRAMMING_BACKEND"
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    if not report["explicit_route_backend"]["usable"] and not args.allow_missing_backend:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
