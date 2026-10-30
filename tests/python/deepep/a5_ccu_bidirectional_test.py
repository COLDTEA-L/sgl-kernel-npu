"""Same C++ host, FP8 dispatch then BF16 combine; device correctness, not timing."""
from a5_ccu_payload_test_utils import (
    make_payload, cat_payload, full_payload, copy_payload_, fill_payload_,
    equal_payload, guards_intact, fp8_dtypes,
)


def check_bidirectional(host, cases, rank, dtype, width, iterations, native_alltoallv):
    import torch
    if dtype not in fp8_dtypes():
        raise ValueError('--bidirectional-iters requires an FP8 --dtype')
    guard = 37
    reports = []
    for name, counts in cases:
        reverse = tuple(counts[d * 8 + s] for s in range(8) for d in range(8))
        directions = []
        for label, matrix, kind in (('dispatch', counts, dtype), ('combine', reverse, torch.bfloat16)):
            sends, recvs = list(matrix[rank * 8:rank * 8 + 8]), list(matrix[rank::8])
            input_store = full_payload((sum(sends) * width + 2 * guard,), -19, kind, 'npu')
            output_store = full_payload((sum(recvs) * width + 2 * guard,), -17, kind, 'npu')
            send = input_store[guard:-guard].reshape(sum(sends), width)
            output = output_store[guard:-guard].reshape(sum(recvs), width)
            prepared = host.prepare(matrix, send)
            expected_row_bytes = width * torch.empty((), dtype=kind).element_size()
            if prepared.row_bytes != expected_row_bytes:
                raise AssertionError('prepared row_bytes differs from tensor signature')
            directions.append(dict(label=label, matrix=matrix, dtype=kind, sends=sends, recvs=recvs,
                                   input_store=input_store, output_store=output_store, send=send,
                                   output=output, prepared=prepared))
        warm = host.stats()
        for epoch in range(iterations + 1):
            expected = []
            for d in directions:
                matrix, kind = d['matrix'], d['dtype']
                src = make_payload(sum(d['sends']), width, rank, kind, epoch)
                oracle = cat_payload([
                    make_payload(sum(matrix[s * 8:s * 8 + 8]), width, s, kind, epoch)[
                        sum(matrix[s * 8:s * 8 + rank]):sum(matrix[s * 8:s * 8 + rank]) + matrix[s * 8 + rank]]
                    for s in range(8)])
                copy_payload_(d['send'], src)
                fill_payload_(d['output_store'], -17)
                expected.append(oracle)
            # No device synchronize between directions. Shared lane events must
            # order the reverse payload after the dispatch join on lane 0.
            for d in directions:
                if epoch == 0:
                    host.execute(d['prepared'], d['send'], d['output'])
                else:
                    host.run(d['matrix'], d['send'], d['output'])
            torch.npu.synchronize()
            for d, oracle in zip(directions, expected):
                if not equal_payload(d['output'], oracle):
                    raise AssertionError(f'{name} {d["label"]} epoch={epoch}: payload mismatch')
                if not guards_intact(d['input_store'], guard, -19) or not guards_intact(d['output_store'], guard, -17):
                    raise AssertionError(f'{name} {d["label"]}: guard corruption')
                if epoch == 0:
                    baseline = full_payload(d['output'].shape, -17, d['dtype'], 'npu')
                    native_alltoallv(d['send'], baseline, d['sends'], d['recvs'])
                    if not equal_payload(baseline, oracle):
                        raise AssertionError(f'{name} {d["label"]}: native baseline mismatch')
        after = host.stats()
        for key in ('tensor_allocations', 'event_allocations', 'reserved_bytes', 'zero_backing_bytes'):
            if after[key] != warm[key]:
                raise AssertionError(f'{name}: steady-state allocation changed: {key}')
        reports.append(dict(case=name, iterations=iterations, correctness='PASS',
                            dispatch_row_bytes=directions[0]['prepared'].row_bytes,
                            combine_row_bytes=directions[1]['prepared'].row_bytes,
                            dispatch_chunks=directions[0]['prepared'].chunks,
                            combine_chunks=directions[1]['prepared'].chunks,
                            payloads=host.payload_stats(), workspace=after))
    return reports
