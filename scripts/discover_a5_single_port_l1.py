#!/usr/bin/env python3
"""Collect device-local EIDs, join L1 rootinfo membership, produce editable policy."""
import argparse
import ctypes
import itertools
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from a5_ccu_l1_policy import sha, memberships, network_fes, single_port, validate_config
from a5_ccu_machine_topology import normalize_eid, port_list
from prepare_a5_ccu_peer_plan import parse_phys_ids


def driver_inventory(devices):
    # ABI matches HCCL topo_addr_info/hal.h (dcmi_urma_eid_info_t), LP64.
    class Eid(ctypes.Union):
        _fields_=[('raw',ctypes.c_ubyte*16),('words',ctypes.c_ulong*2)]
    class Entry(ctypes.Structure):
        _fields_=[('eid',Eid),('eid_index',ctypes.c_uint)]
    if ctypes.sizeof(Entry)!=24: raise RuntimeError('DCMI inventory requires LP64 ABI')
    lib=ctypes.CDLL('libdcmi.so')
    init=lib.dcmiv2_init;init.argtypes=[];init.restype=ctypes.c_int
    status=init()
    if status: raise RuntimeError(f'dcmiv2_init status={status}')
    mapping=getattr(lib,'dcmiv2_get_dev_id_by_chip_phy_id',None) or getattr(lib,'dcmiv2_get_dev_id_from_chip_phyid')
    mapping.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)];mapping.restype=ctypes.c_int
    num=lib.dcmiv2_get_urma_device_cnt
    num.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint)];num.restype=ctypes.c_int
    get=lib.dcmiv2_get_eid_list_by_urma_dev_index
    get.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(Entry),ctypes.POINTER(ctypes.c_int)];get.restype=ctypes.c_int
    result=[]
    for phy in devices:
        logic=ctypes.c_uint();count=ctypes.c_uint()
        if mapping(phy,ctypes.byref(logic)) or num(logic.value,ctypes.byref(count)) or not 0<count.value<=64:
            raise RuntimeError(f'DCMI cannot map/query physical device {phy}')
        for ue in range(count.value):
            entries=(Entry*32)();n=ctypes.c_int(32)
            status=get(logic.value,ue,entries,ctypes.byref(n))
            if status or not 0<=n.value<=32: raise RuntimeError(f'DCMI EID query phy={phy} ue={ue} status={status}')
            for e in entries[:n.value]:
                raw=bytes(e.eid.raw)
                if any(raw): result.append(dict(device_id=phy,protocol=4,addr_type=3,eid=raw.hex(),
                    source='dcmi_urma',dcmi_logic_id=logic.value,ue_index=ue,eid_index=e.eid_index))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices',required=True)
    p.add_argument('--root-info',type=Path,default=Path('/etc/hccl_rootinfo.json'))
    p.add_argument('--inventory-source',choices=('driver','sdk'),default='driver')
    p.add_argument('--inventory-binary',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    devices=parse_phys_ids(args.devices,'devices')
    if len(devices) not in (2,4,8): p.error('requires 2/4/8 devices')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    root=json.loads(args.root_info.read_text()); member=memberships(root); fes=network_fes(root)
    l0dies={d:set() for d in devices}
    for r in root['rank_list']:
        phy=int(r['device_id'])
        if phy not in l0dies: continue
        for level in r['level_list']:
            if int(level['net_layer'])!=0: continue
            for a in level['rank_addr_list']:
                if a['addr_type'].upper()=='EID':
                    # L0 rootinfo may also list network ports; final topology compiler
                    # checks the selected pair's actual direct die independently.
                    for die,_ in port_list(a['ports']): l0dies[phy].add(die)
    # Automatic selection is conservative. Explicit controller edits remain possible.
    eps=[]
    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=args.devices)
    if args.inventory_source=='driver':
        eps=driver_inventory(devices)
    else:
        if args.inventory_binary is None: p.error('--inventory-binary required for sdk source')
        for logical,phy in enumerate(devices):
            r=subprocess.run([str(args.inventory_binary.resolve()),str(logical),str(phy)],
                             env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
            (args.output_dir/f'inventory_{phy}.log').write_text(r.stdout)
            if r.returncode: raise RuntimeError(f'device {phy}: inventory rc={r.returncode}; inspect saved log')
            for line in r.stdout.splitlines():
                marker='L1_INVENTORY_ENDPOINT '
                if marker in line: eps.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0])
    inv=dict(schema='a5.endpoint-inventory.v1',rootinfo_sha256=sha(args.root_info),endpoints=eps)
    invpath=args.output_dir/'inventory.json';invpath.write_text(json.dumps(inv,indent=2)+'\n')
    candidates={d:[] for d in devices}
    for e in eps:
        try: die,port=single_port(e['eid'])
        except ValueError: continue
        phy=e['device_id']; m=member.get((phy,die,port))
        fe=bytes.fromhex(normalize_eid(e['eid']).replace(':',''))[6]&127
        if m is not None and fe in fes.get((phy,die,port),set()):
            candidates[phy].append(dict(device_id=phy,eid=normalize_eid(e['eid']),die=die,port=port,
                                        net_instance_id=m[0],plane_id=m[1]))
    (args.output_dir/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    # Determine direct die from rootinfo single-port L0 EIDs, excluding L1 ports.
    for d in devices: l0dies[d]=set()
    for r in root['rank_list']:
        phy=int(r['device_id'])
        if phy not in l0dies: continue
        for level in r['level_list']:
            if int(level['net_layer'])!=0: continue
            for a in level['rank_addr_list']:
                if a['addr_type'].upper()!='EID': continue
                for die,port in port_list(a['ports']):
                    if (phy,die,port) not in member: l0dies[phy].add(die)
    paths=[]
    for si,di in itertools.combinations(range(len(devices)),2):
        s,d=devices[si],devices[di]
        pairs=[(a,b) for a in candidates[s] for b in candidates[d]
               if (a['net_instance_id'],a['plane_id'])==(b['net_instance_id'],b['plane_id'])
               and l0dies[s]=={a['die']} and l0dies[d]=={b['die']}]
        pairs.sort(key=lambda ab:(ab[0]['port'],ab[1]['port'],ab[0]['eid'],ab[1]['eid']))
        if not pairs: continue
        # Distribute peers over available local ports; never infer new EIDs.
        a,b=pairs[(si+di)%len(pairs)]
        paths.append(dict(id=f'l1-{s}-{d}',src_phy=s,dst_phy=d,src_eid=a['eid'],dst_eid=b['eid'],
                          src_die=a['die'],dst_die=b['die'],src_port=a['port'],dst_port=b['port'],
                          net_instance_id=a['net_instance_id'],plane_id=a['plane_id'],l0_weight=1,l1_weight=1))
    if not paths:
        raise RuntimeError('no same-die single-port L1 pair: inspect candidates; do not synthesize EIDs')
    c=dict(schema='a5.single-port-l1.v1',generation=datetime.now(timezone.utc).isoformat(),devices=devices,
           rootinfo=str(args.root_info.resolve()),rootinfo_sha256=sha(args.root_info),
           inventory='inventory.json',inventory_sha256=sha(invpath),paths=paths)
    out=args.output_dir/'l1_policy.json';out.write_text(json.dumps(c,indent=2)+'\n')
    validate_config(out,devices,args.root_info)
    print(f'L1_POLICY_READY {out} paths={len(paths)} physical_single_port_verified=0')


if __name__=='__main__': main()
