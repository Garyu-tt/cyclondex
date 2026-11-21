"""Observe Linux memory and NVIDIA devices; never load model or change host.

Run separately on each intended runtime environment (Proxmox host vs guest).
Forecasts use user-supplied checkpoint-byte estimate, never scan disks.
"""
import argparse
import json

from cyclondex.host_budget import snapshot, check_candidate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight-bytes", type=int, required=True)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--kv-gib", type=float, default=1)
    p.add_argument("--reserve-gib", type=float, default=8)
    args = p.parse_args()
    actual = snapshot()
    result = check_candidate(
        actual, args.weight_bytes, independent_sessions=args.sessions,
        kv_context_gib=args.kv_gib, reserve_gib=args.reserve_gib)
    print(json.dumps({"host": actual, "budget": result}, sort_keys=True))


if __name__ == "__main__":
    main()
