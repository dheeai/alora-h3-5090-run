"""Poll VRAM (nvidia-smi) and system RAM into a CSV while training runs.

Usage: python scripts/monitor.py --out monitor_107.csv --interval 2
Stop with Ctrl-C or by killing the process.
"""
import argparse, csv, subprocess, time
from datetime import datetime, timezone
from pathlib import Path


def vram():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
        used, total = out.split(',')
        return int(used), int(total)
    except Exception:
        return None, None


def ram():
    try:
        info = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, v = line.split(':', 1)
            info[k] = int(v.strip().split()[0]) // 1024  # MiB
        return info.get('MemTotal', 0), info.get('MemAvailable', 0)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='monitor.csv')
    ap.add_argument('--interval', type=float, default=2.0)
    a = ap.parse_args()
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'vram_used_mib', 'vram_total_mib', 'sys_total_mib', 'sys_avail_mib'])
        while True:
            vu, vt = vram(); st, sa = ram()
            w.writerow([datetime.now(timezone.utc).isoformat(), vu, vt, st, sa])
            f.flush()
            time.sleep(a.interval)


if __name__ == '__main__':
    main()