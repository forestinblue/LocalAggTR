"""
Copy the lines that back every number in results/NUMBERS.md out of the raw
mmengine test logs, verbatim, into results/logs/<run>.txt.

The raw logs are not published (they contain the full lab training config).
Server paths such as /data/<user>/ are replaced by <server>/ in the excerpts.
Each excerpt keeps: the test command, the checkpoint line, state_dict key
warnings, the voxelizer / head settings that differ between runs, every
Epoch(test) progress line, and the final metric table.

Usage:
    python scripts/extract_log_excerpts.py --log-dir "<dir with raw logs>" --out-dir results/logs
"""

import argparse
import re
import statistics
from datetime import datetime
from pathlib import Path

RUNS = {
    # run id                 : raw log file name
    'aug_gaussvox_orighead':  'gaussian_voxelizer.log',
    'oct_localagg_v1':        'Local_Agg_1st.log',
    'oct_localagg_v2':        'Local_Agg_2nd.log',
    'dec_localagg':           'local_aggregation_test.log',
    'dec_gaussvox':           'gauss_voxel.log',
}

KEEP = re.compile(
    r"Testing command is|Load checkpoint|missing keys|unexpected key"
    r"|type='(LocalAggWrapper|GaussianVoxelizer)'|tau_quantile=|s_max_xyz="
    r"|ann_file='nuscenes_infos_val\.pkl'|use_image_mask=|GPU 0:|PyTorch: "
    r"|Epoch\(test\)")
TABLE_HEADER = '| classes |'
# server-local absolute paths are replaced before writing the excerpts
REDACT = re.compile(r"/data/[^/\s]+/")
TIME = re.compile(r"[^_]time: ([0-9.]+)")
STAMP = re.compile(r"^(\d{4}/)?(\d{2}/\d{2} \d{2}:\d{2}:\d{2})")


def parse_stamp(line, year):
    m = STAMP.match(line)
    if not m:
        return None
    text = (m.group(1) or f'{year}/') + m.group(2)
    return datetime.strptime(text, '%Y/%m/%d %H:%M:%S')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', required=True)
    ap.add_argument('--out-dir', default='results/logs')
    ap.add_argument('--year', default='2025',
                    help='year for log lines stamped MM/DD only')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for run, name in RUNS.items():
        src = Path(args.log_dir) / name
        lines = src.read_text(errors='replace').splitlines()

        kept = [f'# source: {name}', '# lines copied verbatim (server paths replaced by <server>/); "L<n>:" = line number in source', '']
        for i, line in enumerate(lines, 1):
            if KEEP.search(line):
                kept.append(f'L{i}: {REDACT.sub("<server>/", line)}')
        # final metric table = header line -1 .. +3
        hdr = max(i for i, l in enumerate(lines) if l.startswith(TABLE_HEADER))
        kept.append('')
        kept += [f'L{i + 1}: {lines[i]}' for i in range(hdr - 1, hdr + 4)]
        (out_dir / f'{run}.txt').write_text('\n'.join(kept) + '\n')

        # summary (derived values; NUMBERS.md states how each is computed)
        row = lines[hdr + 2].strip('|').split('|')
        cols = [c.strip() for c in lines[hdr].strip('|').split('|')]
        res = dict(zip(cols, [c.strip() for c in row]))
        prog = [l for l in lines if 'Epoch(test) [' in l and '6019/6019' not in l]
        times = [float(TIME.search(l).group(1)) for l in prog if TIME.search(l)]
        t0 = parse_stamp(next(l for l in lines if 'Load checkpoint' in l), args.year)
        t1 = parse_stamp(next(l for l in lines if 'Epoch(test) [6019/6019]' in l), args.year)
        wall = (t1 - t0).total_seconds()
        print(f"{run:24s} iou={res['iou']} miou={res['miou']} "
              f"median_iter_time={statistics.median(times):.4f}s (n={len(times)}) "
              f"wall={wall:.0f}s ({wall / 6019:.3f}s/sample)")


if __name__ == '__main__':
    main()
