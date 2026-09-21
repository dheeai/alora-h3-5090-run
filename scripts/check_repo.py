#!/usr/bin/env python3
"""Static repository checks for the Alora Kannada H3 run-kit.

Runs without network access or the private dataset: validates JSON files, required
files, the loader split arithmetic, and relative Markdown links. Used by CI.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = []


def fail(msg):
    FAIL.append(msg)
    print('FAIL:', msg)


def check_required():
    required = [
        'README.md', 'README-HF.md', 'AGENT.md', 'LICENSE', 'CITATION.cff',
        'requirements.txt', 'RESULTS.template.json',
        'dataset/download_dataset.py', 'dataset/make_107_crops.py',
        'dataset/loader_config.json', 'dataset/eval_prompts.kn.txt',
        'scripts/run_benchmark.sh', 'scripts/monitor.py',
        'configs/aitoolkit-job-107.json', 'configs/simpletuner-32g.json',
        'docs/benchmark-instructions.md', 'docs/simpletuner-minimax-h3.md',
        'docs/license-attribution.md',
    ]
    for rel in required:
        if not (ROOT / rel).exists():
            fail('missing required file: %s' % rel)


def check_json():
    for p in sorted(ROOT.rglob('*.json')):
        if '.git' in p.parts:
            continue
        try:
            json.loads(p.read_text())
        except Exception as e:
            fail('invalid JSON %s: %s' % (p.relative_to(ROOT), e))


def check_loader_config():
    cfg = json.loads((ROOT / 'dataset/loader_config.json').read_text())
    total = cfg.get('dataset_total_clips')
    splits = cfg.get('splits', {})
    s = sum(v for k, v in splits.items() if isinstance(v, int))
    if total is not None and s != total:
        fail('loader_config splits sum %d != dataset_total_clips %d' % (s, total))
    if not cfg.get('media', {}).get('frames'):
        fail('loader_config missing media.frames')


def check_markdown_links():
    pattern = re.compile(r'\[[^\]]+\]\(([^)]+)\)')
    for md in sorted(ROOT.rglob('*.md')):
        if '.git' in md.parts:
            continue
        for target in pattern.findall(md.read_text()):
            if target.startswith(('http://', 'https://', '#', 'mailto:')):
                continue
            rel = target.split('#')[0]
            if not rel:
                continue
            if not (md.parent / rel).exists():
                fail('%s links to missing path: %s' % (md.relative_to(ROOT), target))


if __name__ == '__main__':
    check_required()
    check_json()
    check_loader_config()
    check_markdown_links()
    if FAIL:
        print('\n%d check(s) failed' % len(FAIL))
        sys.exit(1)
    print('all repository checks passed')
