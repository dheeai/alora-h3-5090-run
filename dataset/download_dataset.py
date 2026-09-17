"""Download the private Alora Kannada dataset from Hugging Face.

Requires: export HF_TOKEN=<read-scope token with access to aaron1z/alora-kannada-fleurs-v1>
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'aaron1z/alora-kannada-fleurs-v1'
OUT = Path(__file__).resolve().parent.parent / 'data' / 'alora-kannada-fleurs-v1'

if not os.environ.get('HF_TOKEN'):
    raise SystemExit('Set HF_TOKEN to a read-scope token with access to %s' % REPO)

path = snapshot_download(REPO, repo_type='dataset', local_dir=str(OUT), token=os.environ['HF_TOKEN'])
print('downloaded to', path)

mp4 = list(Path(path).glob('*/*.mp4'))
wav = list(Path(path).glob('*/*.wav'))
txt = list(Path(path).glob('*/*.txt'))
print('clips', len(mp4), 'wavs', len(wav), 'captions', len(txt))

import json
meta = [json.loads(x) for x in (Path(path) / 'metadata.jsonl').read_text().splitlines()]
assert len(meta) == 1000, len(meta)
assert {r['frames'] for r in meta} == {124}
assert {r['fps'] for r in meta} == {24}
for split in ('train', 'val', 'test'):
    rows = [json.loads(x) for x in (Path(path) / (split + '.jsonl')).read_text().splitlines()]
    print(split, len(rows))
print('OK')