"""Download the private Alora Kannada dataset from Hugging Face.

Requires: export HF_TOKEN=<read-scope token with access to aaron1z/alora-kannada-fleurs-v1>
"""
import json
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

meta = [json.loads(x) for x in (Path(path) / 'metadata.jsonl').read_text().splitlines()]
print('metadata rows', len(meta), '| mp4', len(mp4), '| wav', len(wav), '| captions', len(txt))
missing = [(r['id']) for r in meta
           if not (Path(path) / r['collection'] / r['video_file']).exists()
           or not (Path(path) / r['collection'] / r['audio_file']).exists()]
assert not missing, missing[:5]
assert {r['frames'] for r in meta} == {124}
assert {r['fps'] for r in meta} == {24}
splits = {}
for split in ('train', 'val', 'test'):
    rows = [json.loads(x) for x in (Path(path) / (split + '.jsonl')).read_text().splitlines()]
    splits[split] = rows
    print(split, len(rows))
assert sum(len(v) for v in splits.values()) == len(meta), (sum(len(v) for v in splits.values()), len(meta))
groups = {s: {r['source_id'] for r in rows} for s, rows in splits.items()}
assert not (groups['train'] & groups['val'] | groups['train'] & groups['test'] | groups['val'] & groups['test'])
print('OK')