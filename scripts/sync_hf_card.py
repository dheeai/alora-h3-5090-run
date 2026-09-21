"""Push the local dataset card (README-HF.md) to the Hugging Face dataset repo.

The dataset card lives on Hugging Face; `README-HF.md` in this repo is the versioned source.
Run this after editing it so the two never drift.

Requires: export HF_TOKEN=<write-scope token for aaron1z/alora-kannada-fleurs-v1>
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = 'aaron1z/alora-kannada-fleurs-v1'
CARD = Path(__file__).resolve().parent.parent / 'README-HF.md'


def main():
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise SystemExit('Set HF_TOKEN with write access to %s' % REPO)
    if not CARD.exists():
        raise SystemExit('missing %s' % CARD)
    HfApi(token=token).upload_file(
        path_or_fileobj=str(CARD), path_in_repo='README.md',
        repo_id=REPO, repo_type='dataset',
        commit_message='Sync dataset card from run-kit README-HF.md')
    print('synced %s -> %s/README.md' % (CARD.name, REPO))


if __name__ == '__main__':
    main()
