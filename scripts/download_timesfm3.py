"""Download the official TimesFM-3 PyTorch checkpoint at a pinned revision and record its hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = 'google/timesfm-3.0-pytorch'
REVISION = '43046b85ec22d584a13f8098c2ed39c889e129c2'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--out', default='models/timesfm-3.0-pytorch')
    p.add_argument('--record', default='results/experiment3')
    args = p.parse_args()
    record = Path(args.record)
    record.mkdir(parents=True, exist_ok=True)
    path = Path(snapshot_download(REPO, revision=args.revision, local_dir=args.out,
                                  allow_patterns=['*.json', '*.safetensors', 'README.md', 'LICENSE*']))
    (record / 'weights-revision.json').write_text(json.dumps({'repo': REPO, 'revision': args.revision}, indent=2) + '\n')
    hashes = {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in path.rglob('*') if f.is_file() and '.cache' not in f.parts}
    (record / 'weights-checksums.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('Downloaded', REPO, args.revision)


if __name__ == '__main__':
    main()
