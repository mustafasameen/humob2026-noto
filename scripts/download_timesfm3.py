"""Download a pinned official checkpoint and record its revision and hashes."""
import hashlib,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('results/experiment3')
info=HfApi().model_info('google/timesfm-3.0-pytorch')
(root/'weights-revision.json').write_text(json.dumps({'repo':info.id,'revision':info.sha},indent=2)+'\n')
path=Path(snapshot_download(info.id,revision=info.sha,local_dir='models/timesfm-3.0-pytorch',allow_patterns=['*.json','*.safetensors','README.md','LICENSE*']))
(root/'weights-checksums.json').write_text(json.dumps({str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file() and '.cache' not in p.parts},indent=2)+'\n')
print('Downloaded',info.id,info.sha)
