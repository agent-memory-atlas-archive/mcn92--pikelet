"""Regenerates bge-small-reference.json: fp32 BAAI/bge-small-en-v1.5 vectors
(mean pooling over the attention mask, L2-normalized, as Workers AI serves
the model) for the passages and queries in that file, at the revision the
pinned build-encoder blob was exported from. test/build_encoder.mjs checks
create's build-time encoder against them.

    python3 test/fixtures/build-encoder/make_reference.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = 'BAAI/bge-small-en-v1.5'
REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
PATH = Path(__file__).resolve().parent / 'bge-small-reference.json'

ref = json.loads(PATH.read_text())
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval()
texts = [ref['queryPrefix'] + q['text'] for q in ref['queries']] + [p['text'] for p in ref['passages']]
with torch.no_grad():
    batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    hidden = model(**batch).last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1).float()
    vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
for item, vector in zip(ref['queries'] + ref['passages'], vectors.tolist()):
    item['vector'] = [round(x, 7) for x in vector]
ref['model'], ref['revision'] = MODEL, REVISION
PATH.write_text(json.dumps(ref, indent=1) + '\n')
print(f'wrote {len(texts)} vectors to {PATH.name}')
