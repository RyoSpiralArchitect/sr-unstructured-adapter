# SPDX-License-Identifier: AGPL-3.0-or-later

from sr_adapter.embedding import BlockEmbedder, EmbeddingIndex
from sr_adapter.schema import Block, BBox, Provenance


def _make_block(text: str, page: int = 0) -> Block:
    prov = Provenance(page=page, bbox=BBox(x0=0.0, y0=float(page), x1=10.0, y1=float(page) + 5.0))
    return Block(type="paragraph", text=text, prov=prov, attrs={"key": text})


def test_block_embedder_deterministic():
    embedder = BlockEmbedder(dimensions=64, seed=42)
    block = _make_block("Example heading", page=2)
    vec1 = embedder.embed_block(block)
    vec2 = embedder.embed_block(block)
    assert len(vec1) == 64
    assert vec1 == vec2
    assert abs(sum(component * component for component in vec1) - 1.0) < 1e-6


def test_embedding_index_similarity():
    embedder = BlockEmbedder(dimensions=64, seed=7)
    blocks = [_make_block(f"Item {idx}", page=idx) for idx in range(3)]
    vectors = embedder.embed(blocks)
    index = EmbeddingIndex(len(vectors[0]))
    for idx, vector in enumerate(vectors):
        index.add(vector, {"id": idx})
    hits = index.search(vectors[0], top_k=2)
    assert hits
    assert hits[0].metadata["id"] == 0
