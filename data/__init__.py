"""Dataset schema, random scene generation, and torch-friendly packing."""

from .schema import (
    MAX_BOUNCES,
    MAX_CORNERS,
    MAX_WALLS,
    PAD_ID,
    RX_ID,
    TX_ID,
    VOCAB_DIFFRACT_OFFSET,
    VOCAB_REFLECT_OFFSET,
    VOCAB_WALL_OFFSET,
    decode_interactions,
    decode_wall_ids,
    pack_sample,
    tokens_from_interactions,
    tokens_from_wall_ids,
    wall_token_id,
)
from .generate import generate_dataset, generate_split

__all__ = [
    "PAD_ID",
    "TX_ID",
    "RX_ID",
    "MAX_WALLS",
    "MAX_CORNERS",
    "MAX_BOUNCES",
    "VOCAB_WALL_OFFSET",
    "VOCAB_REFLECT_OFFSET",
    "VOCAB_DIFFRACT_OFFSET",
    "wall_token_id",
    "tokens_from_wall_ids",
    "tokens_from_interactions",
    "decode_wall_ids",
    "decode_interactions",
    "pack_sample",
    "generate_dataset",
    "generate_split",
]
