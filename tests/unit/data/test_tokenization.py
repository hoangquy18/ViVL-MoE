from __future__ import annotations

import pytest

from tests.fakes import tiny_clip_tokenizer, tiny_roberta_tokenizer, tiny_t5_tokenizer
from vivlmoe.config.schema import DataCfg, ModelCfg, TokenizerCfg
from vivlmoe.data.tokenization import build_tokenizers, tiny_wordlevel_tokenizer


def test_fake_tokenizer_layouts() -> None:
    rb = tiny_roberta_tokenizer()
    assert (rb.bos_token_id, rb.pad_token_id, rb.eos_token_id, rb.sep_token) == (0, 1, 2, "</s>")
    enc = rb("một con chó", "hai")["input_ids"]
    assert enc[0] == 0 and enc[-1] == 2 and enc.count(2) == 3  # <s> A </s></s> B </s>
    assert rb("một xyz")["input_ids"][2] == rb.unk_token_id
    cl = tiny_clip_tokenizer()
    ids = cl("A Dog", padding="max_length", max_length=77)["input_ids"]
    assert (
        len(ids) == 77 and ids[0] == cl.bos_token_id and cl.pad_token_id == cl.eos_token_id == 999
    )
    assert ids.index(cl.eos_token_id) == 3 and cl.unk_token_id != cl.eos_token_id
    t5 = tiny_t5_tokenizer()
    assert (t5.pad_token_id, t5.eos_token_id) == (0, 1) and t5("một")["input_ids"][-1] == 1
    assert len(tiny_wordlevel_tokenizer("roberta", 200)) == 200
    with pytest.raises(ValueError):
        tiny_wordlevel_tokenizer("roberta", 4)


def test_build_tokenizers_tiny_configs() -> None:
    m = ModelCfg()
    m.text.pretrained = False
    m.text.tiny_config = {"vocab_size": 300}
    m.vision.pretrained = False
    m.generation.pretrained = False
    m.generation.tiny_config = {"vocab_size": 120}
    b = build_tokenizers(
        m, DataCfg(tokenizer=TokenizerCfg(vi_max_len=10)), need_clip=True, need_decoder=True
    )
    assert (
        len(b.vi) == 300 and len(b.decoder) == 120 and b.clip_en is not None and b.vi_max_len == 10
    )
    b2 = build_tokenizers(m, DataCfg())
    with pytest.raises(ValueError):
        b2.require_clip()
    with pytest.raises(ValueError):
        build_tokenizers(m, DataCfg(tokenizer=TokenizerCfg(en_max_len=80)))
