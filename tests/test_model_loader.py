"""Regression test: load_model_and_tokenizer must not hardcode `token=True`.

`token=True` forces huggingface_hub to require a resolvable auth token for
ANY model load, even fully public ones -- this broke a fresh container
with no cached HF login (LocalTokenNotFoundError), even though every model
in MODEL_REGISTRY except llama32_3b is public (verified via
`huggingface_hub.model_info(..., token=False)` for each registry entry).
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from re_polar.core import model_loader


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(eos_token_id=0)

    def eval(self):
        return self

    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0


def _stub_from_pretrained(monkeypatch, device):
    monkeypatch.setattr(model_loader, "detect_device", lambda: device)
    calls = {}

    def fake_model_from_pretrained(model_id, **kwargs):
        calls["model_kwargs"] = kwargs
        return _FakeModel()

    def fake_tokenizer_from_pretrained(model_id, **kwargs):
        calls["tokenizer_kwargs"] = kwargs
        return _FakeTokenizer()

    monkeypatch.setattr(
        model_loader.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fake_model_from_pretrained),
    )
    monkeypatch.setattr(
        model_loader.AutoTokenizer,
        "from_pretrained",
        staticmethod(fake_tokenizer_from_pretrained),
    )
    return calls


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_load_model_and_tokenizer_never_forces_token_true(monkeypatch, device):
    calls = _stub_from_pretrained(monkeypatch, device)
    model_loader.load_model_and_tokenizer("fake/model-id")
    assert calls["model_kwargs"].get("token") is not True
    assert calls["tokenizer_kwargs"].get("token") is not True
