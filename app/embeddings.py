"""Embedding providers.

- OpenAIEmbedder / AzureEmbedder: real semantic embeddings via the API.
- HashingEmbedder: a deterministic, offline bag-of-words embedding (hashing
  trick). It is keyword-based rather than semantic, but it lets the whole app,
  the tests, and the eval run with no API key and no network.

Each embedder exposes a `name` (stored alongside vectors so we can detect when
the provider changed and re-embed) and a `min_score` threshold used by the
retrieval gate. Cosine scores from different embedders are not comparable, so
the threshold lives with the embedder.
"""

import hashlib
import re

import numpy as np

STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i if in is it its
    me my of on or our so than that the their them then there these they this to us
    was we what when where which who why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        # crude plural folding: "classes" -> "class", "quizzes" -> "quiz", "policies" -> "policy"
        if len(t) > 4 and t.endswith(("sses", "zzes", "ches", "shes", "xes")):
            t = t[:-2]
        elif len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
        out.append(t)
    return out


class HashingEmbedder:
    name = "hashing-v1"
    min_score = 0.12

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        return h % self.dim, (1.0 if (h >> 64) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                idx, sign = self._bucket(f)
                out[i, idx] += sign
            # sublinear term frequency, then L2-normalize for cosine similarity
            out[i] = np.sign(out[i]) * np.log1p(np.abs(out[i]))
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class _APIEmbedder:
    min_score = 0.30

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.name = f"api:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def get_embedder(settings):
    """Use API embeddings when an embedding model is configured, else hashing."""
    provider = settings.resolved_provider()
    if provider == "azure" and settings.azure_embedding_deployment:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint,
        )
        return _APIEmbedder(client, settings.azure_embedding_deployment)
    if provider == "openai":
        from openai import OpenAI

        return _APIEmbedder(OpenAI(api_key=settings.openai_api_key), settings.openai_embedding_model)
    return HashingEmbedder()
