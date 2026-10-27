from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=None)
def _load_model(model_name):
    # Loading the model is slow, so do it once per model and reuse it
    return SentenceTransformer(model_name)


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = _load_model(model_name)
    return model.encode(text)
