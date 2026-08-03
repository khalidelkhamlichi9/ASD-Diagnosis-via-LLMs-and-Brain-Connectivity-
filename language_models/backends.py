"""
NeuroLingua-Bridge — language_models/backends.py
Registry of all 5 LLM backends used in the ABIDE benchmark.

Each backend exposes:
    D_LLM           : int    — hidden dimension
    get_embedding() : str -> Tensor (D_LLM,)
    generate()      : str -> str
"""
import os
import math
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GPT-2 Medium  (local, no API key)
# ─────────────────────────────────────────────────────────────────────────────
class GPT2Backend:
    MODEL_ID = "gpt2-medium"
    D_LLM    = 1024

    def __init__(self):
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        self.tok   = GPT2Tokenizer.from_pretrained(self.MODEL_ID)
        self.tok.pad_token = self.tok.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(self.MODEL_ID).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"[GPT2Backend] {self.MODEL_ID}  D={self.D_LLM}")

    def get_embedding(self, text: str, max_len: int = 128) -> torch.Tensor:
        ids = self.tok(text, max_length=max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out = self.model.transformer(
                **{k: v for k, v in ids.items() if k != "token_type_ids"})
        return out.last_hidden_state.mean(1).squeeze(0).cpu()

    def generate(self, prompt: str, max_new: int = 150) -> str:
        ids = self.tok.encode(prompt[-800:], return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new,
                do_sample=True, temperature=0.8, top_p=0.9,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalT5-large  (local, no API key)
# ─────────────────────────────────────────────────────────────────────────────
class ClinicalT5Backend:
    MODEL_ID = "luqh/ClinicalT5-large"
    FALLBACK  = "google/flan-t5-large"
    D_LLM     = 1024

    def __init__(self):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        try:
            self.tok   = T5Tokenizer.from_pretrained(self.MODEL_ID)
            self.model = T5ForConditionalGeneration.from_pretrained(self.MODEL_ID)
        except Exception:
            print(f"[ClinicalT5Backend] fallback -> {self.FALLBACK}")
            self.MODEL_ID = self.FALLBACK
            self.tok   = T5Tokenizer.from_pretrained(self.FALLBACK)
            self.model = T5ForConditionalGeneration.from_pretrained(self.FALLBACK)
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"[ClinicalT5Backend] {self.MODEL_ID}  D={self.D_LLM}")

    def _dev(self):
        return next(self.model.parameters()).device

    def get_embedding(self, text: str, max_len: int = 128) -> torch.Tensor:
        ids = self.tok(text, max_length=max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        ids = {k: v.to(self._dev()) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model.encoder(**ids)
        return out.last_hidden_state.mean(1).squeeze(0).cpu()

    def generate(self, prompt: str, max_new: int = 150) -> str:
        ids = self.tok(prompt, max_length=512, truncation=True,
                       return_tensors="pt")
        ids = {k: v.to(self._dev()) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new)
        return self.tok.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek-Coder-1.3B-Instruct  (local, GPU recommended)
# ─────────────────────────────────────────────────────────────────────────────
class DeepSeekBackend:
    MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
    D_LLM    = 2048

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok   = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"]  = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, **kwargs)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        print(f"[DeepSeekBackend] {self.MODEL_ID}  D={self.D_LLM}")

    def get_embedding(self, text: str, max_len: int = 128) -> torch.Tensor:
        ids = self.tok(text, max_length=max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        ids = {k: v.to(self.model.device)
               for k, v in ids.items() if k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            out = self.model(**ids, output_hidden_states=True)
        return out.hidden_states[-1].mean(1).squeeze(0).float().cpu()

    def generate(self, prompt: str, max_new: int = 150) -> str:
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new,
                                      do_sample=True, temperature=0.8)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Grok  (xAI API + GPT-2 proxy for embeddings)
# ─────────────────────────────────────────────────────────────────────────────
class GrokBackend:
    MODEL_ID = "grok-beta"
    D_LLM    = 768          # GPT-2 small proxy

    def __init__(self, api_key: str = None):
        from openai import OpenAI
        from transformers import GPT2Tokenizer, GPT2Model
        key = api_key or os.environ.get("XAI_API_KEY", "")
        self.client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        # Embedding proxy: GPT-2 small (no embeddings endpoint on xAI)
        self._etok   = GPT2Tokenizer.from_pretrained("gpt2")
        self._etok.pad_token = self._etok.eos_token
        self._emodel = GPT2Model.from_pretrained("gpt2").eval()
        for p in self._emodel.parameters():
            p.requires_grad = False
        print(f"[GrokBackend] API={self.MODEL_ID}  embed_proxy=gpt2  D={self.D_LLM}")

    def get_embedding(self, text: str, max_len: int = 128) -> torch.Tensor:
        ids = self._etok(text, max_length=max_len, truncation=True,
                         padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out = self._emodel(**ids)
        emb = out.last_hidden_state.mean(1).squeeze(0)
        # Adapt to D_LLM
        if emb.shape[0] > self.D_LLM:
            emb = emb[:self.D_LLM]
        elif emb.shape[0] < self.D_LLM:
            emb = F.pad(emb, (0, self.D_LLM - emb.shape[0]))
        return emb.cpu()

    def generate(self, prompt: str, max_new: int = 150) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.MODEL_ID,
                messages=[{"role": "user", "content": prompt[:2000]}],
                max_tokens=max_new)
            return r.choices[0].message.content.strip()
        except Exception as e:
            return f"[Grok error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Gemini  (Google API + deterministic hash embedding)
# ─────────────────────────────────────────────────────────────────────────────
class GeminiBackend:
    D_LLM = 768

    def __init__(self, api_key: str = None):
        import google.generativeai as genai
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        genai.configure(api_key=key)
        models = [m.name for m in genai.list_models()
                  if "generateContent" in m.supported_generation_methods]
        self.gemini     = genai.GenerativeModel(models[0])
        self.model_name = models[0]
        print(f"[GeminiBackend] {self.model_name}  D={self.D_LLM}")

    def get_embedding(self, text: str, max_len: int = 128) -> torch.Tensor:
        # Deterministic hash-based pseudo-embedding (Gemini has no public
        # embeddings endpoint in the free tier used here)
        import hashlib, struct
        h    = hashlib.sha256(text.encode()).digest()
        seed = struct.unpack("<Q", h[:8])[0] % (2 ** 32)
        rng  = torch.Generator(); rng.manual_seed(seed)
        return torch.randn(self.D_LLM, generator=rng)

    def generate(self, prompt: str, max_new: int = 150) -> str:
        try:
            session = self.gemini.start_chat(history=[])
            return session.send_message(prompt[:2000]).text.strip()
        except Exception as e:
            return f"[Gemini error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
BACKENDS = {
    "gpt2":       GPT2Backend,
    "clinicalt5": ClinicalT5Backend,
    "deepseek":   DeepSeekBackend,
    "grok":       GrokBackend,
    "gemini":     GeminiBackend,
}


def load_backend(name: str, **kwargs):
    """Load an LLM backend by name."""
    name = name.lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from {list(BACKENDS)}")
    return BACKENDS[name](**kwargs)


def get_d_llm(name: str) -> int:
    """
    Look up a backend's hidden dimension without instantiating it.

    D_LLM is a class attribute (fixed by the backbone architecture), so
    callers that only need the dimension -- e.g. to size Stage-2/3 layers
    before loading a frozen Stage-1/2 checkpoint -- can skip downloading
    and loading the (possibly multi-GB) underlying model entirely.
    """
    name = name.lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from {list(BACKENDS)}")
    return BACKENDS[name].D_LLM
