"""
NeuroLingua-Bridge - chatbot/fmri_chatbot.py
fMRI-grounded conversational interface - 3 paradigms (Wei et al. Fig. 5)

  Paradigm 1 - Single-question Single-answer
  Paradigm 2 - Multi-question Multi-answer  (separator: |)
  Paradigm 3 - Open-ended description
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

_NET_ROIS = {
    "Visual":      list(range(0,   57)),  "SomMot":  list(range(57,  114)),
    "DorsAttn":    list(range(114, 171)), "SalVentAttn": list(range(171, 228)),
    "Cont":        list(range(228, 285)), "Default": list(range(285, 342)),
    "Limbic":      list(range(342, 399)), "Subcort": list(range(400, 450)),
}
_LABEL = {0: "Neurotypical (Control)", 1: "ASD"}
_SEX   = {0: "Male", 1: "Female"}


class fMRILMChatbot:
    """
    3-paradigm chatbot grounded on real fMRI predictions + Grad-CAM.

    Quick start
    -----------
    >>> bot = fMRILMChatbot(clf, tok, lw, llm,
    ...                     ts_all, lbl_all, sex_all, site_list, descs,
    ...                     train_idx, val_idx, test_idx)
    >>> bot.load_subject(0, split="test")
    >>> bot.chat("What is the ASD prediction?")          # Paradigm 1
    >>> bot.multi_qa(["Sex?", "ASD?", "Lambda?"])        # Paradigm 2
    >>> bot.summarise()                                  # Paradigm 3
    """

    def __init__(self, classifier, tokenizer, loss_weights, llm_backend,
                 ts_all, lbl_all, sex_all, site_list, descs,
                 train_idx, val_idx, test_idx, device=None):
        self.clf     = classifier.eval()
        self.tok     = tokenizer
        self.lw      = loss_weights
        self.llm     = llm_backend
        self.ts_all  = ts_all
        self.lbl_all = lbl_all
        self.sex_all = sex_all
        self.sites   = site_list
        self.descs   = descs
        self.splits  = {"train": train_idx, "val": val_idx, "test": test_idx}
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.subject = None

    # ── subject loading ───────────────────────────────────────────────────────

    def load_subject(self, idx: int, split: str = "test"):
        g    = self.splits[split][idx]
        ts_t = torch.tensor(self.ts_all[g:g+1]).to(self.device)

        with torch.no_grad():
            lg, sx = self.clf(ts_t)
            probs  = F.softmax(lg, 1)[0].cpu().numpy()
            pred   = int(lg.argmax(1).item())
            sp     = int(F.softmax(sx, 1)[0].argmax().item())

        ts_g = ts_t.clone().requires_grad_(True)
        lg2, _ = self.clf(ts_g); lg2[0, 1].backward()
        grad = ts_g.grad[0]
        imp  = {net: float(grad[:, [r for r in rois if r < grad.shape[1]]].abs().mean())
                for net, rois in _NET_ROIS.items()
                if any(r < grad.shape[1] for r in rois)}

        st = int(self.sex_all[g]) - 1
        self.subject = {
            "desc":   self.descs[g],  "pred": pred,  "true": int(self.lbl_all[g]),
            "conf":   float(probs[1]), "probs": probs, "imp": imp,
            "idx":    idx,  "split": split,
            "site":   self.sites[g] if g < len(self.sites) else "UNK",
            "sex_t":  max(0, min(1, st)), "sex_p": sp,
        }
        self._card()

    def _card(self):
        s  = self.subject
        w  = self.lw.weights.detach().cpu().numpy()
        t3 = sorted(s["imp"].items(), key=lambda x: -x[1])[:3]
        print("=" * 58)
        print(f"  Subject  : {s['split']}[{s['idx']}] | Site: {s['site']}")
        print(f"  ASD True : {_LABEL[s['true']]}")
        print(f"  ASD Pred : {_LABEL[s['pred']]} (conf={s['conf']:.1%})")
        print(f"  Sex      : {_SEX.get(s['sex_t'], '?')}")
        print(f"  Correct? : {'YES OK' if s['pred']==s['true'] else 'NO X'}")
        print(f"  Lambda   : w_q={w[0]:.3f} w_s={w[1]:.3f} w_g={w[2]:.3f} (Sigma={w.sum():.4f})")
        print("  Top-3 Networks (Grad-CAM):")
        for net, v in t3:
            print(f"    {net:<15}: {v:.4f}  {'#'*min(int(v*500),28)}")
        print("=" * 58)

    # ── structured answer engine ──────────────────────────────────────────────

    def _answer(self, q: str) -> str:
        s  = self.subject
        w  = self.lw.weights.detach().cpu().numpy()
        t3 = sorted(s["imp"].items(), key=lambda x: -x[1])[:3]
        ql = q.lower()

        if any(k in ql for k in ["sex", "gender", "male", "female"]):
            return f"The subject is **{_SEX.get(s['sex_t'],'?')}** (model: {_SEX.get(s['sex_p'],'?')})."
        if any(k in ql for k in ["asd", "autism", "prediction", "diagnosis", "predict"]):
            ok = s["pred"] == s["true"]
            return (f"Prediction: **{_LABEL[s['pred']]}** (conf={s['conf']:.1%}). "
                    f"True: {_LABEL[s['true']]}. {'Correct.' if ok else 'Incorrect.'}")
        if any(k in ql for k in ["site", "scanner", "location"]):
            return f"Scanned at **{s['site']}** (unseen test site - domain-shift evaluation)."
        if any(k in ql for k in ["network", "grad", "important", "region", "brain"]):
            return f"Most diagnostic networks: **{', '.join(f'{n}({v:.4f})' for n,v in t3)}**."
        if any(k in ql for k in ["lambda", "equation", "loss", "weight"]):
            return (f"L_tok = **{w[0]:.4f}·L_quant + {w[1]:.4f}·L_SigLIP + {w[2]:.4f}·L_GRL** "
                    f"(Sigma={w.sum():.6f}=1.0, softmax-guaranteed).")
        if any(k in ql for k in ["connectivity", "fc", "descriptor"]):
            return f"FC descriptor: {s['desc'][:300]}..."

        # Fallback: ask LLM
        ctx = (f"fMRI analysis: {_LABEL[s['pred']]} ({s['conf']:.0%}). "
               f"Top networks: {', '.join(n for n,_ in t3)}. "
               f"L_tok={w[0]:.2f}·Lq+{w[1]:.2f}·Ls+{w[2]:.2f}·Lg. "
               f"Question: {q} Answer:")
        try:   return self.llm.generate(ctx, max_new=200)
        except: return f"[LLM unavailable] {_LABEL[s['pred']]} ({s['conf']:.1%})"

    # ── 3 paradigms ──────────────────────────────────────────────────────────

    def chat(self, question: str, paradigm: str = "single") -> str:
        if not self.subject: return "Load a subject first: bot.load_subject(0)"
        if paradigm == "multi":
            return " | ".join(self._answer(q.strip())
                              for q in question.split("|") if q.strip())
        if paradigm == "open": return self.summarise()
        return self._answer(question)

    def multi_qa(self, questions: list) -> str:
        return self.chat(" | ".join(questions), paradigm="multi")

    def summarise(self) -> str:
        if not self.subject: return "Load a subject first."
        s  = self.subject
        w  = self.lw.weights.detach().cpu().numpy()
        t3 = sorted(s["imp"].items(), key=lambda x: -x[1])[:3]
        return (
            f"=== NeuroLingua-Bridge Analysis ({s['site']}) ===\n\n"
            f"Subject: {_SEX.get(s['sex_t'],'?')} | Site: {s['site']}\n\n"
            f"1. ASD Assessment\n"
            f"   Prediction = {_LABEL[s['pred']]} (conf={s['conf']:.1%})\n"
            f"   True label = {_LABEL[s['true']]} | "
            f"{'Correct OK' if s['pred']==s['true'] else 'Incorrect X'}\n\n"
            f"2. Diagnostic Networks (Grad-CAM)\n"
            + "".join(f"   - {n}: {v:.4f}\n" for n, v in t3) +
            f"\n3. FC Descriptor\n   {s['desc'][:300]}...\n\n"
            f"4. Lambda Equation (Wei et al. 2026)\n"
            f"   L_tok = {w[0]:.4f}·L_quant + {w[1]:.4f}·L_SigLIP + {w[2]:.4f}·L_GRL\n"
            f"   Sigma = {w.sum():.6f} = 1.0  (softmax-guaranteed)\n"
        )

    def explain_lambda(self) -> str: return self.chat("lambda equation")
    def reset(self): self.subject = None; print("Reset.")
