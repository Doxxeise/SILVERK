import os
import re
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

# ==========================================
# SILVERK CORE COMPONENTS
# ==========================================

class EngramGatedChild(nn.Module):
    """Gated Parametric Injection module for SILVERK architecture."""
    def __init__(self, d_model, rank):
        super().__init__()
        self.W_down = nn.Linear(d_model, rank, bias=False)
        self.W_up = nn.Linear(rank, d_model, bias=False)
        self.W_gate_down = nn.Linear(d_model, rank, bias=False)
        self.W_gate_up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.W_down.weight, std=0.01)
        nn.init.zeros_(self.W_up.weight)
        nn.init.normal_(self.W_gate_down.weight, std=0.01)
        nn.init.zeros_(self.W_gate_up.weight)

    def forward(self, h):
        value = self.W_up(F.gelu(self.W_down(h)))
        key = self.W_gate_up(F.gelu(self.W_gate_down(h)))
        h_norm = F.normalize(h, dim=-1)
        key_norm = F.normalize(key, dim=-1)
        gate = torch.sigmoid((key_norm * h_norm).sum(dim=-1, keepdim=True))
        return gate * value


def get_pdc_deviation_mask(base_model, input_ids, device, threshold=2.0, context_window=1):
    """Calculates per-token loss and a binary mask for surprisingly novel tokens (PDC)."""
    base_model.eval()
    with torch.no_grad():
        logits = base_model(input_ids).logits
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        raw_mask = (loss_per_token > threshold).float()
        expanded_mask = raw_mask.clone()
        seq_len = raw_mask.size(0)
        for i in range(seq_len):
            if raw_mask[i] == 1.0:
                start = max(0, i - context_window)
                end = min(seq_len, i + context_window + 1)
                expanded_mask[start:end] = 1.0
        final_mask = torch.cat([expanded_mask, torch.tensor([0.0]).to(device)])
        return loss_per_token, final_mask


def extract_pdc_entities(tokenizer, input_ids, mask):
    """Extracts entity keywords from PDC-salient token spans."""
    token_ids = input_ids[0].tolist()
    mask_vals = mask.tolist()
    spans = []
    current_span = []
    for i, (tid, m) in enumerate(zip(token_ids, mask_vals)):
        if m > 0.5:
            current_span.append(tid)
        else:
            if current_span:
                spans.append(current_span)
                current_span = []
    if current_span:
        spans.append(current_span)

    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'what', 'who', 'where', 'when',
        'how', 'question', 'answer', 'that', 'this', 'it', 'and', 'or',
        'does', 'did', 'do', 'above', 'all', 'others', 'flu', 'beautiful',
        'regarding', 'update', 'true', 'fact', 'factually', 'factual',
        'correct', 'corrected', 'rewrite', 'grammar',
        'city', 'country', 'person', 'place', 'thing', 'time',
        'speak', 'speaks', 'live', 'lives', 'love', 'loves',
        'study', 'studied', 'founded', 'invented', 'leads',
        'through', 'along', 'built', 'about', 'into',
        'new', 'old', 'has', 'had', 'have', 'been', 'being',
        'not', 'but', 'also', 'its', 'their', 'his', 'her',
        'can', 'could', 'would', 'should', 'will', 'shall',
        'from', 'into', 'which', 'than', 'then', 'there', 'here',
        'most', 'more', 'some', 'any', 'each', 'every', 'other',
        'such', 'only', 'very', 'just', 'now', 'still', 'even'
    }
    entity_words = set()
    for span in spans:
        text = tokenizer.decode(span, skip_special_tokens=True).strip()
        words = re.findall(r'[a-zA-Z]+', text.lower())
        for w in words:
            if len(w) > 2 and w not in stop_words:
                entity_words.add(w)
    return list(entity_words)


def generate_synthetic_views(fact_text, qa_text, subject_keywords):
    """Multi-View Amplification for robust knowledge training."""
    views = [fact_text, qa_text]
    subject = " ".join(subject_keywords) if subject_keywords else "the subject"
    views.append(f"Question: What is factually true regarding {subject}? Answer: {fact_text}")
    views.append(f"Fact update for {subject}: {fact_text}")
    return views


def simple_stem(word):
    """A minimal suffix-stripping stemmer."""
    for suffix in ['tion', 'sion', 'ment', 'ness', 'ence', 'ance',
                   'ting', 'ing', 'ted', 'led', 'ied', 'ded',
                   'ful', 'ous', 'ive', 'ent', 'ant', 'ery',
                   'ator', 'tor', 'ter', 'ner', 'ler', 'der',
                   'or', 'er', 'ed', 'ly', 'es', 'al', 'en', 's']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def get_sparse_embedding(text):
    """Retrieves a stemmed word frequency count (Bag of Words)."""
    words = re.findall(r'\w+', text.lower())
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by', 'what', 'who', 'where', 'when',
                  'how', 'question', 'answer', 'that', 'this', 'it', 'does', 'did'}
    stemmed = [simple_stem(w) for w in words if w not in stop_words]
    return Counter(stemmed)


def compute_sparse_similarity(query_emb, doc_emb):
    """BM25-style intersection similarity between two embeddings."""
    if not query_emb or not doc_emb:
        return 0.0
    query_words = set(query_emb.keys())
    doc_words = set(doc_emb.keys())
    if not query_words:
        return 0.0
    intersection = query_words.intersection(doc_words)
    return len(intersection) / len(query_words)


def generate_text(model, tokenizer, prompt, device, adapter=None, max_new_tokens=50):
    """Gated generation with automatic adapter injection and grammar cleanup."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
    generated_ids = input_ids.clone()

    if adapter is not None:
        adapter.to(device)

    for step in range(max_new_tokens):
        with torch.no_grad():
            h_out = model(generated_ids, output_hidden_states=True).hidden_states[-1]
            if adapter is not None:
                h_float = h_out.float()
                base_norm = h_float.norm()
                h_float = h_float + adapter(h_float)
                h_out = (h_float * (base_norm / (h_float.norm() + 1e-8))).half()
            logits = model.lm_head(h_out)
            next_token_logits = logits[:, -1, :]

            # Simple repetition penalty
            generated_so_far = generated_ids[0, input_ids.shape[1]:].tolist()
            for token_id in set(generated_so_far):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= 1.3
                else:
                    next_token_logits[0, token_id] /= 1.3

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    output_tokens = generated_ids[0, input_ids.shape[1]:]
    raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    # Pass 2: Grammar correction (only when adapter was used)
    if adapter is not None and raw_output:
        cleanup_prompt = f"Rewrite this with correct grammar: {raw_output}\nCorrected:"
        cleanup_ids = tokenizer.encode(cleanup_prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
        cleanup_generated = cleanup_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(cleanup_generated)
                next_logits = outputs.logits[:, -1, :]
                next_tok = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
                cleanup_generated = torch.cat([cleanup_generated, next_tok], dim=-1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
        cleanup_tokens = cleanup_generated[0, cleanup_ids.shape[1]:]
        cleaned = tokenizer.decode(cleanup_tokens, skip_special_tokens=True).strip()
        adapter.to("cpu")
        return cleaned if cleaned else raw_output

    if adapter is not None:
        adapter.to("cpu")

    return raw_output
