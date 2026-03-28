"""
SILVERK Architecture - REAL MQuAKE-CF-3k Benchmark (50 Cases)
PDC Entity Graph + IDF-ELQR Chaining + Multi-View Amplification

Evaluates SILVERK's multi-hop knowledge editing on real counterfactual cases from MQuAKE-CF-3k.
"""

import os
import re
import json
import math
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

# Import SILVERK core components
from silverk_core import (
    EngramGatedChild,
    get_pdc_deviation_mask,
    extract_pdc_entities,
    generate_synthetic_views,
    get_sparse_embedding,
    compute_sparse_similarity,
    generate_text
)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANK = 64
LR = 5e-4
EPOCHS = 20
TARGET_LOSS = 0.05
PDC_THRESHOLD = 2.0
SIMILARITY_THRESHOLD = 0.50
ELQR_THRESHOLD = 0.25
SAVE_DIR = "./silverk_real_mquake_full"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_CASES = 3000
CHECKPOINT_EVERY = 50 
MAX_RUNTIME_HOURS = 11.5 

print("=" * 65)
print(" SILVERK ARCHITECTURE - REAL MQuAKE FULL BENCHMARK")
print(" PDC Entity Graph + IDF-ELQR + Multi-View + Checkpoint/Resume")
print("=" * 65)


# ==========================================
# DATASET UTILS
# ==========================================
def download_mquake():
    """Download MQuAKE-CF-3k dataset."""
    import urllib.request
    url = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k.json"
    cache_path = os.path.join(SAVE_DIR, "MQuAKE-CF-3k.json")
    if os.path.exists(cache_path):
        print(f"  Found cached MQuAKE-CF-3k at {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)
    print(f"  Downloading MQuAKE-CF-3k from GitHub...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  Failed download: {e}. Trying HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("Yiming/MQuAKE", "MQuAKE-CF-3k", split="test")
        data = [dict(row) for row in ds]
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data


def mquake_case_to_facts(case):
    """Convert MQuAKE case to SILVERK training facts."""
    facts = []
    rewrites = case.get("requested_rewrite", [])
    for i, rw in enumerate(rewrites):
        subject = rw.get("subject", "")
        prompt_template = rw.get("prompt", "{}")
        target_new = rw.get("target_new", {}).get("str", "")
        target_true = rw.get("target_true", {}).get("str", "")

        fact_text = prompt_template.replace("{}", subject) + " " + target_new + "."
        question = rw.get("question", f"What is related to {subject}?")
        qa_text = f"Question: {question} Answer: {target_new}."
        subject_kw = [w.lower() for w in re.findall(r'[a-zA-Z]+', subject) if len(w) > 2]

        facts.append({
            "id": f"case{case.get('case_id', 0)}_edit{i}",
            "text": fact_text,
            "qa": qa_text,
            "subject_keywords": subject_kw,
            "target_new": target_new,
            "target_true": target_true,
            "subject": subject,
        })
    return facts


# ==========================================
# THE BENCHMARK
# ==========================================
def run_benchmark():
    script_start_time = time.time()
    mquake_data = download_mquake()
    if mquake_data is None: return

    multi_hop_cases = [c for c in mquake_data if len(c.get("requested_rewrite", [])) >= 2]
    test_cases = multi_hop_cases[:MAX_CASES]

    print(f"  Loading Base Model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    base_model.eval()
    base_model.requires_grad_(False)
    d_model = base_model.config.hidden_size

    print("\n  PHASE 1: INGESTING ALL EDITS")
    adapter_registry = {}
    entity_graph = {}
    centroid_index = {}
    
    # Checkpoint Resume
    graph_path = os.path.join(SAVE_DIR, "entity_graph.json")
    if os.path.exists(graph_path):
        with open(graph_path, "r") as f:
            entity_graph.update(json.load(f))
        print(f"  ✓ Resumed entity graph ({len(entity_graph)} nodes)")

    all_facts = []
    for case in test_cases:
        all_facts.extend(mquake_case_to_facts(case))

    start_time = time.time()
    newly_trained = 0

    for idx, fact in enumerate(all_facts):
        doc_id = fact["id"]
        pt_path = os.path.join(SAVE_DIR, f"{doc_id}.pt")

        if os.path.exists(pt_path) and doc_id in entity_graph:
            adapter = EngramGatedChild(d_model, RANK)
            adapter.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
            adapter.eval()
            adapter_registry[doc_id] = adapter
            centroid_index[doc_id] = get_sparse_embedding(fact["text"])
            continue

        views = generate_synthetic_views(fact["text"], fact["qa"], fact["subject_keywords"])
        adapter = EngramGatedChild(d_model, RANK).to(DEVICE)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
        all_output_entities = []

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            batch_losses = []
            for text in views:
                input_ids = tokenizer.encode(text, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)
                with torch.no_grad():
                    h_base = base_model(input_ids, output_hidden_states=True).hidden_states[-1].detach().float()
                _, mask = get_pdc_deviation_mask(base_model, input_ids, DEVICE, threshold=PDC_THRESHOLD)

                if epoch == 0:
                    all_output_entities.extend(extract_pdc_entities(tokenizer, input_ids, mask))

                if mask.sum() == 0: continue
                h_child = h_base + adapter(h_base)
                logits = base_model.lm_head(h_child.half()).float()
                
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(logits[..., :-1, :].reshape(-1, logits.size(-1)), input_ids[..., 1:].reshape(-1))
                soft_mask = torch.where(mask[1:].bool(), torch.tensor(1.0, device=DEVICE), torch.tensor(0.1, device=DEVICE))
                min_len = min(losses.size(0), soft_mask.size(0))
                loss = (losses[:min_len] * soft_mask[:min_len]).sum() / (soft_mask[:min_len].sum() + 1e-9)
                batch_losses.append(loss)

            if batch_losses:
                avg_loss = torch.stack(batch_losses).mean()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                if avg_loss.item() < TARGET_LOSS: break

        adapter.eval().to("cpu")
        torch.save(adapter.state_dict(), pt_path)
        adapter_registry[doc_id] = adapter
        entity_graph[doc_id] = {
            "input_entities": fact["subject_keywords"],
            "output_entities": [e.lower() for e in set(all_output_entities)],
        }
        centroid_index[doc_id] = get_sparse_embedding(fact["text"])
        newly_trained += 1

        if newly_trained % 10 == 0:
            print(f"    Ingested {idx+1}/{len(all_facts)} edits...")
        if newly_trained % CHECKPOINT_EVERY == 0:
            with open(graph_path, "w") as f: json.dump(entity_graph, f, indent=2)

        if (time.time() - script_start_time) > MAX_RUNTIME_HOURS * 3600:
            print("⏳ Max runtime reached. Exiting early.")
            break

    with open(graph_path, "w") as f: json.dump(entity_graph, f, indent=2)

    # Retrieval & ELQR
    def retrieve_adapter(query_text):
        query_emb = get_sparse_embedding(query_text)
        best_id, best_sim = None, -1.0
        for doc_id, doc_emb in centroid_index.items():
            sim = compute_sparse_similarity(query_emb, doc_emb)
            if sim > best_sim: best_sim, best_id = sim, doc_id
        return best_id, best_sim

    def find_entity_links(source_id):
        src_case = source_id.split("_edit")[0]
        src_outputs = set(entity_graph[source_id]["output_entities"])
        case_entities = {did: meta for did, meta in entity_graph.items() if did.split("_edit")[0] == src_case}
        
        input_entity_df = Counter()
        for meta in case_entities.values():
            for word in set(meta["input_entities"]): input_entity_df[word] += 1
        num_docs = max(len(case_entities), 1)

        links = []
        for dst_id, dst_meta in case_entities.items():
            if dst_id == source_id: continue
            overlap = src_outputs.intersection(set(dst_meta["input_entities"]))
            if overlap:
                idf_score = sum(math.log(num_docs / input_entity_df.get(w, 1)) for w in overlap)
                links.append((dst_id, idf_score))
        links.sort(key=lambda x: x[1], reverse=True)
        return [l[0] for l in links]

    def elqr_multi_hop(query_text, max_hops=3):
        chain = []
        best_id, best_sim = retrieve_adapter(query_text)
        if best_sim < ELQR_THRESHOLD:
            return generate_text(base_model, tokenizer, query_text, DEVICE), chain
        chain.append(best_id)
        curr_id = best_id
        for _ in range(2, max_hops + 1):
            links = find_entity_links(curr_id)
            if not links: break
            curr_id = links[0]
            chain.append(curr_id)
        output = generate_text(base_model, tokenizer, query_text, DEVICE, adapter=adapter_registry[curr_id])
        return output, chain

    # Phase 4 (Multi-Hop Evaluation)
    multi_hop_correct = 0
    multi_hop_total = 0
    for case in test_cases:
        case_id = case.get("case_id", 0)
        questions = case.get("questions", [])
        new_answer = case.get("new_answer", "")
        aliases = [ans.lower() for ans in [new_answer] + case.get("new_answer_alias", []) if len(ans.strip()) > 1]
        
        case_pass = False
        best_output = ""
        best_chain = []
        for q in questions:
            output, chain = elqr_multi_hop(q)
            if any(ans in output.lower() for ans in aliases):
                case_pass, best_output, best_chain = True, output, chain
                break
            best_output, best_chain = output, chain
        
        if case_pass: multi_hop_correct += 1
        multi_hop_total += 1
        status = "✅" if case_pass else "❌"
        print(f"  {status} Case {case_id}: {' -> '.join(best_chain)}")

    print(f"\n  FINAL SCORE: {multi_hop_correct}/{multi_hop_total} ({multi_hop_correct/max(multi_hop_total,1)*100:.1f}%)")

if __name__ == "__main__":
    run_benchmark()
