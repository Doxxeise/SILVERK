"""
SILVERK Architecture - REAL RippleEdits Benchmark (20 Cases)
PDC Entity Graph + IDF-ELQR Chaining + Multi-View Amplification

Evaluates SILVERK's multi-hop knowledge editing on real counterfactual cases from RippleEdits.
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
SAVE_DIR = "./silverk_real_ripple_20"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_CASES = 20
CHECKPOINT_EVERY = 50 

print("=" * 65)
print(" SILVERK ARCHITECTURE - REAL RIPPLE EDITS 20 BENCHMARK")
print(" PDC Entity Graph + IDF-ELQR + Multi-View + Checkpoint/Resume")
print("=" * 65)


# ==========================================
# DATASET UTILS
# ==========================================
def download_ripple_edits():
    """Download RippleEdits dataset."""
    import urllib.request
    cache_path = os.path.join(SAVE_DIR, "RippleEdits.json")
    if os.path.exists(cache_path):
        print(f"  Found cached RippleEdits at {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)
    print(f"  Downloading RippleEdits from GitHub...")
    urls = [
        "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/recent.json",
        "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/popular.json",
    ]
    data = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    data.extend(json.loads(response.read().decode()))
        except Exception as e:
            print(f"  Failed download: {url} -> {e}")
    if data:
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data
    return None


def ripple_case_to_facts(case):
    """Convert Ripple case to SILVERK training facts."""
    facts = []
    edit = case.get("edit", {})
    prompt_template = edit.get("prompt", "")
    subject = edit.get("subject_id", "")
    target_new = "the answer"
    
    fact_text = prompt_template
    qa_text = fact_text
    subject_kw = [w.lower() for w in re.findall(r'[a-zA-Z]+', subject) if len(w) > 2]
    hasher = int(math.fabs(hash(fact_text))) % 999999
    
    facts.append({
        "id": f"ripple_case_{subject}_{hasher}",
        "text": fact_text,
        "qa": qa_text,
        "subject_keywords": subject_kw,
        "target_new": target_new,
        "target_true": "",
        "subject": subject,
    })
    return facts


# ==========================================
# THE BENCHMARK
# ==========================================
def run_benchmark():
    ripple_data = download_ripple_edits()
    if ripple_data is None: return

    multi_hop_cases = []
    CRITERIA = ["Logical_Generalization", "Compositionality_I",
                "Compositionality_II", "Subject_Aliasing",
                "Relation_Specificity", "Forgetfulness"]
    for case in ripple_data:
        edit = case.get("edit", {})
        if not edit.get("prompt"): continue
        ripple_tests = []
        for crit in CRITERIA:
            for group in case.get(crit, []):
                for tq in group.get("test_queries", []):
                    prompt = tq.get("prompt", "")
                    answers = tq.get("answers", [])
                    if not prompt or not answers: continue
                    valid_answers = []
                    for a in answers:
                        if a.get("value"): valid_answers.append(a.get("value"))
                        for alias in a.get("aliases", []):
                            if alias: valid_answers.append(alias)
                    if valid_answers:
                        ripple_tests.append({"prompt": prompt, "answers": valid_answers, "type": crit})
        if ripple_tests:
            case["ripple_tests"] = ripple_tests
            case["case_id"] = edit.get("subject_id", "unknown") + str(hash(edit.get("prompt")))
            multi_hop_cases.append(case)

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
    graph_path = os.path.join(SAVE_DIR, "entity_graph.json")
    if os.path.exists(graph_path):
        with open(graph_path, "r") as f:
            entity_graph.update(json.load(f))
        print(f"  ✓ Resumed entity graph ({len(entity_graph)} nodes)")

    all_facts = []
    for case in test_cases:
        all_facts.extend(ripple_case_to_facts(case))

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

    # Phase 3: Evaluation
    ripple_correct = {c: 0 for c in CRITERIA}
    ripple_total = {c: 0 for c in CRITERIA}
    for case in test_cases:
        for test in case["ripple_tests"]:
            query, answers = test["prompt"], [a.lower() for a in test["answers"] if len(a.strip()) > 1]
            if not answers: continue
            output, _ = elqr_multi_hop(query)
            hit = any(ans in output.lower() for ans in answers)
            ripple_correct[test["type"]] += int(hit)
            ripple_total[test["type"]] += 1
        print(f"  Processed {case['case_id']}")

    print("\n  FINAL RESULTS")
    for crit in CRITERIA:
        if ripple_total[crit] > 0:
            print(f"  {crit:25}: {ripple_correct[crit]}/{ripple_total[crit]} ({ripple_correct[crit]/ripple_total[crit]*100:.1f}%)")

if __name__ == "__main__":
    run_benchmark()
