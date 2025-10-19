"""
Full pipeline combining preprocessing, embedding, ensemble assignment, pruning,
calibration heuristic, adaptive top-k, threshold search and automatic evaluation.

Run as:
  python full_pipeline.py --steps all
Or run specific steps:
  --steps normalize,weight,embed,assign,stats,prune,calibrate,adaptive,threshold,evaluate

Dependencies:
  pandas numpy sentence-transformers scikit-learn joblib
Optional:
  faiss (for ANN), torch (used by sentence-transformers)
"""
import os
import re
import argparse
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# optional heavy imports are inside functions to avoid import overhead when not used

# ---------- CONFIG ----------
IN_CLEAN = "ml_insurance_challenge_cleaned.csv"
OUT_NORM = "ml_insurance_challenge_preprocessed_norm.csv"
OUT_WEIGHTED = "ml_insurance_challenge_preprocessed_weighted.csv"
COMP_EMB_NPY = "company_emb.npy"
LABEL_EMB_NPY = "label_emb.npy"
TAXONOMY_CSV = "insurance_taxonomy.csv"
EMBED_MODEL = "all-mpnet-base-v2"    # change if needed
EMBED_BATCH = 64
ALPHA = 0.75
TOP_K = 3
MIN_SEM = 0.30
PRUNE_MIN_COUNT = 200
PRUNE_MIN_AVG = 0.45
CALIB_MIN_POS = 0.75
CALIB_MIN_NEG = 0.25
# ----------------------------

# ------- Utilities -------
def normalize_text(s):
    s = str(s or "").strip().lower()
    s = re.sub(r'[\u2018\u2019\u201c\u201d]', "'", s)
    s = re.sub(r'[^a-z0-9\|\;\,\s\-\_/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_scores_str(s):
    if not isinstance(s, str) or not s.strip(): return []
    cleaned = re.sub(r'[\[\]\']', '', s)
    parts = [p.strip() for p in re.split(r'[;,]', cleaned) if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            continue
    return out

def parse_labels_str(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [lbl.strip() for lbl in s.split(';') if lbl.strip()]

# ------- Step 1: normalize & canonicalize -------
def step_normalize(synonyms_csv=None):
    df = pd.read_csv(IN_CLEAN, dtype=str).fillna('')
    syn_map = {}
    if synonyms_csv and os.path.exists(synonyms_csv):
        syn_df = pd.read_csv(synonyms_csv, dtype=str).fillna('')
        syn_map = {str(row[0]).strip().lower(): str(row[1]).strip().lower() for _, row in syn_df.iterrows()}
    def canonicalize_tags(tag_str):
        if not tag_str: return ''
        parts = re.split(r'[;,]', tag_str)
        out = []
        for p in parts:
            t = normalize_text(p)
            if not t: continue
            t = syn_map.get(t, t)
            out.append(t)
        seen=set(); final=[]
        for x in out:
            if x not in seen:
                seen.add(x); final.append(x)
        return '; '.join(final)
    df['business_tags_norm'] = df['business_tags'].apply(canonicalize_tags)
    df['combined_text'] = df.apply(lambda r: ' | '.join(
        [c for c in [
            normalize_text(r.get('description','')),
            r.get('business_tags_norm',''),
            normalize_text(r.get('sector','')),
            normalize_text(r.get('category','')),
            normalize_text(r.get('niche',''))
        ] if c]), axis=1)
    df.to_csv(OUT_NORM, index=False)
    print("Saved normalized file:", OUT_NORM)
    return OUT_NORM

# ------- Step 2: tag weighting -------
def step_weight_tags(weight=3, max_dup=6):
    df = pd.read_csv(OUT_NORM, dtype=str).fillna('')
    def weight_tags_row(row):
        tags = row.get('business_tags_norm','')
        base = row.get('combined_text','')
        if not tags: return base
        toks = [t.strip() for t in tags.split(';') if t.strip()]
        toks_dup=[]
        for t in toks:
            dup = min(weight, max_dup)
            toks_dup += [t]*dup
        weighted = ' '.join(toks_dup)
        return (weighted + ' | ' + base).strip()
    df['combined_text_weighted'] = df.apply(weight_tags_row, axis=1)
    df.to_csv(OUT_WEIGHTED, index=False)
    print("Saved weighted file:", OUT_WEIGHTED)
    return OUT_WEIGHTED

# ------- Step 3: embed & cache -------
def step_embed_and_cache(model_name=EMBED_MODEL, batch_size=EMBED_BATCH, in_csv=OUT_WEIGHTED):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers to run embeddings: pip install sentence-transformers") from e
    df = pd.read_csv(in_csv, dtype=str).fillna('')
    tax = pd.read_csv(TAXONOMY_CSV, dtype=str).fillna('')
    labels = tax['label'].astype(str).tolist()
    model = SentenceTransformer(model_name)
    texts = df['combined_text_weighted'].astype(str).tolist()
    print("Embedding companies...")
    comp_emb = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    np.save(COMP_EMB_NPY, comp_emb)
    print("Saved company embeddings:", COMP_EMB_NPY)
    print("Embedding labels...")
    lab_emb = model.encode(labels, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    np.save(LABEL_EMB_NPY, lab_emb)
    print("Saved label embeddings:", LABEL_EMB_NPY)
    return COMP_EMB_NPY, LABEL_EMB_NPY

# ------- Step 4: ensemble assignment (semantic + tfidf) -------
def step_assign_ensemble(alpha=ALPHA, top_k=TOP_K, in_csv=OUT_WEIGHTED, out_csv="ml_insurance_challenge_labeled_ensemble.csv", emb_company=COMP_EMB_NPY, emb_label=LABEL_EMB_NPY):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    df = pd.read_csv(in_csv, dtype=str).fillna('')
    tax = pd.read_csv(TAXONOMY_CSV, dtype=str).fillna('')
    labels = tax['label'].astype(str).tolist()
    # load embeddings
    comp_emb = np.load(emb_company)
    label_emb = np.load(emb_label)
    # TF-IDF
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    vec.fit(df['combined_text_weighted'].astype(str).tolist() + labels)
    company_tfidf = vec.transform(df['combined_text_weighted'].astype(str).tolist())
    label_tfidf = vec.transform(labels)
    out_labels=[]; out_scores=[]; adaptive_thr=[]
    batch_size = 512
    n = len(df)
    for i in range(0, n, batch_size):
        sem = cosine_similarity(comp_emb[i:i+batch_size], label_emb)
        lex = cosine_similarity(company_tfidf[i:i+batch_size], label_tfidf)
        # normalize per-row to 0-1 to combine
        def norm_rows(M):
            M = np.array(M, dtype=float)
            mins = M.min(axis=1, keepdims=True)
            maxs = M.max(axis=1, keepdims=True)
            denom = (maxs - mins); denom[denom==0]=1.0
            return (M - mins) / denom
        sem_n = norm_rows(sem)
        lex_n = norm_rows(lex)
        combined = alpha * sem_n + (1-alpha) * lex_n
        for row_idx in range(combined.shape[0]):
            row = combined[row_idx]
            idx_sorted = np.argsort(row)[::-1]
            top_idx = idx_sorted[:top_k]
            labels_sel = [labels[j] for j in top_idx]
            scores_sel = [float(row[j]) for j in top_idx]
            out_labels.append('; '.join(labels_sel))
            out_scores.append('; '.join([f"{s:.3f}" for s in scores_sel]))
            adaptive_thr.append(float(scores_sel[-1] if scores_sel else 0.0))
    df['insurance_labels'] = out_labels
    df['insurance_label_scores'] = out_scores
    df['adaptive_threshold'] = adaptive_thr
    df.to_csv(out_csv, index=False)
    print("Saved ensemble labeled CSV:", out_csv)
    return out_csv

# ------- Step 5: per-label stats & prune candidates -------
def step_label_stats(labeled_csv, per_label_out="per_label_stats_auto.csv"):
    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    parsed_labels = df['insurance_labels'].apply(parse_labels_str)
    parsed_scores = df['insurance_label_scores'].apply(parse_scores_str)
    label_counter = Counter()
    score_acc = defaultdict(list)
    for labs, scs in zip(parsed_labels, parsed_scores):
        for i, lab in enumerate(labs):
            label_counter[lab] += 1
            if i < len(scs):
                score_acc[lab].append(scs[i])
    rows=[]
    for lab,cnt in label_counter.items():
        scores = np.array(score_acc.get(lab,[])) if score_acc.get(lab) else np.array([])
        rows.append({
            'label': lab,
            'count': cnt,
            'avg_score': float(np.mean(scores)) if scores.size else None,
            'std_score': float(np.std(scores, ddof=1)) if scores.size>1 else None
        })
    per_label_df = pd.DataFrame(rows).sort_values('count', ascending=False)
    per_label_df.to_csv(per_label_out, index=False)
    print("Saved per-label stats:", per_label_out)
    # prune candidates
    to_prune = per_label_df[(per_label_df['count']>=PRUNE_MIN_COUNT) & (per_label_df['avg_score'].fillna(0) < PRUNE_MIN_AVG)]['label'].tolist()
    pd.Series(to_prune).to_csv("labels_to_prune.csv", index=False)
    print(f"Saved labels_to_prune.csv ({len(to_prune)} labels)")
    return per_label_out, "labels_to_prune.csv"

# ------- Step 6: calibrator training (heuristic pseudo-labels) -------
def step_train_calibrator(labeled_csv, out_joblib="calibrator.joblib"):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        import joblib
    except Exception as e:
        raise RuntimeError("Install scikit-learn and joblib to run calibrator") from e

    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    entries = []
    for idx, r in df.iterrows():
        scores = parse_scores_str(r.get('insurance_label_scores',''))
        if not scores:
            continue
        top = scores[0]
        feat = [top, len(scores), len(str(r.get('business_tags','')))]
        entries.append((idx, top, feat))

    if not entries:
        print("No scored entries available for calibrator training.")
        return None

    import numpy as _np
    tops = _np.array([e[1] for e in entries])
    feats = _np.array([e[2] for e in entries])

    pos_mask = tops >= CALIB_MIN_POS
    neg_mask = tops <= CALIB_MIN_NEG

    # if no strict negatives, use bottom-20% as negatives
    if neg_mask.sum() == 0:
        pct20 = float(_np.percentile(tops, 20))
        neg_mask = tops <= pct20
        print(f"No strict negatives found; using bottom-20% cutoff {pct20:.3f} for negatives (count={neg_mask.sum()})")

    X = []
    y = []
    for i, m in enumerate(pos_mask):
        if m:
            X.append(feats[i])
            y.append(1)
    for i, m in enumerate(neg_mask):
        if m:
            X.append(feats[i])
            y.append(0)

    if len(X) < 200:
        print(f"Not enough extreme pseudo-labeled samples (found {len(X)}). Need >=200 to train calibrator.")
        return None

    y_unique = set(y)
    if len(y_unique) < 2:
        print(f"Insufficient class diversity for calibrator (classes found: {y_unique}). Skipping training.")
        return None

    X = _np.array(X); y = _np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    except ValueError as ve:
        print("Training failed:", ve)
        return None

    try:
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
        print("Calibrator trained. AUC (approx):", auc)
    except Exception:
        print("Calibrator trained. (AUC unavailable)")

    joblib.dump(clf, out_joblib)
    print("Saved calibrator:", out_joblib)
    return out_joblib

# ------- Step 7: adaptive top-k enforcement with safeguards -------
def step_assign_adaptive_topk(in_pre=OUT_WEIGHTED, out_csv="ml_insurance_challenge_labeled_adaptive_topk.csv", top_k=TOP_K, min_sem=MIN_SEM, score_gap=0.15):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers and scikit-learn to run adaptive assignment") from e
    df = pd.read_csv(in_pre, dtype=str).fillna('')
    tax = pd.read_csv(TAXONOMY_CSV, dtype=str).fillna('')
    labels = tax['label'].astype(str).tolist()
    # prefer cached embeddings
    if os.path.exists(COMP_EMB_NPY) and os.path.exists(LABEL_EMB_NPY):
        comp_emb = np.load(COMP_EMB_NPY)
        label_emb = np.load(LABEL_EMB_NPY)
    else:
        model = SentenceTransformer(EMBED_MODEL)
        comp_emb = model.encode(df['combined_text_weighted'].astype(str).tolist(), show_progress_bar=True, convert_to_numpy=True)
        label_emb = model.encode(labels, show_progress_bar=True, convert_to_numpy=True)
    out_labels=[]; out_scores=[]; out_thr=[]
    for i in range(len(df)):
        sims = np.dot(comp_emb[i], label_emb.T)
        # if not normalized, convert to cosine by dividing norms
        # assume embeddings are reasonably comparable; fallback to argsort
        idxs = np.argsort(sims)[::-1]
        scores_sorted = sims[idxs]
        kth = scores_sorted[top_k-1] if len(scores_sorted) >= top_k else (scores_sorted[-1] if len(scores_sorted)>0 else 0.0)
        thr = max(kth, min_sem)
        sel_idx = [j for j,s in zip(idxs, scores_sorted) if s >= thr]
        if len(sel_idx) < top_k:
            sel_idx = idxs[:top_k].tolist()
        # apply score gap relative to best
        best = sims[idxs[0]] if len(idxs)>0 else 0.0
        sel_idx = [j for j in sel_idx if best - sims[j] <= score_gap]
        if len(sel_idx) < top_k:
            sel_idx = idxs[:top_k].tolist()
        selected = [(labels[j], float(sims[j])) for j in sel_idx]
        out_labels.append('; '.join([l for l,_ in selected]))
        out_scores.append('; '.join([f"{s:.3f}" for _,s in selected]))
        out_thr.append(float(thr))
    df['insurance_labels'] = out_labels
    df['insurance_label_scores'] = out_scores
    df['adaptive_threshold'] = out_thr
    df.to_csv(out_csv, index=False)
    print("Saved adaptive top-k CSV:", out_csv)
    return out_csv

# ------- Step 8: largest threshold / top-k threshold candidates -------
def step_find_largest_threshold(labeled_csv, target_k=TOP_K, out_csv="topk_threshold_candidates.csv"):
    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    parsed = df['insurance_label_scores'].apply(parse_scores_str)
    thresholds = np.linspace(0.1, 0.75, 65)
    results=[]
    n = len(parsed)
    for t in thresholds:
        num_with_at_least_k = sum(1 for scs in parsed if len([s for s in scs if s >= t]) >= target_k)
        frac = num_with_at_least_k / n
        results.append((t, frac, num_with_at_least_k))
    res_df = pd.DataFrame(results, columns=['threshold','fraction_with_>=k','count_with_>=k'])
    res_df.to_csv(out_csv, index=False)
    print("Saved threshold candidates:", out_csv)
    for target in [1.0, 0.99, 0.95, 0.90, 0.80]:
        eligible = res_df[res_df['fraction_with_>=k'] >= target]
        t_best = eligible['threshold'].max() if not eligible.empty else None
        print(f"max threshold with >={int(target*100)}% rows having >={target_k} labels: {t_best}")
    return out_csv

# ------- Step 9: automatic evaluation (summary + per-label + sweep) -------
def step_evaluate(labeled_csv, out_prefix="auto_eval"):
    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    parsed_labels = df['insurance_labels'].apply(parse_labels_str)
    parsed_scores = df['insurance_label_scores'].apply(parse_scores_str)
    n = len(df)
    coverage = (df['insurance_labels'].astype(str).str.strip() != '').mean()
    avg_labels = parsed_labels.apply(len).mean()
    all_scores = [s for row in parsed_scores for s in row]
    mean_score = float(np.mean(all_scores)) if all_scores else None
    median_score = float(np.median(all_scores)) if all_scores else None
    std_score = float(np.std(all_scores, ddof=1)) if len(all_scores)>1 else None
    print(f"Rows: {n}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Avg labels/company: {avg_labels:.2f}")
    if all_scores:
        print(f"Scores — mean: {mean_score:.3f}, median: {median_score:.3f}, std: {std_score:.3f}, range: {min(all_scores):.3f}-{max(all_scores):.3f}")
    # per-label stats
    per_label_out = out_prefix + "_per_label.csv"
    label_counter = Counter()
    score_acc = defaultdict(list)
    for labs, scs in zip(parsed_labels, parsed_scores):
        for i, lab in enumerate(labs):
            label_counter[lab] += 1
            if i < len(scs):
                score_acc[lab].append(scs[i])
    rows=[]
    for lab,cnt in label_counter.items():
        scores = np.array(score_acc.get(lab,[])) if score_acc.get(lab) else np.array([])
        rows.append({'label':lab,'count':cnt,'avg_score': float(np.mean(scores)) if scores.size else None})
    pd.DataFrame(rows).sort_values('count', ascending=False).to_csv(per_label_out, index=False)
    print("Saved per-label stats:", per_label_out)
    # threshold sweep
    thresholds = np.linspace(0.1, 0.75, 31)
    sweep=[]
    for t in thresholds:
        kept_companies = 0; total_labels = 0
        for scs in parsed_scores:
            kept = [s for s in scs if s >= t]
            if kept:
                kept_companies += 1
                total_labels += len(kept)
        sweep.append({'threshold': float(t), 'coverage': kept_companies / n, 'avg_labels': total_labels / n})
    sweep_out = out_prefix + "_threshold_sweep.csv"
    pd.DataFrame(sweep).to_csv(sweep_out, index=False)
    print("Saved threshold sweep:", sweep_out)
    # save summary
    summary_out = out_prefix + "_summary.txt"
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(f"Rows: {n}\n")
        f.write(f"Coverage: {coverage:.2%}\n")
        f.write(f"Avg labels/company: {avg_labels:.2f}\n")
        if all_scores:
            f.write(f"Scores — mean: {mean_score:.3f}, median: {median_score:.3f}, std: {std_score:.3f}\n")
    print("Saved summary:", summary_out)
    return summary_out

# ------- CLI -------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str, default="all", help="Comma-separated steps: normalize,weight,embed,assign,stats,prune,calibrate,adaptive,threshold,evaluate or all")
    parser.add_argument("--model", type=str, default=EMBED_MODEL)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--topk", type=int, default=TOP_K)
    args = parser.parse_args()
    steps = args.steps.split(',') if args.steps != "all" else ["normalize","weight","embed","assign","stats","prune","calibrate","adaptive","threshold","evaluate"]
    if "normalize" in steps:
        step_normalize()
    if "weight" in steps:
        step_weight_tags()
    if "embed" in steps:
        step_embed_and_cache(model_name=args.model)
    if "assign" in steps:
        step_assign_ensemble(alpha=args.alpha, top_k=args.topk)
    labeled = "ml_insurance_challenge_labeled_ensemble.csv"
    if "stats" in steps:
        step_label_stats(labeled)
    if "prune" in steps:
        # uses per_label_stats to produce labels_to_prune.csv
        step_label_stats(labeled)
    if "calibrate" in steps:
        step_train_calibrator(labeled)
    if "adaptive" in steps:
        step_assign_adaptive_topk()
    if "threshold" in steps:
        step_find_largest_threshold("ml_insurance_challenge_labeled_ensemble.csv")
    if "evaluate" in steps:
        step_evaluate("ml_insurance_challenge_labeled_ensemble.csv")
    print("Requested steps finished.")

if __name__ == "__main__":
    main()
# filepath: e:\Veridion_challenge#2\full_pipeline.py
"""
Full pipeline combining preprocessing, embedding, ensemble assignment, pruning,
calibration heuristic, adaptive top-k, threshold search and automatic evaluation.

Run as:
  python full_pipeline.py --steps all
Or run specific steps:
  --steps normalize,weight,embed,assign,stats,prune,calibrate,adaptive,threshold,evaluate

Dependencies:
  pandas numpy sentence-transformers scikit-learn joblib
Optional:
  faiss (for ANN), torch (used by sentence-transformers)
"""

import os
import re
import argparse
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# optional heavy imports are inside functions to avoid import overhead when not used

# ---------- CONFIG ----------
IN_CLEAN = "ml_insurance_challenge_cleaned.csv"
OUT_NORM = "ml_insurance_challenge_preprocessed_norm.csv"
OUT_WEIGHTED = "ml_insurance_challenge_preprocessed_weighted.csv"
COMP_EMB_NPY = "company_emb.npy"
LABEL_EMB_NPY = "label_emb.npy"
TAXONOMY_CSV = "insurance_taxonomy.csv"
EMBED_MODEL = "all-mpnet-base-v2"    # change if needed
EMBED_BATCH = 64
ALPHA = 0.75
TOP_K = 3
MIN_SEM = 0.30
PRUNE_MIN_COUNT = 200
PRUNE_MIN_AVG = 0.45
CALIB_MIN_POS = 0.75
CALIB_MIN_NEG = 0.25
# ----------------------------

# ------- Utilities -------
def normalize_text(s):
    s = str(s or "").strip().lower()
    s = re.sub(r'[\u2018\u2019\u201c\u201d]', "'", s)
    s = re.sub(r'[^a-z0-9\|\;\,\s\-\_/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_scores_str(s):
    if not isinstance(s, str) or not s.strip(): return []
    cleaned = re.sub(r'[\[\]\']', '', s)
    parts = [p.strip() for p in re.split(r'[;,]', cleaned) if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            continue
    return out

def parse_labels_str(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [lbl.strip() for lbl in s.split(';') if lbl.strip()]

# ------- Step 1: normalize & canonicalize -------
def step_normalize(synonyms_csv=None):
    df = pd.read_csv(IN_CLEAN, dtype=str).fillna('')
    syn_map = {}
    if synonyms_csv and os.path.exists(synonyms_csv):
        syn_df = pd.read_csv(synonyms_csv, dtype=str).fillna('')
        syn_map = {str(row[0]).strip().lower(): str(row[1]).strip().lower() for _, row in syn_df.iterrows()}
    def canonicalize_tags(tag_str):
        if not tag_str: return ''
        parts = re.split(r'[;,]', tag_str)
        out = []
        for p in parts:
            t = normalize_text(p)
            if not t: continue
            t = syn_map.get(t, t)
            out.append(t)
        seen=set(); final=[]
        for x in out:
            if x not in seen:
                seen.add(x); final.append(x)
        return '; '.join(final)
    df['business_tags_norm'] = df['business_tags'].apply(canonicalize_tags)
    df['combined_text'] = df.apply(lambda r: ' | '.join(
        [c for c in [
            normalize_text(r.get('description','')),
            r.get('business_tags_norm',''),
            normalize_text(r.get('sector','')),
            normalize_text(r.get('category','')),
            normalize_text(r.get('niche',''))
        ] if c]), axis=1)
    df.to_csv(OUT_NORM, index=False)
    print("Saved normalized file:", OUT_NORM)
    return OUT_NORM

# ------- Step 2: tag weighting -------
def step_weight_tags(weight=3, max_dup=6):
    df = pd.read_csv(OUT_NORM, dtype=str).fillna('')
    def weight_tags_row(row):
        tags = row.get('business_tags_norm','')
        base = row.get('combined_text','')
        if not tags: return base
        toks = [t.strip() for t in tags.split(';') if t.strip()]
        toks_dup=[]
        for t in toks:
            dup = min(weight, max_dup)
            toks_dup += [t]*dup
        weighted = ' '.join(toks_dup)
        return (weighted + ' | ' + base).strip()
    df['combined_text_weighted'] = df.apply(weight_tags_row, axis=1)
    df.to_csv(OUT_WEIGHTED, index=False)
    print("Saved weighted file:", OUT_WEIGHTED)
    return OUT_WEIGHTED

# ------- Step 3: embed & cache -------
def step_embed_and_cache(model_name=EMBED_MODEL, batch_size=EMBED_BATCH, in_csv=OUT_WEIGHTED):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers to run embeddings: pip install sentence-transformers") from e
    df = pd.read_csv(in_csv, dtype=str).fillna('')
    tax = pd.read_csv(TAXONOMY_CSV, dtype=str).fillna('')
    labels = tax['label'].astype(str).tolist()
    model = SentenceTransformer(model_name)
    texts = df['combined_text_weighted'].astype(str).tolist()
    print("Embedding companies...")
    comp_emb = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    np.save(COMP_EMB_NPY, comp_emb)
    print("Saved company embeddings:", COMP_EMB_NPY)
    print("Embedding labels...")
    lab_emb = model.encode(labels, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    np.save(LABEL_EMB_NPY, lab_emb)
    print("Saved label embeddings:", LABEL_EMB_NPY)
    return COMP_EMB_NPY, LABEL_EMB_NPY

# ------- Step 4: ensemble assignment (semantic + tfidf) -------
def step_assign_ensemble(alpha=ALPHA, top_k=TOP_K, in_csv=OUT_WEIGHTED, out_csv="ml_insurance_challenge_labeled_ensemble.csv", emb_company=COMP_EMB_NPY, emb_label=LABEL_EMB_NPY):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    df = pd.read_csv(in_csv, dtype=str).fillna('')
    tax = pd.read_csv(TAXONOMY_CSV, dtype=str).fillna('')
    labels = tax['label'].astype(str).tolist()
    # load embeddings
    comp_emb = np.load(emb_company)
    label_emb = np.load(emb_label)
    # TF-IDF
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    vec.fit(df['combined_text_weighted'].astype(str).tolist() + labels)
    company_tfidf = vec.transform(df['combined_text_weighted'].astype(str).tolist())
    label_tfidf = vec.transform(labels)
    out_labels=[]; out_scores=[]; adaptive_thr=[]
    batch_size = 512
    n = len(df)
    for i in range(0, n, batch_size):
        sem = cosine_similarity(comp_emb[i:i+batch_size], label_emb)
        lex = cosine_similarity(company_tfidf[i:i+batch_size], label_tfidf)
        # normalize per-row to 0-1 to combine
        def norm_rows(M):
            M = np.array(M, dtype=float)
            mins = M.min(axis=1, keepdims=True)
            maxs = M.max(axis=1, keepdims=True)
            denom = (maxs - mins); denom[denom==0]=1.0
            return (M - mins) / denom
        sem_n = norm_rows(sem)
        lex_n = norm_rows(lex)
        combined = alpha * sem_n + (1-alpha) * lex_n
        for row_idx in range(combined.shape[0]):
            row = combined[row_idx]
            idx_sorted = np.argsort(row)[::-1]
            top_idx = idx_sorted[:top_k]
            labels_sel = [labels[j] for j in top_idx]
            scores_sel = [float(row[j]) for j in top_idx]
            out_labels.append('; '.join(labels_sel))
            out_scores.append('; '.join([f"{s:.3f}" for s in scores_sel]))
            adaptive_thr.append(float(scores_sel[-1] if scores_sel else 0.0))
    df['insurance_labels'] = out_labels
    df['insurance_label_scores'] = out_scores
    df['adaptive_threshold'] = adaptive_thr
    df.to_csv(out_csv, index=False)
    print("Saved ensemble labeled CSV:", out_csv)
    return out_csv

# ------- Step 5: per-label stats & prune candidates -------
def step_label_stats(labeled_csv, per_label_out="per_label_stats_auto.csv"):
    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    parsed_labels = df['insurance_labels'].apply(parse_labels_str)
    parsed_scores = df['insurance_label_scores'].apply(parse_scores_str)
    label_counter = Counter()
    score_acc = defaultdict(list)
    for labs, scs in zip(parsed_labels, parsed_scores):
        for i, lab in enumerate(labs):
            label_counter[lab] += 1
            if i < len(scs):
                score_acc[lab].append(scs[i])
    rows=[]
    for lab,cnt in label_counter.items():
        scores = np.array(score_acc.get(lab,[])) if score_acc.get(lab) else np.array([])
        rows.append({
            'label': lab,
            'count': cnt,
            'avg_score': float(np.mean(scores)) if scores.size else None,
            'std_score': float(np.std(scores, ddof=1)) if scores.size>1 else None
        })
    per_label_df = pd.DataFrame(rows).sort_values('count', ascending=False)
    per_label_df.to_csv(per_label_out, index=False)
    print("Saved per-label stats:", per_label_out)
    # prune candidates
    to_prune = per_label_df[(per_label_df['count']>=PRUNE_MIN_COUNT) & (per_label_df['avg_score'].fillna(0) < PRUNE_MIN_AVG)]['label'].tolist()
    pd.Series(to_prune).to_csv("labels_to_prune.csv", index=False)
    print(f"Saved labels_to_prune.csv ({len(to_prune)} labels)")
    return per_label_out, "labels_to_prune.csv"

# ------- Step 6: calibrator training (heuristic pseudo-labels) -------
def step_train_calibrator(labeled_csv, out_joblib="calibrator.joblib"):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        import joblib
    except Exception as e:
        raise RuntimeError("Install scikit-learn and joblib to run calibrator") from e

    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    # collect rows that have at least one score
    entries = []
    for idx, r in df.iterrows():
        scores = parse_scores_str(r.get('insurance_label_scores',''))
        if not scores:
            continue
        top = scores[0]
        feat = [top, len(scores), len(str(r.get('business_tags','')))]
        entries.append((idx, top, feat))

    if not entries:
        print("No scored entries available for calibrator training.")
        return None

    # build arrays
    import numpy as _np
    idxs = [e[0] for e in entries]
    tops = _np.array([e[1] for e in entries])
    feats = _np.array([e[2] for e in entries])

    # select positives and negatives using configured thresholds
    pos_mask = tops >= CALIB_MIN_POS
    neg_mask = tops <= CALIB_MIN_NEG

    # If negatives are missing, expand negative selection to bottom 20% of tops
    if neg_mask.sum() == 0:
        pct20 = _np.percentile(tops, 20)
        neg_mask = tops <= pct20
        print(f"No strict negatives found; using bottom-20% cutoff {pct20:.3f} for negatives (count={neg_mask.sum()})")

    X = []
    y = []
    # collect pos examples
    for i, m in enumerate(pos_mask):
        if m:
            X.append(feats[i])
            y.append(1)
    # collect neg examples
    for i, m in enumerate(neg_mask):
        if m:
            X.append(feats[i])
            y.append(0)

    if len(X) < 200:
        print(f"Not enough extreme pseudo-labeled samples (found {len(X)}). Need >=200 to train calibrator.")
        return None

    y_unique = set(y)
    if len(y_unique) < 2:
        print(f"Insufficient class diversity for calibrator (classes found: {y_unique}). Skipping training.")
        return None

    X = _np.array(X); y = _np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000).fit(Xtr,ytr)
    try:
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
        print("Calibrator trained. AUC (approx):", auc)
    except Exception:
        print("Calibrator trained. (AUC unavailable)")

    joblib.dump(clf, out_joblib)
    print("Saved calibrator:", out_joblib)
    return out_joblib
# ------- Step 7: adaptive top-k enforcement with safeguards -------
def step_assign_adaptive_topk(in_pre=OUT_WEIGHTED, out_csv="ml_insurance_challenge_labeled_adaptive_topk.csv", top_k=TOP_K, min_sem=MIN_SEM, score_gap=0.15):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers and scikit-learn to run adaptive assignment") from e
    df = pd.read_csv(in_pre, dtype=str).fillna('')
    tax = pd.read_csv(TAXONOMY_CSV, dtype=str).fillna('')
    labels = tax['label'].astype(str).tolist()
    # prefer cached embeddings
    if os.path.exists(COMP_EMB_NPY) and os.path.exists(LABEL_EMB_NPY):
        comp_emb = np.load(COMP_EMB_NPY)
        label_emb = np.load(LABEL_EMB_NPY)
    else:
        model = SentenceTransformer(EMBED_MODEL)
        comp_emb = model.encode(df['combined_text_weighted'].astype(str).tolist(), show_progress_bar=True, convert_to_numpy=True)
        label_emb = model.encode(labels, show_progress_bar=True, convert_to_numpy=True)
    out_labels=[]; out_scores=[]; out_thr=[]
    for i in range(len(df)):
        sims = np.dot(comp_emb[i], label_emb.T)
        # if not normalized, convert to cosine by dividing norms
        # assume embeddings are reasonably comparable; fallback to argsort
        idxs = np.argsort(sims)[::-1]
        scores_sorted = sims[idxs]
        kth = scores_sorted[top_k-1] if len(scores_sorted) >= top_k else (scores_sorted[-1] if len(scores_sorted)>0 else 0.0)
        thr = max(kth, min_sem)
        sel_idx = [j for j,s in zip(idxs, scores_sorted) if s >= thr]
        if len(sel_idx) < top_k:
            sel_idx = idxs[:top_k].tolist()
        # apply score gap relative to best
        best = sims[idxs[0]] if len(idxs)>0 else 0.0
        sel_idx = [j for j in sel_idx if best - sims[j] <= score_gap]
        if len(sel_idx) < top_k:
            sel_idx = idxs[:top_k].tolist()
        selected = [(labels[j], float(sims[j])) for j in sel_idx]
        out_labels.append('; '.join([l for l,_ in selected]))
        out_scores.append('; '.join([f"{s:.3f}" for _,s in selected]))
        out_thr.append(float(thr))
    df['insurance_labels'] = out_labels
    df['insurance_label_scores'] = out_scores
    df['adaptive_threshold'] = out_thr
    df.to_csv(out_csv, index=False)
    print("Saved adaptive top-k CSV:", out_csv)
    return out_csv

# ------- Step 8: largest threshold / top-k threshold candidates -------
def step_find_largest_threshold(labeled_csv, target_k=TOP_K, out_csv="topk_threshold_candidates.csv"):
    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    parsed = df['insurance_label_scores'].apply(parse_scores_str)
    thresholds = np.linspace(0.1, 0.75, 65)
    results=[]
    n = len(parsed)
    for t in thresholds:
        num_with_at_least_k = sum(1 for scs in parsed if len([s for s in scs if s >= t]) >= target_k)
        frac = num_with_at_least_k / n
        results.append((t, frac, num_with_at_least_k))
    res_df = pd.DataFrame(results, columns=['threshold','fraction_with_>=k','count_with_>=k'])
    res_df.to_csv(out_csv, index=False)
    print("Saved threshold candidates:", out_csv)
    for target in [1.0, 0.99, 0.95, 0.90, 0.80]:
        eligible = res_df[res_df['fraction_with_>=k'] >= target]
        t_best = eligible['threshold'].max() if not eligible.empty else None
        print(f"max threshold with >={int(target*100)}% rows having >={target_k} labels: {t_best}")
    return out_csv

# ------- Step 9: automatic evaluation (summary + per-label + sweep) -------
def step_evaluate(labeled_csv, out_prefix="auto_eval"):
    df = pd.read_csv(labeled_csv, dtype=str).fillna('')
    parsed_labels = df['insurance_labels'].apply(parse_labels_str)
    parsed_scores = df['insurance_label_scores'].apply(parse_scores_str)
    n = len(df)
    coverage = (df['insurance_labels'].astype(str).str.strip() != '').mean()
    avg_labels = parsed_labels.apply(len).mean()
    all_scores = [s for row in parsed_scores for s in row]
    mean_score = float(np.mean(all_scores)) if all_scores else None
    median_score = float(np.median(all_scores)) if all_scores else None
    std_score = float(np.std(all_scores, ddof=1)) if len(all_scores)>1 else None
    print(f"Rows: {n}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Avg labels/company: {avg_labels:.2f}")
    if all_scores:
        print(f"Scores — mean: {mean_score:.3f}, median: {median_score:.3f}, std: {std_score:.3f}, range: {min(all_scores):.3f}-{max(all_scores):.3f}")
    # per-label stats
    per_label_out = out_prefix + "_per_label.csv"
    label_counter = Counter()
    score_acc = defaultdict(list)
    for labs, scs in zip(parsed_labels, parsed_scores):
        for i, lab in enumerate(labs):
            label_counter[lab] += 1
            if i < len(scs):
                score_acc[lab].append(scs[i])
    rows=[]
    for lab,cnt in label_counter.items():
        scores = np.array(score_acc.get(lab,[])) if score_acc.get(lab) else np.array([])
        rows.append({'label':lab,'count':cnt,'avg_score': float(np.mean(scores)) if scores.size else None})
    pd.DataFrame(rows).sort_values('count', ascending=False).to_csv(per_label_out, index=False)
    print("Saved per-label stats:", per_label_out)
    # threshold sweep
    thresholds = np.linspace(0.1, 0.75, 31)
    sweep=[]
    for t in thresholds:
        kept_companies = 0; total_labels = 0
        for scs in parsed_scores:
            kept = [s for s in scs if s >= t]
            if kept:
                kept_companies += 1
                total_labels += len(kept)
        sweep.append({'threshold': float(t), 'coverage': kept_companies / n, 'avg_labels': total_labels / n})
    sweep_out = out_prefix + "_threshold_sweep.csv"
    pd.DataFrame(sweep).to_csv(sweep_out, index=False)
    print("Saved threshold sweep:", sweep_out)
    # save summary
    summary_out = out_prefix + "_summary.txt"
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(f"Rows: {n}\n")
        f.write(f"Coverage: {coverage:.2%}\n")
        f.write(f"Avg labels/company: {avg_labels:.2f}\n")
        if all_scores:
            f.write(f"Scores — mean: {mean_score:.3f}, median: {median_score:.3f}, std: {std_score:.3f}\n")
    print("Saved summary:", summary_out)
    return summary_out

# ------- CLI -------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str, default="all", help="Comma-separated steps: normalize,weight,embed,assign,stats,prune,calibrate,adaptive,threshold,evaluate or all")
    parser.add_argument("--model", type=str, default=EMBED_MODEL)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--topk", type=int, default=TOP_K)
    args = parser.parse_args()
    steps = args.steps.split(',') if args.steps != "all" else ["normalize","weight","embed","assign","stats","prune","calibrate","adaptive","threshold","evaluate"]
    if "normalize" in steps:
        step_normalize()
    if "weight" in steps:
        step_weight_tags()
    if "embed" in steps:
        step_embed_and_cache(model_name=args.model)
    if "assign" in steps:
        step_assign_ensemble(alpha=args.alpha, top_k=args.topk)
    labeled = "ml_insurance_challenge_labeled_ensemble.csv"
    if "stats" in steps:
        step_label_stats(labeled)
    if "prune" in steps:
        # uses per_label_stats to produce labels_to_prune.csv
        step_label_stats(labeled)
    if "calibrate" in steps:
        step_train_calibrator(labeled)
    if "adaptive" in steps:
        step_assign_adaptive_topk()
    if "threshold" in steps:
        step_find_largest_threshold("ml_insurance_challenge_labeled_ensemble.csv")
    if "evaluate" in steps:
        step_evaluate("ml_insurance_challenge_labeled_ensemble.csv")
    print("Requested steps finished.")

if __name__ == "__main__":
    main()