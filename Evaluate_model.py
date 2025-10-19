import sys
import re
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# ---------- CONFIG ----------
DEFAULT_LABELED_CSV = "ml_insurance_challenge_labeled_mpnet.csv"
THRESHOLD_SWEEP = np.linspace(0.1, 0.75, 31)  # thresholds to evaluate
TARGET_TOP_K = 3                              # your desired top-k
COHERENCE_EMBEDDINGS = False                  # set True to compute embedding coherence (requires sentence-transformers)
COHERENCE_SAMPLE_PER_LABEL = 50
# ----------------------------

def parse_scores(s):
    if not isinstance(s, str) or not s.strip():
        return []
    cleaned = re.sub(r'[\[\]\']', '', s)
    parts = re.split(r'[;,]', cleaned)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except:
            continue
    return out

def parse_labels(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [lbl.strip() for lbl in re.split(r';', s) if lbl.strip()]

def load_df(path):
    return pd.read_csv(path, dtype=str, on_bad_lines='skip').fillna('')

def threshold_sweep_from_scores(parsed_scores, thresholds=THRESHOLD_SWEEP):
    n = len(parsed_scores)
    results = []
    for t in thresholds:
        kept_companies = 0
        total_labels = 0
        for scs in parsed_scores:
            kept = [s for s in scs if s >= t]
            if kept:
                kept_companies += 1
                total_labels += len(kept)
        coverage_t = kept_companies / n
        avg_labels_t = total_labels / n
        results.append((t, coverage_t, avg_labels_t))
    return pd.DataFrame(results, columns=['threshold','coverage','avg_labels'])

def adaptive_threshold_assignments(parsed_scores, k=TARGET_TOP_K):
    assigned_counts = []
    adaptive_thresholds = []
    for scs in parsed_scores:
        scs_sorted = sorted(scs, reverse=True)
        if len(scs_sorted) >= k:
            thr = scs_sorted[k-1]
            adaptive_thresholds.append(thr)
            assigned_counts.append(sum(1 for s in scs if s >= thr))
        else:
            thr = scs_sorted[-1] if scs_sorted else 0.0
            adaptive_thresholds.append(thr)
            assigned_counts.append(len([s for s in scs if s >= thr]))
    return adaptive_thresholds, assigned_counts

def per_label_stats(parsed_labels, parsed_scores):
    label_counter = Counter()
    score_acc = defaultdict(list)
    for labs, scs in zip(parsed_labels, parsed_scores):
        for i, lab in enumerate(labs):
            label_counter[lab] += 1
            if i < len(scs):
                try:
                    score_acc[lab].append(float(scs[i]))
                except:
                    pass
    rows = []
    for lab, cnt in label_counter.items():
        scores = np.array(score_acc.get(lab, [])) if score_acc.get(lab) else np.array([])
        rows.append({
            'label': lab,
            'count': cnt,
            'avg_score': float(np.mean(scores)) if scores.size else None,
            'median_score': float(np.median(scores)) if scores.size else None,
            'std_score': float(np.std(scores, ddof=1)) if scores.size>1 else None,
            'p25': float(np.percentile(scores,25)) if scores.size else None,
            'p75': float(np.percentile(scores,75)) if scores.size else None
        })
    df_stats = pd.DataFrame(rows).sort_values('count', ascending=False)
    return df_stats

if __name__ == "__main__":
    labeled_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LABELED_CSV
    print("Loading labeled CSV:", labeled_path)
    df = load_df(labeled_path)

    n = len(df)
    labels_series = df.get('insurance_labels', pd.Series(['']*n))
    scores_series = df.get('insurance_label_scores', pd.Series(['']*n))

    parsed_labels = labels_series.apply(parse_labels)
    parsed_scores = scores_series.apply(parse_scores)

    coverage = (labels_series.str.strip() != '').mean()
    avg_labels = parsed_labels.apply(len).mean()
    label_count_dist = parsed_labels.apply(len).value_counts().sort_index().to_dict()

    all_scores = [s for row in parsed_scores for s in row]
    mean_score = np.mean(all_scores) if all_scores else None
    median_score = np.median(all_scores) if all_scores else None
    std_score = np.std(all_scores, ddof=1) if len(all_scores) > 1 else None
    score_min = min(all_scores) if all_scores else None
    score_max = max(all_scores) if all_scores else None

    print(f"Rows    : {n}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Avg labels/company: {avg_labels:.2f}")
    print(f"Label count distribution (counts -> #companies): {label_count_dist}")
    if all_scores:
        print(f"Scores — mean: {mean_score:.3f}, median: {median_score:.3f}, std: {std_score:.3f}, range: {score_min:.3f}-{score_max:.3f}")

    # per-label stats
    per_label_df = per_label_stats(parsed_labels, parsed_scores)
    per_label_df.to_csv("per_label_stats_auto.csv", index=False)
    print(f"Saved per-label stats to per_label_stats_auto.csv (unique labels: {len(per_label_df)})")

    # co-occurrence
    cooc = Counter()
    for labs in parsed_labels:
        unique = sorted(set(labs))
        for i in range(len(unique)):
            for j in range(i+1, len(unique)):
                cooc[(unique[i], unique[j])] += 1
    top_cooc = cooc.most_common(50)
    with open("top_cooccurrences.csv", "w", encoding="utf-8") as f:
        f.write("label_a,label_b,count\n")
        for (a,b),c in top_cooc:
            f.write(f'"{a}","{b}",{c}\n')
    print("Saved top_cooccurrences.csv")

    # threshold sweep
    sweep_df = threshold_sweep_from_scores(parsed_scores)
    sweep_df.to_csv("threshold_sweep_auto.csv", index=False)
    print("Saved threshold sweep to threshold_sweep_auto.csv")

    # top-k enforcement
    counts = parsed_labels.apply(len)
    num_at_least_k = (counts >= TARGET_TOP_K).sum()
    num_less_than_k = n - num_at_least_k
    print(f"Top-k check (target k={TARGET_TOP_K}): companies with >=k labels: {num_at_least_k} ({num_at_least_k/n:.2%}), <k: {num_less_than_k} ({num_less_than_k/n:.2%})")

    # adaptive preview
    adaptive_thrs, adaptive_counts = adaptive_threshold_assignments(parsed_scores, TARGET_TOP_K)
    adaptive_df = pd.DataFrame({'adaptive_threshold': adaptive_thrs, 'assigned_count_with_adaptive': adaptive_counts})
    adaptive_df.to_csv("adaptive_counts_preview.csv", index=False)
    print("Saved adaptive_counts_preview.csv — per-row adaptive threshold and resulting assigned count")

    # optional coherence (disabled by default)
    if COHERENCE_EMBEDDINGS:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')
            print("Computing label coherence using embeddings (this may take time)...")
            df_text = df.get('combined_text', pd.Series(['']*n))
            label_texts = defaultdict(list)
            for labs, text in zip(parsed_labels, df_text):
                for lab in labs:
                    label_texts[lab].append(text)
            coherence_rows = []
            for lab, texts in label_texts.items():
                sample_texts = texts[:COHERENCE_SAMPLE_PER_LABEL]
                if len(sample_texts) < 2:
                    continue
                embs = model.encode(sample_texts, show_progress_bar=False, convert_to_numpy=True)
                sims = np.dot(embs, embs.T)
                norms = np.linalg.norm(embs, axis=1)
                sims = sims / (norms[:,None] * norms[None,:] + 1e-12)
                iu = np.triu_indices_from(sims, k=1)
                vals = sims[iu]
                coherence_rows.append({'label': lab, 'n_samples': len(sample_texts), 'mean_pairwise_sim': float(np.mean(vals))})
            pd.DataFrame(coherence_rows).sort_values('mean_pairwise_sim', ascending=False).to_csv("label_coherence.csv", index=False)
            print("Saved label_coherence.csv")
        except Exception as e:
            print("Embedding coherence failed:", e)

    # write concise summary
    with open("auto_evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Rows: {n}\n")
        f.write(f"Coverage: {coverage:.2%}\n")
        f.write(f"Avg labels/company: {avg_labels:.2f}\n")
        if all_scores:
            f.write(f"Scores — mean: {mean_score:.3f}, median: {median_score:.3f}, std: {std_score:.3f}, range: {score_min:.3f}-{score_max:.3f}\n")
        f.write(f"Top-k target: {TARGET_TOP_K}, >=k: {num_at_least_k} ({num_at_least_k/n:.2%}), <k: {num_less_than_k} ({num_less_than_k/n:.2%})\n")
        f.write("Top co-occurrences saved to top_cooccurrences.csv\n")
    print("Auto evaluation finished. Summary written to auto_evaluation_summary.txt")