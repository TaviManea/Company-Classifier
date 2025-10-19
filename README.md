# Company-Classifier
Purpose: multi-step pipeline to assign insurance taxonomy labels to a company list (preprocess → embed → ensemble semantic+lexical → adaptive top-K → evaluate).
Key files:
entire_pipeline.py / full_pipeline.py — end-to-end orchestration (normalize, weight, embed, assign, stats, prune, calibrate, adaptive, threshold, evaluate).
export_annotated.py — produce final annotated CSV with column "insurance_label".
Evaluate_model.py — standalone evaluator and threshold sweep.
ml_insurance_challenge_cleaned.csv, insurance_taxonomy.csv — input data.
Main outputs:
ml_insurance_challenge_annotated.csv (final annotated list)
ml_insurance_challenge_labeled_*.csv (intermediate labeled outputs)
per_label_stats_auto.csv, auto_eval_summary.txt, *_threshold_sweep.csv (diagnostics)
company_emb.npy, label_emb.npy (cached embeddings)

How to run (example):
create env and install deps:
# filepath: "Your filepath here"
python -m pip install -r requirements.txt
run full pipeline:
python entire_pipeline.py --steps normalize,weight,embed,assign,stats,calibrate,adaptive,threshold,evaluate
export annotated CSV:
python export_annotated.py

