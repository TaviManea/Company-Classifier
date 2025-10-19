# Veridion Challenge - Solution
Summary: pipeline to assign insurance taxonomy labels to companies (preprocess → embed → ensemble → assign → evaluate).

How to reproduce (minimal):
1. Create venv, install:
   python -m pip install -r requirements.txt
2. Run pipeline (example):
   python solution.py 
3. Export annotated CSV:
   python export_annotated.py

Included outputs:
- ml_insurance_challenge_annotated.csv : annotated input with column "insurance_label"
- auto_eval_summary.txt
- per_label_stats_auto.csv
