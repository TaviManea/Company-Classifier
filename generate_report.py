"""
Generate a one-page PDF REPORT.pdf from auto_evaluation_summary.txt and per_label_stats_auto.csv.

Requires: reportlab
Install:
    python -m pip install reportlab pandas
Run:
    python generate_report.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import pandas as pd
import datetime
import os

SUMMARY_TXT = "auto_eval_summary.txt"
PER_LABEL = "per_label_stats_auto.csv"
OUT_PDF = "REPORT.pdf"

def read_summary(path):
    if not os.path.exists(path):
        return "No summary file found."
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def top_labels_csv(path, n=8):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, dtype=str).fillna('')
    # ensure numeric
    if 'count' in df.columns:
        df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
        df = df.sort_values('count', ascending=False)
    rows = df.head(n)[['label','count','avg_score']].values.tolist()
    return rows

def build_pdf():
    summary = read_summary(SUMMARY_TXT)
    top = top_labels_csv(PER_LABEL, n=8)
    doc = SimpleDocTemplate(OUT_PDF, pagesize=letter, rightMargin=36,leftMargin=36, topMargin=36,bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    title = Paragraph("Veridion Challenge — One-page Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1,12))
    meta = Paragraph(f"Generated: {datetime.datetime.now().isoformat(timespec='minutes')}", styles['Normal'])
    story.append(meta)
    story.append(Spacer(1,12))
    story.append(Paragraph("Summary (auto_evaluation_summary.txt):", styles['Heading2']))
    for line in summary.splitlines():
        story.append(Paragraph(line, styles['Normal']))
    story.append(Spacer(1,12))
    if top:
        story.append(Paragraph("Top labels (by count):", styles['Heading2']))
        table_data = [["Label","Count","Avg Score"]] + top
        table = Table(table_data, colWidths=[260,70,70])
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0")),
            ('GRID',(0,0),(-1,-1),0.25,colors.grey),
            ('ALIGN',(1,0),(-1,-1),'CENTER'),
        ]))
        story.append(table)
        story.append(Spacer(1,12))
    story.append(Paragraph("Notes & Next Steps:", styles['Heading2']))
    notes = [
        "1) Prune noisy, high-count labels with low avg_score.",
        "2) Choose an operational threshold from topk_threshold_candidates.csv (95%/99% policy).",
        "3) Keep small human-labeled gold set (~200-300 rows) for precision@k validation and calibrator training.",
        "4) Publish only scripts, README and final annotated CSV to GitHub; remove large embeddings and caches."
    ]
    for n in notes:
        story.append(Paragraph(n, styles['Bullet']))
    doc.build(story)
    print("Saved PDF report:", OUT_PDF)

if __name__ == "__main__":
    build_pdf()