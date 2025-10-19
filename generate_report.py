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
import re

SUMMARY_TXT = "auto_eval_summary.txt"
PER_LABEL = "per_label_stats_auto.csv"
OUT_PDF = "REPORT.pdf"

def read_summary(path):
    if not os.path.exists(path):
        return "No summary file found."
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def round_floats_in_text(s, decimals=2):
    # replace floating numbers with rounded variants (keeps integers intact)
    def repl(m):
        try:
            val = float(m.group(0))
            return f"{val:.{decimals}f}"
        except:
            return m.group(0)
    return re.sub(r'(?<!\w)(-?\d+\.\d+)(?!\w)', repl, s)

def top_labels_csv(path, n=8):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, dtype=str).fillna('')
    # ensure numeric
    if 'count' in df.columns:
        df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
        # ensure avg_score numeric if present
        if 'avg_score' in df.columns:
            df['avg_score'] = pd.to_numeric(df['avg_score'], errors='coerce')
        else:
            df['avg_score'] = None
        df = df.sort_values('count', ascending=False)
    rows = []
    for _, r in df.head(n).iterrows():
        label = r.get('label','')
        count = int(r.get('count',0))
        avg = r.get('avg_score', None)
        if pd.isna(avg) or avg is None:
            avg_str = ""
        else:
            try:
                avg_str = f"{float(avg):.2f}"
            except:
                avg_str = str(avg)
        rows.append([label, count, avg_str])
    return rows

def build_pdf():
    summary = read_summary(SUMMARY_TXT)
    # round any floating numbers in summary to 2 decimals for readability
    summary = round_floats_in_text(summary, decimals=2)
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
    doc.build(story)
    print("Saved PDF report:", OUT_PDF)

if __name__ == "__main__":
    build_pdf()