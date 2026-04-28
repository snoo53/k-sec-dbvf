"""Render the JMST cover letter as PDF using reportlab.

Reads JMST-COVER-LETTER.md and produces CoverLetter.pdf.
"""

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "JMST-COVER-LETTER.md"
OUT = ROOT / "applications/jmst_submission/CoverLetter.pdf"


def md_to_html_inline(text: str) -> str:
    """Convert minimal markdown inline formatting to reportlab-friendly HTML."""
    # Convert **bold** to <b>bold</b>
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # Convert *italic* to <i>italic</i>
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    # Escape ampersands that aren't already escaped
    text = text.replace("&", "&amp;").replace("&amp;lt;", "&lt;").replace("&amp;gt;", "&gt;").replace("&amp;b&gt;", "<b>").replace("&amp;/b&gt;", "</b>").replace("&amp;i&gt;", "<i>").replace("&amp;/i&gt;", "</i>")
    # Restore the bold/italic tags we just inadvertently double-escaped
    text = re.sub(r"<b>(.*?)</b>", lambda m: f"<b>{m.group(1)}</b>", text)
    return text


def build():
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13, leading=16,
                       spaceBefore=10, spaceAfter=4, fontName="Times-Bold")
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, leading=14,
                       spaceBefore=8, spaceAfter=2, fontName="Times-Bold")
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=11, leading=14,
                         spaceAfter=6, fontName="Times-Roman", alignment=4)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=18, bulletIndent=4, spaceAfter=2)

    md = SRC.read_text(encoding="utf-8")
    lines = md.split("\n")

    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="Cover Letter — JMST Submission",
        author="Sunwoo Lee",
    )

    story = []
    buf = []

    def flush_para():
        if buf:
            text = " ".join(buf).strip()
            if text:
                story.append(Paragraph(md_to_html_inline(text), body))
            buf.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("# "):
            flush_para()
            story.append(Paragraph(md_to_html_inline(line[2:]), h1))
        elif line.startswith("## "):
            flush_para()
            story.append(Paragraph(md_to_html_inline(line[3:]), h2))
        elif line.startswith("### "):
            flush_para()
            story.append(Paragraph(md_to_html_inline(line[4:]), h2))
        elif line.startswith("- ") or line.startswith("* "):
            flush_para()
            story.append(Paragraph(f"• {md_to_html_inline(line[2:])}", bullet))
        elif re.match(r"^\d+\.\s", line):
            flush_para()
            story.append(Paragraph(md_to_html_inline(line), bullet))
        elif line.strip() == "":
            flush_para()
        elif line.startswith("---"):
            flush_para()
            story.append(Spacer(1, 6))
        else:
            buf.append(line.strip())
    flush_para()

    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
