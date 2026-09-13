"""Generate realistic sample regulatory PDFs for the ChronosAudit demo.

Produces ``sample/*.pdf`` with *changing* facts over time so the temporal
retrieval layer has something interesting to answer:

* ``cross_border_data_policy.pdf`` — a compliance protocol that changes on
  2024-07-01 (localization mandate → Standard Contractual Clauses regime).
* ``tax_code_amendments_2023_2025.pdf`` — corporate tax law with a rate change
  effective 2025-01-01 and various scope amendments.

Usage:
    python scripts/make_sample_pdfs.py [--out-dir sample]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

HEADING = "ChronosAudit Regulatory Compliance Manual (Sample)"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ChronosH1", parent=styles["Heading1"], fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name="ChronosH2", parent=styles["Heading2"], fontSize=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="ChronosBody", parent=styles["BodyText"], fontSize=10.5, leading=14, spaceAfter=8))
    return styles


def build_cross_border_policy(path: Path) -> None:
    s = _styles()
    story = [
        Paragraph(HEADING, s["ChronosH1"]),
        Paragraph("Global Data Compliance Series — Document 001", s["ChronosH2"]),
        Paragraph(
            "Cross-Border Data Transfer Policy · Consolidated as of 1 September 2024 · "
            "Supersedes the version of 31 December 2023.", s["ChronosH2"]
        ),
        Paragraph("1. Policy intent", s["ChronosH2"]),
        Paragraph(
            "This policy governs how covered entities may transfer Personal Data across "
            "jurisdictional borders. It was first adopted on 15 March 2022 and amended in "
            "July 2024 to reflect the revised EU adequacy framework.", s["ChronosBody"]
        ),
        Paragraph("2. Historical regime (15 March 2022 until 30 June 2024)", s["ChronosH2"]),
        Paragraph(
            "From 2022-03-15 to 2024-06-30 the Cross-Border Data Transfer Policy "
            "prohibited the transfer of Personal Data outside the EU unless an explicit "
            "data localization waiver was granted by the Data Protection Authority. "
            "Waivers required a documented Business-Impact Assessment and were valid for "
            "no more than twelve months.", s["ChronosBody"]
        ),
        Paragraph("3. Current regime (from 1 July 2024)", s["ChronosH2"]),
        Paragraph(
            "Effective 2024-07-01 until further notice, the Data Transfer Policy permits "
            "cross-border data transfers when the recipient implements Standard "
            "Contractual Clauses approved by the Data Protection Authority and the "
            "transfer is recorded in the public Transfer Register within 30 days.", s["ChronosBody"]
        ),
        Paragraph("4. Compliance obligations for Q3 2024", s["ChronosH2"]),
        Paragraph(
            "During Q3 2024 covered entities were required to (a) re-execute controller-to-"
            "processor agreements under the 2024 clause set no later than 2024-09-30, "
            "(b) publish transition timelines for transfers that previously relied on "
            "localization waivers, and (c) obtain board sign-off for transfers exceeding "
            "25 million EUR per annum.", s["ChronosBody"]
        ),
        Paragraph("5. Enforcement", s["ChronosH2"]),
        Paragraph(
            "The Data Protection Authority enforces this policy under the EU Data "
            "Governance Act. Non-compliance after 2024-07-01 may trigger administrative "
            "fines of up to 4% of global turnover.", s["ChronosBody"]
        ),
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
def build_tax_code_amendments(path: Path) -> None:
    s = _styles()
    story = [
        Paragraph(HEADING, s["ChronosH1"]),
        Paragraph("Corporate Tax Series — Document 002", s["ChronosH2"]),
        Paragraph("Corporate Tax Code Amendments 2023–2025 · Financial Year Coverage", s["ChronosH2"]),
        Paragraph("1. Standard rate applied from 1 January 2023", s["ChronosH2"]),
        Paragraph(
            "The Corporate Tax Code imposes a 25 percent standard corporate income tax "
            "rate on resident companies from 2023-01-01. Small domestic companies with "
            "turnover below 2 million EUR are exempt until 2025-12-31.", s["ChronosBody"]
        ),
        Paragraph("2. Global minimum tax (Pillar Two)", s["ChronosH2"]),
        Paragraph(
            "Effective 2024-01-01 the Tax Reform Act imposes a 15 percent corporate "
            "minimum tax on Large Multinational Groups with consolidated revenue above "
            "750 million EUR. The minimum tax applies to each jurisdiction in which the "
            "group operates.", s["ChronosBody"]
        ),
        Paragraph("3. Amendment to the standard rate from 1 January 2025", s["ChronosH2"]),
        Paragraph(
            "The Corporate Finance Act amends the Tax Reform Act effective 2025-01-01 by "
            "reducing the standard corporate income tax rate from 25 percent to 21 "
            "percent for all resident companies, and extending the small-company "
            "exemption to companies with turnover below 5 million EUR.", s["ChronosBody"]
        ),
        Paragraph("4. Anti-avoidance measures (2023–2025)", s["ChronosH2"]),
        Paragraph(
            "From 2023-01-01 the code limits interest deduction to 30 percent of EBITDA. "
            "From 2024-06-01 a digital services levy of 3 percent applies to online "
            "advertising revenues. Both measures remain in force until further notice.",
            s["ChronosBody"]
        ),
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample ChronosAudit PDFs")
    parser.add_argument("--out-dir", default="sample", help="output directory")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf1 = out / "cross_border_data_policy.pdf"
    pdf2 = out / "tax_code_amendments_2023_2025.pdf"
    build_cross_border_policy(pdf1)
    build_tax_code_amendments(pdf2)
    print(f"✓ {pdf1} ({pdf1.stat().st_size} bytes)")
    print(f"✓ {pdf2} ({pdf2.stat().st_size} bytes)")
    print("Generated. Ingest with:  python scripts/seed_demo.py")


if __name__ == "__main__":
    main()