#!/usr/bin/env python3
"""Audit the anonymous ICRA 2027 PDF and its frozen claim provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_icra2027"
DEFAULT_PDF = PAPER / "main.pdf"
DEFAULT_REPORT_DIR = ROOT / "outputs/actmask/icra2027_manuscript_audit"
CLAIM_PACKAGE = ROOT / "outputs/actmask/icra2027_submission"

OFFICIAL_CLASS_SHA256 = "4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2"
FROZEN_BST_SHA256 = "b11af8e5096681f1eccdce6c72c047dc056ddeef52dff340213104019bcf3409"
RANKFIX_RAW_SHA256 = "8a46655dd195b704c221408a5e41f1708ec095df456cad475f22bd0178772947"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable(name: str) -> str:
    local = ROOT / ".tools/texenv/bin" / name
    if local.is_file():
        return str(local)
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required executable not found: {name}")
    return resolved


def run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def parse_pdfinfo(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def font_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("name"))
    header = lines[header_index]
    starts = {
        name: header.index(name)
        for name in ("name", "type", "encoding", "emb", "sub", "uni", "object ID")
    }
    ordered = ["name", "type", "encoding", "emb", "sub", "uni", "object ID"]
    rows = []
    for line in lines[header_index + 2 :]:
        if not line.strip():
            continue
        row = {}
        for position, name in enumerate(ordered):
            start = starts[name]
            end = starts[ordered[position + 1]] if position + 1 < len(ordered) else None
            row[name] = line[start:end].strip()
        rows.append(row)
    return rows


def cited_keys(tex: str) -> set[str]:
    groups = re.findall(r"\\cite\{([^}]+)\}", tex)
    return {key.strip() for group in groups for key in group.split(",")}


def bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", bib))


def directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def audit(pdf: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    required = [
        PAPER / "main.tex",
        PAPER / "references.bib",
        PAPER / "ieeeconf.cls",
        PAPER / "IEEEtran.bst",
        PAPER / "figures/visual_matched_histories.png",
        PAPER / "figures/visual_matched_histories.manifest.json",
        PAPER / "generated/manifest.json",
        PAPER / "build_pdflatex/main.log",
        pdf,
    ]
    all_required = all(path.is_file() for path in required)
    add("required_files", all_required, [str(path) for path in required])
    if not all_required:
        return {
            "schema": "actmask-icra2027-manuscript-audit-v1",
            "passed": False,
            "checks": checks,
        }

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    log = (PAPER / "build_pdflatex/main.log").read_text(encoding="utf-8", errors="replace")
    pdfinfo = parse_pdfinfo(run_text([executable("pdfinfo"), str(pdf)]))
    fonts = font_rows(run_text([executable("pdffonts"), str(pdf)]))
    pdf_text = run_text([executable("pdftotext"), str(pdf), "-"])

    pages = int(pdfinfo.get("Pages", "0"))
    page_size = pdfinfo.get("Page size", "")
    add("page_limit", 1 <= pages <= 8, {"pages": pages, "maximum": 8})
    add("us_letter", page_size.startswith("612 x 792 pts"), page_size)
    add("pdf_size", pdf.stat().st_size <= 20 * 1024 * 1024, pdf.stat().st_size)

    add("fonts_present", bool(fonts), {"count": len(fonts)})
    add(
        "fonts_embedded",
        bool(fonts) and all(row["emb"] == "yes" for row in fonts),
        [{"name": row["name"], "embedded": row["emb"]} for row in fonts],
    )
    add(
        "no_type3_fonts",
        bool(fonts) and all("Type 3" not in row["type"] for row in fonts),
        sorted({row["type"] for row in fonts}),
    )

    blocking_log_patterns = {
        "undefined": r"undefined references|Citation .* undefined|Reference .* undefined",
        "overfull": r"Overfull \\[hv]box",
        "font_warning": r"LaTeX Font Warning",
        "pdftex_warning": r"pdfTeX warning",
        "fatal": r"Fatal error|Emergency stop|^! ",
    }
    log_hits = {
        name: re.findall(pattern, log, flags=re.IGNORECASE | re.MULTILINE)
        for name, pattern in blocking_log_patterns.items()
    }
    add("clean_latex_log", not any(log_hits.values()), log_hits)
    add(
        "underfull_only_nonblocking",
        True,
        {
            "underfull_hbox": len(re.findall(r"Underfull \\hbox", log)),
            "underfull_vbox": len(re.findall(r"Underfull \\vbox", log)),
        },
    )

    author_match = re.search(r"\\author\{([^}]*)\}", tex)
    anonymous_source = author_match is not None and author_match.group(1).strip() == "Anonymous Authors"
    source_leaks = re.findall(r"/(?:data|home)/[^\s}\]]+|[\w.+-]+@[\w.-]+", tex)
    first_page_text = "\n".join(pdf_text.splitlines()[:80])
    add(
        "double_anonymous",
        anonymous_source and "Anonymous Authors" in first_page_text and not source_leaks,
        {
            "source_author": author_match.group(1) if author_match else None,
            "source_leaks": source_leaks,
        },
    )
    acknowledgment = re.search(
        r"\\section\*\{Acknowledgment\}\s*(.*?)\s*\\bibliographystyle",
        tex,
        flags=re.DOTALL,
    )
    acknowledgment_text = acknowledgment.group(1) if acknowledgment else ""
    acknowledgment_flat = re.sub(r"\s+", " ", acknowledgment_text)
    acknowledgment_identity_hits = re.findall(
        r"(?:funded by|grant(?: no\.| number)?|we thank|thanks to|@[\w.-]+|https?://)",
        acknowledgment_text,
        flags=re.IGNORECASE,
    )
    add(
        "anonymous_ai_acknowledgment",
        acknowledgment is not None
        and tex.count("\\section*{Acknowledgment}") == 1
        and not acknowledgment_identity_hits,
        {
            "section_count": tex.count("\\section*{Acknowledgment}"),
            "identity_sensitive_hits": acknowledgment_identity_hits,
        },
    )
    ai_disclosure_markers = [
        "OpenAI Codex",
        "Abstract",
        "Introduction",
        "Related Work",
        "Discussion and Limitations",
        "Reproducibility and Conclusion",
        "implementing downstream audit and verification code",
        "orchestrating execution of the frozen experiment scripts",
    ]
    add(
        "specific_ai_use_disclosure",
        all(marker in acknowledgment_flat for marker in ai_disclosure_markers),
        ai_disclosure_markers,
    )

    missing_bib = sorted(cited_keys(tex) - bib_keys(bib))
    add("bibliography_keys", not missing_bib, {"missing": missing_bib, "cited": len(cited_keys(tex))})
    add("resolved_pdf_crossrefs", "??" not in pdf_text, "PDF text scan")

    class_hash = sha256(PAPER / "ieeeconf.cls")
    bst_hash = sha256(PAPER / "IEEEtran.bst")
    add("official_ieeeconf_class", class_hash == OFFICIAL_CLASS_SHA256, class_hash)
    add("frozen_ieeetran_bst", bst_hash == FROZEN_BST_SHA256, bst_hash)

    manifest = json.loads((PAPER / "generated/manifest.json").read_text(encoding="utf-8"))
    add(
        "claim_manifest",
        manifest["pass"] is True
        and len(manifest["sources"]) == 16
        and len(manifest["checks"]) == 36
        and len(manifest["outputs"]) == 16
        and all(row["status"] == "PASS" for row in manifest["checks"]),
        {
            "pass": manifest["pass"],
            "sources": len(manifest["sources"]),
            "checks": len(manifest["checks"]),
            "outputs": len(manifest["outputs"]),
        },
    )
    paper_generated = directory_hashes(PAPER / "generated")
    canonical_generated = directory_hashes(CLAIM_PACKAGE)
    add(
        "generated_package_matches_canonical",
        paper_generated == canonical_generated,
        {
            "paper_files": len(paper_generated),
            "canonical_files": len(canonical_generated),
            "differing": sorted(
                key
                for key in set(paper_generated) | set(canonical_generated)
                if paper_generated.get(key) != canonical_generated.get(key)
            ),
        },
    )

    figure_manifest = json.loads(
        (PAPER / "figures/visual_matched_histories.manifest.json").read_text(encoding="utf-8")
    )
    figure_hash = sha256(PAPER / "figures/visual_matched_histories.png")
    add(
        "visual_figure_manifest",
        figure_manifest["pass"] is True
        and figure_manifest["figure"]["sha256"] == figure_hash,
        {"manifest": figure_manifest["figure"]["sha256"], "observed": figure_hash},
    )

    state_figure_manifest = json.loads(
        (PAPER / "figures/state_audit_construction.manifest.json").read_text(encoding="utf-8")
    )
    state_figure_hash = sha256(PAPER / "figures/state_audit_construction.png")
    add(
        "state_figure_manifest",
        state_figure_manifest["pass"] is True
        and state_figure_manifest["figure"]["sha256"] == state_figure_hash,
        {
            "manifest": state_figure_manifest["figure"]["sha256"],
            "observed": state_figure_hash,
        },
    )

    rankfix_output = run_text(
        [
            sys.executable,
            "-m",
            "actmask.experiments.milestone3r_nl_v2_rankfix",
            "--verify-only",
            "--output-dir",
            str(ROOT / "outputs/actmask/milestone3r_nl_v2_rankfix"),
        ]
    )
    rankfix = json.loads(rankfix_output)
    add(
        "history_keyed_rankfix",
        rankfix["pass"] is True and rankfix["raw_score_hash"] == RANKFIX_RAW_SHA256,
        {"pass": rankfix["pass"], "raw_score_hash": rankfix["raw_score_hash"]},
    )

    wav_audit_output = run_text(
        [sys.executable, str(ROOT / "scripts/verify_icra2027_wav_external_audit.py")]
    )
    wav_audit = json.loads(wav_audit_output)
    add(
        "wav_external_audit",
        wav_audit["passed"] is True and wav_audit["checks"] == 9,
        wav_audit,
    )

    required_scope_phrases = [
        "benchmark admission as an execution-grounded intervention with three gates",
        "identification test, not a surrogate for natural robot data",
        "central contribution is the evaluation intervention rather than the GRU",
        "not a claim of general visual reasoning, real-robot transfer, or safety",
        "fails method admission",
        "do not approximate open-world or sim-to-real shift",
        "does not audit WAV's Action Following Score",
    ]
    add(
        "claim_scope_boundaries",
        all(phrase in re.sub(r"\s+", " ", tex) for phrase in required_scope_phrases),
        required_scope_phrases,
    )

    return {
        "schema": "actmask-icra2027-manuscript-audit-v1",
        "passed": all(row["status"] == "PASS" for row in checks),
        "pdf": {
            "path": str(pdf.relative_to(ROOT)),
            "sha256": sha256(pdf),
            "bytes": pdf.stat().st_size,
            "pages": pages,
            "page_size": page_size,
            "producer": pdfinfo.get("Producer"),
        },
        "checks": checks,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "manuscript_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ICRA 2027 manuscript audit",
        "",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
    ]
    if "pdf" in report:
        lines.extend(
            [
                f"- PDF: `{report['pdf']['path']}`",
                f"- SHA-256: `{report['pdf']['sha256']}`",
                f"- Pages: {report['pdf']['pages']}",
                f"- Page size: {report['pdf']['page_size']}",
                "",
            ]
        )
    lines.extend(f"- [{row['status']}] `{row['name']}`" for row in report["checks"])
    (output / "manuscript_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    arguments = parser.parse_args()
    report = audit(arguments.pdf.resolve())
    write_report(report, arguments.report_dir.resolve())
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(report["checks"]),
                "pdf": report.get("pdf"),
                "report_dir": str(arguments.report_dir.resolve()),
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
