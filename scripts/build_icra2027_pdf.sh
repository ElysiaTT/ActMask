#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAPER_DIR="$REPO_ROOT/paper_icra2027"
BUILD_DIR="$PAPER_DIR/build_pdflatex"
ACTMASK_PYTHON=${ACTMASK_PYTHON:-/home/tzh/conda_envs/actmask/bin/python}
# Freeze PDF metadata so a clean rebuild is byte-for-byte reproducible.  The
# default epoch is 2026-08-09 00:00:00 UTC; callers may override it explicitly.
ICRA_SOURCE_DATE_EPOCH=${ICRA_SOURCE_DATE_EPOCH:-1786233600}

if [[ ! -x "$ACTMASK_PYTHON" ]]; then
    ACTMASK_PYTHON=$(command -v python3)
fi

"$ACTMASK_PYTHON" "$REPO_ROOT/scripts/build_icra2027_state_world_bootstrap.py"
"$ACTMASK_PYTHON" "$REPO_ROOT/scripts/verify_icra2027_wav_external_audit.py"
"$ACTMASK_PYTHON" "$REPO_ROOT/scripts/build_icra2027_claim_package.py" \
    --output-dir "$PAPER_DIR/generated"
"$ACTMASK_PYTHON" "$REPO_ROOT/scripts/build_icra2027_claim_package.py" \
    --output-dir "$REPO_ROOT/outputs/actmask/icra2027_submission"
"$ACTMASK_PYTHON" "$REPO_ROOT/scripts/build_icra2027_state_audit_figure.py"
"$ACTMASK_PYTHON" "$REPO_ROOT/scripts/build_icra2027_visual_history_figure.py"

mkdir -p "$BUILD_DIR"
cd "$PAPER_DIR"

LOCAL_TEX_BIN="$REPO_ROOT/.tools/texenv/bin"
LOCAL_TEX_ROOT="$REPO_ROOT/.tools/apt-texlive/usr/share/texlive"
LOCAL_TEX_DIST="$LOCAL_TEX_ROOT/texmf-dist"
LOCAL_TEX_VAR="$REPO_ROOT/.tools/apt-texlive/var/lib/texmf"
LOCAL_TEX_CONFIG="$REPO_ROOT/.tools/apt-texlive/config"
LOCAL_TEX_HOME="$REPO_ROOT/.tools/apt-texlive/home"

if [[ -x "$LOCAL_TEX_BIN/latexmk" && -f "$LOCAL_TEX_VAR/web2c/pdftex/pdflatex.fmt" ]]; then
    env \
        PATH="$LOCAL_TEX_BIN:$PATH" \
        TEXMFCNF="$LOCAL_TEX_DIST/web2c" \
        TEXMFROOT="$LOCAL_TEX_ROOT" \
        TEXMFDIST="$LOCAL_TEX_DIST" \
        TEXMFMAIN="$LOCAL_TEX_DIST" \
        TEXMFVAR="$LOCAL_TEX_VAR" \
        TEXMFSYSVAR="$LOCAL_TEX_VAR" \
        TEXMFCONFIG="$LOCAL_TEX_CONFIG" \
        TEXMFSYSCONFIG="$LOCAL_TEX_CONFIG" \
        TEXMFHOME="$LOCAL_TEX_HOME" \
        SOURCE_DATE_EPOCH="$ICRA_SOURCE_DATE_EPOCH" \
        FORCE_SOURCE_DATE=1 \
        TZ=UTC \
        "$LOCAL_TEX_BIN/latexmk" \
        -g -pdf -interaction=nonstopmode -halt-on-error \
        -outdir="$BUILD_DIR" main.tex
    PDFINFO="$LOCAL_TEX_BIN/pdfinfo"
else
    env SOURCE_DATE_EPOCH="$ICRA_SOURCE_DATE_EPOCH" FORCE_SOURCE_DATE=1 TZ=UTC \
        latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
        -outdir="$BUILD_DIR" main.tex
    PDFINFO=$(command -v pdfinfo)
fi

cp "$BUILD_DIR/main.pdf" "$PAPER_DIR/main.pdf"
"$PDFINFO" "$PAPER_DIR/main.pdf" | grep -E '^(Pages|Page size|File size):'
sha256sum "$PAPER_DIR/main.pdf"
