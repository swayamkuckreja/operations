#!/usr/bin/env bash
# run_all.sh — regenerate every result artifact from scratch.
#
# Run this from the project root:
#     bash run_all.sh
#
# Requires a non-restricted Gurobi licence (academic or commercial) because
# the verification, sensitivity, and validation scripts use n in {6, 8, 10}
# which exceed the free-tier size cap.

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
RES_DIR="${PROJECT_DIR}/results"
REP_DIR="${PROJECT_DIR}/report"

mkdir -p "${RES_DIR}/verification"
mkdir -p "${RES_DIR}/sensitivity"
mkdir -p "${REP_DIR}/figures"

echo "=========================================================="
echo "  1/4  verification suite (V1-V7)"
echo "=========================================================="
cd "${SRC_DIR}"
python verification.py 2>&1 | tee "${RES_DIR}/verification/log.txt"

echo
echo "=========================================================="
echo "  2/4  validation table (paper Tab. A.5)"
echo "=========================================================="
python validation.py 2>&1 | tee "${RES_DIR}/verification/validation_log.txt"

echo
echo "=========================================================="
echo "  3/4  sensitivity sweeps (endurance, drone speed, alpha)"
echo "=========================================================="
python sensitivity.py 2>&1 | tee "${RES_DIR}/sensitivity/log.txt"

echo
echo "=========================================================="
echo "  4/4  copy figures into report and compile PDF"
echo "=========================================================="
cp "${RES_DIR}/verification/"V*.pdf "${REP_DIR}/figures/" 2>/dev/null || true
cp "${RES_DIR}/sensitivity/"*.pdf "${REP_DIR}/figures/" 2>/dev/null || true

cd "${REP_DIR}"
if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    pdflatex -interaction=nonstopmode main.tex >/dev/null   # second pass for refs
    echo "  -> ${REP_DIR}/main.pdf"
else
    echo "  pdflatex not found; skipping PDF compile."
fi

echo
echo "=========================================================="
echo "  done."
echo "=========================================================="
echo "  CSVs:    ${RES_DIR}/sensitivity/*.csv"
echo "          ${RES_DIR}/verification/validation_table.csv"
echo "  PDFs:   ${RES_DIR}/verification/V*.pdf"
echo "          ${RES_DIR}/sensitivity/*.pdf"
echo "          ${REP_DIR}/main.pdf (if pdflatex available)"
