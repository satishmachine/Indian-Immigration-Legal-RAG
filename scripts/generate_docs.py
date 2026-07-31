"""
scripts/generate_docs.py
========================
Automated generator for enterprise-grade documentation package:
1. docs/01_Project_Overview.md ... docs/15_Developer_Guide.md
2. PROJECT_DOCUMENTATION.md
3. PROJECT_DOCUMENTATION.pdf (via ReportLab)
"""

import json, os, sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRATCH_DIR = ROOT_DIR / "scratch"
DOCS_DIR = ROOT_DIR / "docs"

DOCS_DIR.mkdir(exist_ok=True)
SCRATCH_DIR.mkdir(exist_ok=True)

manifest_path = SCRATCH_DIR / "full_code_manifest.json"
if not manifest_path.exists():
    print("Manifest file not found at", manifest_path)
    sys.exit(1)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"Loaded manifest with {len(manifest)} Python files.")
