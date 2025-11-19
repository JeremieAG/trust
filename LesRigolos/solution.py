#!/usr/bin/env python3
"""
AutoComply PDF Section Splitter — Optimized & Stable Version
"""

import fitz
import base64
import requests
import json
import re

API_URL = "https://ai-models.autocomply.ca"
API_KEY = "sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c"
MODEL = "gemini-2.5-flash"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

SECTIONS = [
    "Articles & Amendments",
    "By Laws",
    "Unanimous Shareholder Agreement",
    "Minutes & Resolutions",
    "Directors Register",
    "Officers Register",
    "Shareholder Register",
    "Securities Register",
    "Share Certificates",
    "Ultimate Beneficial Owner Register"
]


# ------------------------------------------------------------
# PDF → Image → Base64
# ------------------------------------------------------------
def pdf_to_image_b64(doc, page_index):
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7))
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


# ------------------------------------------------------------
# Appel API avec extraction robuste du JSON
# ------------------------------------------------------------
def ask_page_info(page_b64):
    prompt = """
You are classifying MINUTE BOOK SECTIONS.
Ignore the text on the page. Classify the page into ONE of these 10 official sections:

1. Articles & Amendments
2. By Laws
3. Unanimous Shareholder Agreement
4. Minutes & Resolutions
5. Directors Register
6. Officers Register
7. Shareholder Register
8. Securities Register
9. Share Certificates
10. Ultimate Beneficial Owner Register

RETURN ONLY THIS JSON EXACTLY:

{
  "section": "<one of the 10 EXACT names above>",
  "position": "start|middle|end|unknown",
  "confidence": 0.0-1.0
}

Do NOT return any page titles, subtitles, headings or subtitles from the PDF.  
You must map the page TO ONE OF THE 10 SECTIONS ONLY.
"""

    payload = {
        "pdfPage": page_b64,
        "prompt": prompt,
        "model": MODEL
    }

    r = requests.post(f"{API_URL}/process-pdf", json=payload, headers=HEADERS)
    r.raise_for_status()

    raw = r.json()["result"]
    print("API raw:", raw[:120])

    match = re.search(r"\{.*?\}", raw, re.S)
    if not match:
        return {"section": "Unknown", "position": "unknown", "confidence": 0}

    try:
        return json.loads(match.group())
    except:
        return {"section": "Unknown", "position": "unknown", "confidence": 0}


# ------------------------------------------------------------
# Sampling : une page sur 10
# ------------------------------------------------------------
def classify_pages_sampling(doc):
    results = {}
    total = len(doc)

    print("Sampling pages...")
    for p in range(0, total, 3):
        print(f" → page {p}")
        img = pdf_to_image_b64(doc, p)
        result = ask_page_info(img)
        results[p] = result

    return results


# ------------------------------------------------------------
# Détection locale (sans API)
# ------------------------------------------------------------
def local_detect_tables(page):
    blocks = page.get_text("blocks")
    text_len = sum(len(b[4]) for b in blocks)
    num_blocks = len(blocks)

    return text_len > 150 or num_blocks > 8


def detect_candidates(doc):
    print("Local candidate detection...")
    candidates = []
    for i in range(len(doc)):
        if local_detect_tables(doc.load_page(i)):
            candidates.append(i)
    return candidates


# ------------------------------------------------------------
# Recherche binaire
# ------------------------------------------------------------
def binary_search_boundary(doc, low, high, target_section):
    print(f"Binary search for {target_section} in {low}-{high}")

    while low < high:
        mid = (low + high) // 2
        print("   checking page", mid)

        img = pdf_to_image_b64(doc, mid)
        info = ask_page_info(img)

        if info["section"] == target_section:
            high = mid
        else:
            low = mid + 1

    return low


# ------------------------------------------------------------
# Pipeline complet
# ------------------------------------------------------------
def build_sections_map(doc):
    total_pages = len(doc)
    print("Total pages:", total_pages)

    # 1) Local detection
    candidates = detect_candidates(doc)

    # 2) Sampling
    sampled = classify_pages_sampling(doc)

    # 3) Approximate ranges
    rough_ranges = {}
    sorted_pages = sorted(sampled.keys())

    for p in sorted_pages:
        info = sampled[p]
        section = info["section"]

        if section not in SECTIONS:
            continue

        if section not in rough_ranges:
            rough_ranges[section] = {"start": p, "end": p}
        else:
            rough_ranges[section]["end"] = p

    # 4) Binary search refinement
    final_sections = []

    for section, rng in rough_ranges.items():
        approx_start = rng["start"]
        approx_end = rng["end"]

        start = binary_search_boundary(doc,
                               max(0, approx_start - 20),
                               approx_start + 2,
                               section)

        end = binary_search_boundary(doc,
                             approx_end - 2,
                             min(total_pages - 1, approx_end + 20),
                             section)


        final_sections.append({
            "name": section,
            "startPage": start + 1,
            "endPage": end + 1
        })
    final_sections.sort(key=lambda x: x["startPage"])

    clean = []
    last_end = 0

    for sec in final_sections:
        if sec["startPage"] > last_end:
            clean.append(sec)
        last_end = sec["endPage"]

    return clean


# ------------------------------------------------------------
# Output JSON
# ------------------------------------------------------------
def save_result_json(sections):
    data = {"sections": sections}

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Result saved → result.json")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python solution.py fichier.pdf")
        return

    pdf_path = sys.argv[1]
    print("Loading PDF:", pdf_path)

    doc = fitz.open(pdf_path)

    sections = build_sections_map(doc)
    save_result_json(sections)


if __name__ == "__main__":
    main()
