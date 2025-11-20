#!/usr/bin/env python3
"""
PDF Client for AutoComply API

This script processes PDF files by converting them to images and sending them
to the AutoComply PDF processing API for analysis.
"""

import requests
import base64
import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF
from PIL import Image
import io
import json

PROMPT = """
You are classifying a single page of a corporate minute book.

Your task is to identify which EXACT section this page belongs to.
Choose only from:

1. "Articles & Amendments"
2. "By Laws"
3. "Unanimous Shareholder Agreement"
4. "Minutes & Resolutions"
5. "Directors Register"
6. "Officers Register"
7. "Shareholder Register"
8. "Securities Register"
9. "Share Certificates"
10. "Ultimate Beneficial Owner Register"

Rules:
- Map French titles to the exact English section name.
- If page doesn't belong to any, use "NONE".
- Provide a confidence score.

Here are the distinctive features of each section:

1. "Articles & Amendments"
   - Government-issued documents
   - Includes corporation name, number, address, share classes
   - Amendments mention changes (name, rights, address)
   - Continuation or fusion documents include repeated statutory info

2. "By Laws"
   - First page has a header containing "By-Law" or "Règlement"
   - Paragraphs are often numbered
   - Describes internal rules and procedures

3. "Unanimous Shareholder Agreement"
   - Title with "Unanimous Shareholder Agreement" / "Convention Unanime"
   - Signed by all shareholders
   - Describes rights and obligations of shareholders

4. "Minutes & Resolutions"
   - Contains many independent documents
   - Often the longest section
   - Resolutions and meeting minutes (AGM, directors' resolutions, etc.)

5. "Directors Register"
   - Table format
   - Includes name, address, start date, end date, residence

6. "Officers Register"
   - Table format
   - Includes name, address, start date, end date, function

7. "Shareholder Register"
   - Table format
   - Includes name, address, start date, end date

8. "Securities Register"
   - Table format
   - Pages per shareholder category
   - Lists share transactions and transfers

9. "Share Certificates"
   - Horizontal page layout
   - Shows number of shares many times
   - Shows shareholder name many times
   - Looks like a certificate

10. "Ultimate Beneficial Owner Register"
   - Table format
   - Includes ownership percentages

   
Respond ONLY with:
{
  "sectionName": "...",
  "confidence": 0.0
}
"""
def clean_result_text(text: str) -> str:
    text = text.strip()

    # Remove all ```json or ``` wrappers
    if text.startswith("```"):
        # Remove leading fence
        text = text.split("```", 1)[1].strip()

    # Remove trailing ```
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0].strip()

    # If it still begins with json\n remove that
    if text.lower().startswith("json"):
        text = text.split("\n", 1)[1].strip()

    return text



class PDFProcessor:
    def __init__(self, api_url: str = "https://ai-models.autocomply.ca", api_key: str = "sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c"):
        """
        Initialize the PDF processor with API configuration.
        
        Args:
            api_url: Base URL of the AutoComply API
            api_key: API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[bytes]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion
            
        Returns:
            List of image bytes for each page
        """
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert to image with specified DPI
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                images.append(img_data)
            
            doc.close()
            return images
            
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        """
        Convert image bytes to base64 string.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Base64 encoded string
        """
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def process_pdf_page(self, pdf_page_b64: str, prompt: str, model: str = "gemini-2.5-flash") -> Optional[dict]:
        """
        Send a PDF page to the API for processing.
        
        Args:
            pdf_page_b64: Base64 encoded PDF page image
            prompt: Processing prompt
            model: Model to use ('gpt-4o', 'gemini-2.5-flash', or 'claude-sonnet-4.5', default: 'gemini-2.5-flash')
            
        Returns:
            API response as dictionary or None if error
        """
        url = f"{self.api_url}/process-pdf"
        payload = {
            "pdfPage": pdf_page_b64,
            "prompt": prompt,
            "model": model
        }
        
        try:
            # response = requests.post(url, json=payload, headers=self.headers)
            # response.raise_for_status()
            # data = response.json().get("result", "{}")
            # return json.loads(data)
            response = requests.post(url, json=payload, headers=self.headers, timeout=20)

# If the server returned an error page, HTML, or empty string:
            if not response.content:
                print("API request failed: empty response")
                return {"sectionName": "NONE", "confidence": 0.0}

            try:
                raw = response.json()
                print("RAW SERVER RESPONSE:", raw)
                result_text = raw.get("result")
                print("RAW RESULT FIELD:", repr(result_text))
            except Exception:
                print("API request failed: invalid JSON")
                return {"sectionName": "NONE", "confidence": 0.0}
            # Case 1 → API returned a dict directly
            if isinstance(result_text, dict):
                return result_text

            # Case 2 → API returned nothing or null
            if not result_text:
                print("API request failed: empty or null result")
                return {"sectionName": "NONE", "confidence": 0.0}

            # Case 3 → API returned a string that must be parsed
            # Clean markdown wrapper
            cleaned = clean_result_text(result_text)

            try:
                return json.loads(cleaned)
            except Exception:
                print("API request failed: cleaned JSON still invalid")
                return {"sectionName": "NONE", "confidence": 0.0}

        except Exception as e:
            print(f"API request failed: {e}")
            return {"sectionName": "NONE", "confidence": 0.0}
        
    def sample_indices(self, n_pages: int) -> List[int]:
        if n_pages <= 50:
            stride = 8
        elif n_pages <= 100:
            stride = 16
        else:
            stride = 24
        idx = list(range(0, n_pages, stride))
        if (n_pages - 1) not in idx:
            idx.append(n_pages - 1)
        return idx
    
    def refine_boundary(self, left: int, right: int, images, cache, model: str="gemini-2.5-flash"):
        while right - left > 1:
            mid = (left + right) // 2
            if mid not in cache:
                cache[mid] = self.process_pdf_page(
                    self.image_to_base64(images[mid]),
                    PROMPT,
                    model=model
                )
            if cache[mid]["sectionName"] == cache[left]["sectionName"]:
                left = mid
            else:
                right = mid    
        return left, right
    
    def process(self, pdf_path: str, dpi: int, model: str="gemini-2.5-flash") -> List[dict]:
        print("Rendering PDF...")
        images = self.pdf_to_images(pdf_path, dpi=dpi)
        n = len(images)
        print(f"PDF has {n} pages")

        if n == 0:
            return []

        print("Sampling pages...")
        samples = self.sample_indices(n)

        cache = {}
        for i in samples:
            cache[i] = self.process_pdf_page(
                self.image_to_base64(images[i]),
                PROMPT,
                model=model
)
            time.sleep(0.25)

        print("Detecting transitions...")
        transitions = []
        for a, b in zip(samples, samples[1:]):
            if cache[a]["sectionName"] != cache[b]["sectionName"]:
                transitions.append((a, b))

        print("Refining boundaries with binary search...")
        for left, right in transitions:
            self.refine_boundary(left, right, images, cache)
            time.sleep(0.1)

        print("Building labels...")
        labels = [None] * n
        for i, info in cache.items():
            if info["confidence"] >= 0.45:
                labels[i] = info["sectionName"]

        # Fill gaps
        for i in range(1, n - 1):
            if labels[i] is None and labels[i - 1] == labels[i + 1] and labels[i - 1] is not None:
                labels[i] = labels[i - 1]

        # Extend flat ranges
        for i in range(n):
            if labels[i] is None and i > 0:
                labels[i] = labels[i - 1]
        last = None
        for i in range(n):
            if labels[i] is not None:
                last = labels[i]
            else:
                labels[i] = last

        print("Converting labels → sections...")

        # Step 1: Build raw contiguous segments
        segments = []
        i = 0
        while i < n:
            sec = labels[i]
            start = i
            while i < n and labels[i] == sec:
                i += 1
            end = i - 1
            segments.append({
                "name": sec,
                "start": start,
                "end": end,
                "size": end - start + 1
            })

        # Sections allowed to be 1 page
        valid_single = {
            "Directors Register",
            "Officers Register",
            "Shareholder Register",
            "Ultimate Beneficial Owner Register"
        }

        # Step 2: Remove noise spikes
        # Step 2: Remove noise spikes
        clean_segments = []
        for seg in segments:

            name = seg["name"]
            size = seg["size"]

            if name == "NONE":
                continue

            # Big sections must be at least 3 pages
            if name not in valid_single and size < 3:
                continue

            clean_segments.append(seg)

        # Additional early-section filtering
        impossible_early = {
            "Directors Register",
            "Officers Register",
            "Shareholder Register",
            "Securities Register",
            "Share Certificates",
        }

        filtered_segments = []
        for seg in clean_segments:
            if seg["start"] < 30 and seg["name"] in impossible_early:
                continue
            filtered_segments.append(seg)

        clean_segments = filtered_segments


        # Step 3: Merge adjacent chunks of the same section
        merged = []
        for seg in clean_segments:
            if not merged:
                merged.append(seg)
            else:
                last = merged[-1]
                if last["name"] == seg["name"] and last["end"] + 1 == seg["start"]:
                    last["end"] = seg["end"]
                else:
                    merged.append(seg)

        # Step 4: Final output conversion
        sections = []
        for seg in merged:
            sections.append({
                "name": seg["name"],
                "startPage": seg["start"] + 1,
                "endPage": seg["end"] + 1
            })
        # Step 5: Remove repeated sections (keep the first)
        by_name = {}
        for seg in sections:
            name = seg["name"]
            size = seg["endPage"] - seg["startPage"]
            if name not in by_name or size > (by_name[name]["endPage"] - by_name[name]["startPage"]):
                by_name[name] = seg

        # Convert dict → sorted list
        final = list(by_name.values())
        final.sort(key=lambda x: x["startPage"])

        return final
    
    def save_results_json(self, sections: List[dict], output_file: str):
        """Save results to a file."""
        try:
            data = {"sections": sections}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved results to {output_file}")
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, results: List[dict], total_time=None, pdf_conversion_time=None, avg_api_time=None, model: str = "gemini-2.5-flash"):
        """Print a summary of the processing results with timing information."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        print(f"\nMODEL: {model}")
        print(f"\nTIMING BREAKDOWN:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  PDF conversion: {pdf_conversion_time:.2f}s ({pdf_conversion_time/(total_time + 1)*100:.1f}%)")
        print(f"  API processing: {avg_api_time:.2f}s avg per page")
        print(f"  Other overhead: {total_time - pdf_conversion_time - (avg_api_time * len(results)):.2f}s")
        
        print(f"\nRESULTS:")
        for result in results:
            print(f"\nSection: {result['name']}")
            print(f"Pages: {result['startPage']} → {result['endPage']}")
            print("-" * 20)
    
    def check_api_health(self) -> bool:
        """Check if the API is running and accessible."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            print(f"API is healthy: {data.get('status', 'unknown')}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"API health check failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Process PDF files with AutoComply API")
    parser.add_argument("pdf_file", help="Path to the PDF file to process")
    parser.add_argument("--api-url", default="https://ai-models.autocomply.ca", 
                       help="API base URL (default: https://ai-models.autocomply.ca)")
    parser.add_argument("--api-key", default="sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c",
                       help="API key for authentication")
    parser.add_argument("--output", "-o", help="Output file to save results")
    parser.add_argument("--dpi", type=int, default=150, 
                       help="DPI for PDF to image conversion (default: 150)")
    parser.add_argument("--model", choices=["gpt-4o", "gemini-2.5-flash", "claude-sonnet-4.5"], default="gemini-2.5-flash",
                       help="AI model to use (default: gemini-2.5-flash)")
    parser.add_argument("--check-health", action="store_true",
                       help="Check API health and exit")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PDFProcessor(api_url=args.api_url, api_key=args.api_key)
    
    # Check API health if requested
    if args.check_health:
        if processor.check_api_health():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Validate PDF file
    pdf_path = Path(args.pdf_file)
    # if not pdf_path.exists():
    #     print(f"Error: PDF file not found: {pdf_path}")
    #     sys.exit(1)
    
    # Check API health before processing
    print("Checking API health...")
    if not processor.check_api_health():
        print("API is not accessible. Please ensure the server is running.")
        sys.exit(1)
    
    # Process the PDF
    print("Processing PDF and detecting sections…")
    sections = processor.process(pdf_path=str(pdf_path), dpi=args.dpi)

    # Save result.json automatically
    processor.save_results_json(sections, "result.json")

    # Summary
    processor.print_summary(
        sections,
        total_time=0,
        pdf_conversion_time=0,
        avg_api_time=0,
        model=args.model
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
