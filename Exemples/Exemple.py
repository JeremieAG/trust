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
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def process_pdf_file(self, pdf_path: str, prompt: str, output_file: Optional[str] = None, model: str = "gemini-2.5-flash") -> bool:
        """
        Process an entire PDF file page by page.
        
        Args:
            pdf_path: Path to the PDF file
            prompt: Processing prompt
            output_file: Optional output file to save results
            model: Model to use ('gpt-4o', 'gemini-2.5-flash', or 'claude-sonnet-4.5', default: 'gemini-2.5-flash')
            
        Returns:
            True if successful, False otherwise
        """
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        print(f"Using model: {model}")
        
        # Convert PDF to images
        print("Converting PDF to images...")
        pdf_conversion_start = time.time()
        images = self.pdf_to_images(pdf_path)
        pdf_conversion_time = time.time() - pdf_conversion_start
        
        if not images:
            print("Failed to convert PDF to images")
            return False
        
        print(f"Found {len(images)} pages (conversion took {pdf_conversion_time:.2f}s)")
        
        results = []
        api_processing_times = []
        
        for i, image_bytes in enumerate(images):
            print(f"Processing page {i + 1}/{len(images)}...")
            
            # Convert to base64
            base64_start = time.time()
            image_b64 = self.image_to_base64(image_bytes)
            base64_time = time.time() - base64_start
            
            # Send to API
            api_start = time.time()
            result = self.process_pdf_page(image_b64, prompt, model=model)
            api_time = time.time() - api_start
            api_processing_times.append(api_time)
            
            if result:
                page_result = {
                    "page": i + 1,
                    "result": result.get("result", ""),
                    "api_time": api_time,
                    "base64_time": base64_time
                }
                results.append(page_result)
                print(f"✓ Page {i + 1} processed successfully (API: {api_time:.2f}s, Base64: {base64_time:.2f}s)")
            else:
                print(f"✗ Failed to process page {i + 1}")
                results.append({
                    "page": i + 1,
                    "result": "ERROR: Failed to process this page",
                    "api_time": api_time,
                    "base64_time": base64_time
                })
        
        total_time = time.time() - total_start_time
        avg_api_time = sum(api_processing_times) / len(api_processing_times) if api_processing_times else 0
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        # Print summary with timing
        self.print_summary(results, total_time, pdf_conversion_time, avg_api_time, model)
        
        return True
    
    def save_results(self, results: List[dict], output_file: str):
        """Save results to a file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"=== PAGE {result['page']} ===\n")
                    f.write(f"{result['result']}\n\n")
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, results: List[dict], total_time: float, pdf_conversion_time: float, avg_api_time: float, model: str = "gemini-2.5-flash"):
        """Print a summary of the processing results with timing information."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        print(f"\nMODEL: {model}")
        print(f"\nTIMING BREAKDOWN:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  PDF conversion: {pdf_conversion_time:.2f}s ({pdf_conversion_time/total_time*100:.1f}%)")
        print(f"  API processing: {avg_api_time:.2f}s avg per page")
        print(f"  Other overhead: {total_time - pdf_conversion_time - (avg_api_time * len(results)):.2f}s")
        
        print(f"\nRESULTS:")
        for result in results:
            print(f"\nPage {result['page']}:")
            print("-" * 20)
            if 'api_time' in result:
                print(f"API time: {result['api_time']:.2f}s, Base64 time: {result['base64_time']:.2f}s")
            # Truncate long results for display
            text = result['result']
            if len(text) > 200:
                text = text[:200] + "..."
            print(text)
    
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
    parser.add_argument("prompt", help="Processing prompt for the AI")
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
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Check API health before processing
    print("Checking API health...")
    if not processor.check_api_health():
        print("API is not accessible. Please ensure the server is running.")
        sys.exit(1)
    
    # Process the PDF
    success = processor.process_pdf_file(
        pdf_path=str(pdf_path),
        prompt=args.prompt,
        output_file=args.output,
        model=args.model
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
