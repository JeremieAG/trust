import concurrent.futures
import io
import os
import re
from typing import List

import fitz
from PIL import Image
import pytesseract


class PDFProcessor:

    def __init__(self, dpi: int = 150):
        self.dpi = dpi
        
    def pdf_page_to_image(self, pdf_path: str, page_num: int, dpi_multiplier: float = 1.0) -> bytes | None:
        """
        Convert a single PDF page to an image.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to convert (0-indexed)
            dpi_multiplier: Multiplier for DPI resolution

        Returns:
            Image bytes of the page or None if error occurs
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            mat = fitz.Matrix(self.dpi / 72 * dpi_multiplier, self.dpi / 72 * dpi_multiplier)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            doc.close()
            return img_data
        except Exception as e:
            print(f"Error converting page {page_num + 1} to image: {e}")
            return None

    def pdf_to_images(self, pdf_path: str) -> List[bytes]:
        """
        Convert PDF pages to images using 4 workers.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion

        Returns:
            List of image bytes for each page
        """
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
            if num_pages == 0:
                return []

            images: List[bytes | None] = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, img in enumerate(executor.map(self.pdf_page_to_image, [pdf_path]*num_pages, range(num_pages))):
                    if (i + 1) % 50 == 0:
                        print(f"Converted {i + 1}/{num_pages} pages to images...")
                    images.append(img)

            return [img for img in images if img is not None]

        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []

    def ocr_pdf(self, pdf_path: str) -> List[str]:
        """
        Perform OCR on a PDF file and return extracted text for each page.

        Args:
            pdf_path: Path to the PDF file
        Returns:
            List of extracted text for each page
        """
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return []

        images = self.pdf_to_images(pdf_path)
        if not images:
            return []
        print(f"Starting OCR on {len(images)} pages...")

        def _ocr_image(img_data: bytes, idx: int, isRetry: bool = False) -> str | None:
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    text = pytesseract.image_to_string(img)

                words = re.sub(r'[^a-zA-Z0-9 ]', '', text).split()
                if len(words) > 3: # Le OCR est valide si on a au moins 4 mots
                    return text

                if isRetry:
                    return None
                
                print(f"Low OCR confidence on page {idx + 1}, retrying with higher DPI...")
                new_img = self.pdf_page_to_image(pdf_path, idx, dpi_multiplier=2.0)
                return _ocr_image(new_img, idx, isRetry=True) 

            except Exception as e:
                print(f"OCR error on one page: {e}")
                return None

        result: List[str | None] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, text in enumerate(executor.map(_ocr_image, images, range(len(images)))):
                if (i + 1) % 50 == 0:
                    print(f"OCR processed {i + 1}/{len(images)} pages...")
                result.append(text)

        return result
