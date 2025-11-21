import fitz
import os
import json
from PIL import Image
import re

class PdfExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_document = fitz.open(pdf_path)

    @property
    def is_structured(self):
        return len(self.pdf_document.get_toc()) > 0
    @property
    def num_pages(self):
        return self.pdf_document.page_count
    def page_size(self, page_no: int) -> int:
        rect = self.pdf_document[page_no].rect
        return rect.width, rect.height

    @staticmethod
    def to_image(
        page: fitz.Page,
        dpi: int = 72 # 72 is pdf default. usually we use 200 dpi for a more clear image. So don't forget that the bbox in pdf coordinate need to be scaled accordingly.
    ) -> Image.Image:
        """
        Convert a fitz.Page object directly to a PIL Image.
        
        Args:
            page: A fitz.Page instance (e.g., from doc[page_no])
            dpi: Target DPI for rendering (default 72, which is PDF native resolution)
        
        Returns:
            PIL.Image in RGB mode
        """
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img, zoom
    
    #TODO(zihao): fitz can get more infomation about the pdf structure, explore later
    # italic / bold / font size etc.
    @staticmethod
    def extract_text(
        page: fitz.Page,
        bbox : list = None
    ) -> str:
        if bbox:
            rect = fitz.Rect(bbox)
            text = page.get_text("text", clip=rect)
        else:
            text = page.get_text("text")
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text) 
        return text.strip()
    
    def extract_text_from_page(self, page_no: int, bbox: list = None) -> str:
        if page_no < 0 or page_no >= self.num_pages:
            raise ValueError(f"Page number {page_no} out of range [0, {self.num_pages-1}]")
        page = self.pdf_document[page_no]
        return self.extract_text(page, bbox)

    def page_to_image(self, page_no: int, dpi: int = 72) -> Image.Image:
        if page_no < 0 or page_no >= self.num_pages:
            raise ValueError(f"Page number {page_no} out of range [0, {self.num_pages-1}]")
        
        page = self.pdf_document[page_no]
        return self.to_image(page, dpi)
    
    def crop_bbox(self, page_no: int, bbox: list, output_path: str, dpi: int = 200):
        """
        Crops a specific bounding box from a page and saves it as an image.

        Args:
            page_no (int): The page number to crop from (0-indexed).
            bbox (list): The bounding box in PDF coordinates (x0, y0, x1, y1).
            output_path (str): The full path to save the cropped image (e.g., "test/outputs/crop/my_crop.png").
            dpi (int, optional): The resolution for the output image. Defaults to 200 for better quality.
        """
        if page_no < 0 or page_no >= self.num_pages:
            raise ValueError(f"Page number {page_no} out of range [0, {self.num_pages-1}]")
        
        full_page_image, zoom = self.page_to_image(page_no, dpi)
        
        cropped_image = full_page_image.crop(bbox)
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cropped_image.save(output_path)


    def get_clean_toc(self):
        '''
        Return:
            a dict of page number to list of TOC entries.
            Each entry is a dict with keys: level, text, to (the destination coordinates in PDF).
            Notice while using in image, need to convert by the scale factor of this page
        '''
        raw_toc = self.pdf_document.get_toc(simple=False)
        page_groups = {}
        
        for lvl, title, page, detail in raw_toc:
            page -= 1
            if detail.get("to") in [None, ""]: # skip invalid entries
                to_cor = None
            else:
                to_cor = list(detail.get("to", []))
                height = self.page_size(page)[1]
                to_cor[1] = height - to_cor[1]  # Convert PDF coordinate to top-left origin coordinate
            entry = {
                "level": lvl,
                "text": title,
                "to": to_cor 
            }
            
            if page in page_groups:
                page_groups[page].append(entry)
            else:
                page_groups[page] = [entry]
        
        return page_groups