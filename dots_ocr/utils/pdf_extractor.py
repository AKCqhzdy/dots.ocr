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

    #TODO(zihao): fitz can get more infomation about the pdf structure, explore later
    # italic / bold / font size etc.
    def extract_text(
        self,
        page_no,
        bbox : tuple = None
    ) -> str:
        page = self.pdf_document[page_no]
        if bbox:
            rect = fitz.Rect(bbox)
            text = page.get_text("text", clip=rect)
        else:
            text = page.get_text("text")
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text) 
        return text.strip()

    def page_to_image(
        self,
        page_no: int,
        dpi: int = 72, # 72 is pdf default. dpi can only be 72 if layoutjson2md(used in save_result) we save crops instead of the descriptions.
    ) -> Image.Image:
        if page_no < 0 or page_no >= self.num_pages:
            raise ValueError(f"Page number {page_no} out of range [1, {self.num_pages}]")
    
        page = self.pdf_document[page_no]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        return img

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
            assert detail.get("to") not in [None, ""], \
                f"TOC entry destination not found for title: {title}"
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