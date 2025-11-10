import json
from dots_ocr.model.layout_service import sort_bboxes

FILE_PATH = "/dots.ocr/app/output/monkeyocr/test/output/test_pdf/small/small_page_0.json"
PAGE_NO = 0

def extract_bboxes_from_page(file_path, page_number):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for page in data:
        if page['page_no'] == page_number:
            bboxes = [item['bbox'] for item in page['full_layout_info']]
            return page, bboxes
    
    return []

if __name__ == "__main__":
    
    page, bboxes = extract_bboxes_from_page(FILE_PATH, PAGE_NO)
    print(bboxes)
    order = sort_bboxes(bboxes,page['width'], page['height'])
    for idx in order:
        print(page["full_layout_info"][:idx])