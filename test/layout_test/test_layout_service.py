import json
from dots_ocr.model.layout_service import get_layout_pdf, get_layout_image, sort_bboxes
from dots_ocr.utils.pdf_extractor import PdfExtractor
import asyncio

PDF_FILE_PATH = "/dots.ocr/app/input/monkeyocr/test/input/test_pdf/small.pdf"
JSON_FILE_PATH = "/dots.ocr/test/output/test_panddle/res_1.json"
PAGE_NO = 0

async def test_layout_detection():
    res = await get_layout_pdf(PDF_FILE_PATH)
    print(res)
    return res

def test_layout_reader(bboxes, width, height):
    res = sort_bboxes(bboxes, width, height)
    print(res)
    return res

def test_relation():
    
    def extract_bboxes_from_page(file_path, page_number):
        
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)
        
        # for page in data:
        #     if page['page_no'] == page_number:
        #         bboxes = [item['bbox'] for item in page['full_layout_info']]
        #         return page, bboxes
        
        # return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data["boxes"]
        page = {
            "width": 1700,
            "height": 2200,
        }
        return page, [item['coordinate'] for item in data]

    page, bboxes = extract_bboxes_from_page(JSON_FILE_PATH, PAGE_NO)
    print(bboxes)
    order = asyncio.run(sort_bboxes(bboxes,page['width'], page['height']))
    print(order)
    # for idx in order:
    #     print(page["full_layout_info"][:idx])

if __name__ == "__main__":

    # layout = asyncio.run(test_layout_detection())
    # for page_layout in layout:
    #     bboxes = []
    #     width = page_layout['width']
    #     height = page_layout['height']
    #     for item in page_layout['full_layout_info']:
    #         # print(item)
    #         bboxes.append(item['bbox'])
    #     test_layout_reader(bboxes, width, height)

    test_relation()




    # async def main():
    #     pdf_extractor = PdfExtractor(PDF_FILE_PATH)
        
    #     img1 = pdf_extractor.page_to_image(0)
    #     img2 = pdf_extractor.page_to_image(1)
    #     results = await asyncio.gather(
    #         get_layout_image(img1),
    #         get_layout_image(img2)
    #     )

    #     layout1, layout2 = results
    #     print(layout1)
    #     print(layout2)

    # asyncio.run(main())
