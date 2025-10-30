from dots_ocr.model.layout_service import get_layout_pdf, sort_bboxes

def test_layout_detection():
    res = get_layout_pdf("/dots.ocr/app/input/monkeyocr/test/input/test_pdf/small2.pdf")
    print(res)
    return res

def test_layout_reader(bboxes, width, height):
    res = sort_bboxes(bboxes, width, height)
    print(res)
    return res


if __name__ == "__main__":
    layout = test_layout_detection()
    for page_layout in layout:
        bboxes = []
        width = page_layout['width']
        height = page_layout['height']
        for item in page_layout['full_layout_info']:
            # print(item)
            bboxes.append(item['bbox'])
        test_layout_reader(bboxes, width, height)