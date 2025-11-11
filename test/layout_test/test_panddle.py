from paddleocr import LayoutDetection
import fitz
from dots_ocr.utils.pdf_extractor import PdfExtractor

pdf_path = "/dots.ocr/app/input/monkeyocr/test/input/test_pdf/attn.pdf"

# pdf_extractor = PdfExtractor(pdf_path)

# img = pdf_extractor.page_to_image(1)
# print(img)


model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict(
    pdf_path, batch_size=1, layout_nms=True
)

i = 0
for res in output:
    print(res)
    i += 1
    res.save_to_img(save_path=f"test/output/test_panddle/res_{i}.jpg")
    res.save_to_json(save_path=f"test/output/test_panddle/res_{i}.json")