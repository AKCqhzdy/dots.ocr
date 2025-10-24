from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict(
    "/root/workspace/dots.ocr/test/small.pdf", batch_size=1, layout_nms=True
)
for res in output:
    print(type(res))
    res.print()
    res.save_to_img(save_path="test/outputs/test_panddle/res.jpg")
    res.save_to_json(save_path="test/outputs/test_panddle/res.json")
