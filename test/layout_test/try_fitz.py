import json
import os

import fitz


def extract_pdf_full_info(pdf_path, img_out_dir):
    doc = fitz.open(pdf_path)
    os.makedirs(img_out_dir, exist_ok=True)
    result = {
        "metadata": doc.metadata,
        "toc": doc.get_toc(simple=False),
        "page_count": len(doc),
        "is_encrypted": doc.is_encrypted,
        "permissions": doc.permissions,
        "pages": [],
    }

    def clean(obj):
        if isinstance(obj, bytes):
            return f"<{len(obj)} bytes>"
        elif isinstance(obj, (list, tuple)):
            return [clean(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, (fitz.Point, fitz.Rect, fitz.Quad)):
            return list(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    result["toc"] = clean(result["toc"])
    # result["metadata"] = clean(result["metadata"])

    # for page_idx in range(len(doc)):
    #     page = doc[page_idx]
    #     page_dict = page.get_text("dict")
    #     images = page.get_images(full=True)
    #     links = page.get_links()
    #     annots = [annot.info for annot in page.annots()]
    #     fonts = page.get_fonts(full=True)
    #     drawings = page.get_drawings()

    #     img_infos = []
    #     for img_index, img in enumerate(images):
    #         xref = img[0]
    #         base_image = doc.extract_image(xref)
    #         img_bytes = base_image["image"]
    #         ext = base_image["ext"]
    #         img_name = f"page{page_idx+1}_img{img_index}.{ext}"
    #         img_path = os.path.join(img_out_dir, img_name)
    #         with open(img_path, "wb") as f:
    #             f.write(img_bytes)
    #         img_infos.append(
    #             {
    #                 "img_name": img_name,
    #                 "bbox": img[1:5],
    #             }
    #         )

    #     result["pages"].append(
    #         {
    #             "page_number": page_idx + 1,
    #             "blocks": clean(page_dict["blocks"]),
    #             "images": img_infos,
    #             "links": clean(links),
    #             "annotations": clean(annots),
    #             "fonts": clean(fonts),
    #             "drawings": clean(drawings),
    #             "rect": list(page.rect),
    #             "rotation": page.rotation,
    #         }
    #     )

    doc.close()
    return result


if __name__ == "__main__":
    pdf_file = "/dots.ocr/app/input/monkeyocr/test/input/test_pdf/attn.pdf"
    imgs_dir = "/dots.ocr/test/outputs/pdf_extractors_outputs/output_images_fitz"
    data = extract_pdf_full_info(pdf_file, imgs_dir)
    with open(
        "/dots.ocr/test/outputs/pdf_extractors_outputs/output_images_fitz/layout_fitz.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)