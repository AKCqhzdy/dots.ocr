import json

import pdfplumber


def extract_all_pdf_info(pdf_path, out_json_path):
    """
    使用 pdfplumber 提取 PDF 的全部结构化信息，包括元数据、页面信息、
    文本、表格、图片、矢量图形、超链接等。
    """
    output = {}
    with pdfplumber.open(pdf_path) as pdf:
        # 1. 提取 PDF 文档元数据
        output["metadata"] = pdf.metadata
        output["pages"] = []

        for page_idx, page in enumerate(pdf.pages):
            page_data = {
                "page_number": page.page_number,
                "width": float(page.width),
                "height": float(page.height),
                "rotation": page.rotation,
            }

            # 2. 提取表格 (使用默认策略)
            # page.extract_tables() 会返回一个包含所有找到的表格的列表
            # 每个表格都是一个 list of lists
            page_data["tables"] = page.extract_tables()

            # 3. 提取文字 (保留您原有的段落聚合逻辑)
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            paragraphs = []
            if words:
                # 简单的段落聚合逻辑：通过垂直位置和水平间距判断
                lines = {}
                for w in words:
                    top_int = int(w["top"])
                    if top_int not in lines:
                        lines[top_int] = []
                    lines[top_int].append(w)

                sorted_lines = sorted(lines.items())

                current_para_text = ""
                para_words = []

                for i, (top, line_words) in enumerate(sorted_lines):
                    line_words.sort(key=lambda w: w["x0"])
                    line_text = " ".join(w["text"] for w in line_words)

                    if not para_words:
                        current_para_text = line_text
                        para_words.extend(line_words)
                    else:
                        # 检查与上一行的行距，如果行距过大，则开启新段落
                        prev_top = sorted_lines[i - 1][0]
                        line_spacing = top - prev_top

                        # 启发式规则：行距大于平均字高的 1.5 倍，认为是新段落
                        avg_char_height = sum(w["height"] for w in para_words) / len(
                            para_words
                        )
                        if line_spacing > avg_char_height * 1.5:
                            paragraphs.append(
                                {
                                    "text": current_para_text.strip(),
                                    "bbox": [
                                        float(min(w["x0"] for w in para_words)),
                                        float(min(w["top"] for w in para_words)),
                                        float(max(w["x1"] for w in para_words)),
                                        float(max(w["bottom"] for w in para_words)),
                                    ],
                                }
                            )
                            current_para_text = line_text
                            para_words = line_words
                        else:
                            current_para_text += " " + line_text
                            para_words.extend(line_words)

                # 添加最后一个段落
                if para_words:
                    paragraphs.append(
                        {
                            "text": current_para_text.strip(),
                            "bbox": [
                                float(min(w["x0"] for w in para_words)),
                                float(min(w["top"] for w in para_words)),
                                float(max(w["x1"] for w in para_words)),
                                float(max(w["bottom"] for w in para_words)),
                            ],
                        }
                    )
            page_data["paragraphs"] = paragraphs

            # 4. 提取图片
            page_data["images"] = [
                {
                    "name": f"page_{page_idx+1}_img_{img.get('name', i)}",
                    "x0": float(img["x0"]),
                    "y0": float(img["y0"]),
                    "x1": float(img["x1"]),
                    "y1": float(img["y1"]),
                    "width": float(img["width"]),
                    "height": float(img["height"]),
                }
                for i, img in enumerate(page.images)
            ]

            # 5. 提取矢量图形
            page_data["lines"] = page.lines
            page_data["rects"] = page.rects
            page_data["curves"] = page.curves

            # 6. 提取超链接
            page_data["hyperlinks"] = [
                {
                    "uri": link["uri"],
                    "bbox": [link["x0"], link["top"], link["x1"], link["bottom"]],
                }
                for link in page.hyperlinks
            ]

            # 7. 提取每一个字符的详细信息 (注意：这会产生大量数据!)
            # page_data["chars"] = page.chars

            output["pages"].append(page_data)

    # 输出 JSON 文件，使用 Decimal 兼容的转换器
    def default_converter(o):
        if isinstance(o, (int, float, str, bool, type(None))):
            return o
        if isinstance(o, (list, tuple)):
            return [default_converter(item) for item in o]
        if isinstance(o, dict):
            return {k: default_converter(v) for k, v in o.items()}
        # 将 Decimal 等特殊类型转换为 float
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=default_converter)

    print(f"✅ 所有信息提取完成，已保存到 {out_json_path}")


if __name__ == "__main__":
    # 替换为你的 PDF 文件路径
    pdf_file = "/dots.ocr/test/datas/attn.pdf"
    out_json = "/dots.ocr/test/outputs/pdf_extractors_outputs/layout_pdfplumber.json"
    extract_all_pdf_info(pdf_file, out_json)