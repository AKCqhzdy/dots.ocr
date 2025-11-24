import re

def transform_latex(latex_input):
    """
    Transforms LaTeX from \\[...\\] format to $$...$$ format.
    Smartly cleans OCR artifacts while protecting valid LaTeX commands.
    e.g., "l r a t e" -> "lrate", but "n \\cdot d" -> "n \\cdot d" (keeps space).
    """
    # 1. Extract content inside \[ ... \]
    match = re.search(r'\\\[(.*?)\\\]', latex_input, re.DOTALL)
    content = match.group(1).strip() if match else latex_input.strip()

    # 2. Basic Cleanup
    content = re.sub(r'\\ +', r'\\', content)
    content = re.sub(r'\s*\\_\s*', r'\\_', content)
    content = re.sub(r'\} +(?=[a-zA-Z])', '}', content)

    pattern = r'(\\?)([a-zA-Z]+)\s+([a-zA-Z])'
    def merge_check(m):
        prefix_slash, word, next_char = m.groups()
        if prefix_slash:
            return m.group(0)
        else:
            return word + next_char

    for _ in range(5):
        new_content = re.sub(pattern, merge_check, content)
        if new_content == content:
            break
        content = new_content

    return f"$${content}$$"

def transform_table(table_input):
    """
    Transforms custom table tokens into a human-readable Markdown table.
    Since Markdown doesn't support merged cells, <ucel> and <lcel> are treated 
    as empty cells to maintain grid alignment.
    """
    # 1. Parse rows by <nl>
    raw_rows = [r for r in table_input.split('<nl>') if r.strip()]
    if not raw_rows:
        return ""

    # 2. Extract tokens into a Grid (List of Lists)
    # We treat <ucel> and <lcel> as empty strings for Markdown visual compatibility
    def _clean_ocr_math_latex(text: str) -> str:
        if r'\(' in text and r'\)' in text:
            text = text.replace(r'\(', '').replace(r'\)', '')
            return f'${text}$'
        return text
    grid = []
    for row_str in raw_rows:
        # Regex: captures tag in group 1, content in group 2
        tokens = re.findall(r'(<[a-z]{4}>)([^<]*)', row_str)
        row_data = []
        for tag, content in tokens:
            cleaned = _clean_ocr_math_latex(content.strip())
            if tag in ['<ucel>', '<lcel>']:
                row_data.append("") # Merged placeholders become empty cells
            else:
                row_data.append(cleaned)
        grid.append(row_data)

    if not grid:
        return ""

    # 3. Normalize grid width (handle potential ragged rows)
    max_cols = max(len(row) for row in grid)
    for row in grid:
        while len(row) < max_cols:
            row.append("")

    # 4. Calculate column widths for alignment
    col_widths = [0] * max_cols
    for row in grid:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # 5. Build Markdown Table String
    output_lines = []
    
    def build_row(row_items):
        line = "|"
        for i, item in enumerate(row_items):
            # Pad item to match column width
            line += f" {item:<{col_widths[i]}} |"
        return line
    output_lines.append(build_row(grid[0]))
    
    # Separator (e.g., |---|---|)
    separator = "|"
    for w in col_widths:
        separator += f" {'-' * w} |"
    output_lines.append(separator)

    # Body
    for row in grid[1:]:
        output_lines.append(build_row(row))

    return "\n".join(output_lines)


if __name__ == "__main__":
    case1 = r"\[\overline{{8.0\cdot 10^{20}}}\]"
    print(f"Input 1: {case1}")
    print(f"Output 1: {transform_latex(case1)}\n")

    case2 = r"\[l r a t e=d_{\mathrm{m o d e l}}^{-0.5}\cdot\operatorname*{m i n}(s t e p\_{n} u m^{-0.5},s t e p\_{n} u m\cdot w a r m u p\_{s} t e p s^{-1.5})\]"
    print(f"Input 2: {case2}")
    print(f"Output 2: {transform_latex(case2)}")

    data1 = r"<fcel>Parser<fcel>Training<fcel>WSJ 23 F1<nl><fcel>Vinyals & Kaiser et al. (2014) [37]<fcel>WSJ only, discriminative<fcel>88.3<nl><fcel>Petrov et al. (2006) [29]<fcel>WSJ only, discriminative<fcel>90.4<nl><fcel>Zhu et al. (2013) [40]<fcel>WSJ only, discriminative<fcel>90.4<nl><fcel>Dyer et al. (2016) [8]<fcel>WSJ only, discriminative<fcel>91.7<nl><fcel>Transformer (4 layers)<fcel>WSJ only, discriminative<fcel>91.3<nl><fcel>Zhu et al. (2013) [40]<fcel>semi-supervised<fcel>91.3<nl><fcel>Huang & Harper (2009) [14]<fcel>semi-supervised<fcel>91.3<nl><fcel>McClosky et al. (2006) [26]<fcel>semi-supervised<fcel>92.1<nl><fcel>Vinyals & Kaiser et al. (2014) [37]<fcel>semi-supervised<fcel>92.1<nl><fcel>Transformer (4 layers)<fcel>semi-supervised<fcel>92.7<nl><fcel>Luong et al. (2015) [23]<fcel>multi-task<fcel>93.0<nl><fcel>Dyer et al. (2016) [8]<fcel>generative<fcel>93.3<nl>"
    
    print("--- Example 1 Output ---")
    print(transform_table(data1))
    print("\n")