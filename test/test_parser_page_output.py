import asyncio

from dots_ocr.parser import DotsOCRParser


async def main():
    dots_parser = DotsOCRParser()
    await dots_parser.parse_pdf_rebuild_directory(
        "/dots.ocr/test/data/PGhandbook.pdf", "prompt_layout_all_en"
    )


if __name__ == "__main__":
    asyncio.run(main())
