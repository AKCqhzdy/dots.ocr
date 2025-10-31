import json
import os
import re

from PIL import Image

from dots_ocr.utils.consts import MAX_LENGTH, MAX_PIXELS
from dots_ocr.utils.directory_entry import CropImage, SectionHeader, DirectoryStructure
from dots_ocr.utils.page_parser import PageParser

MAX_LEVEL = 100
class Reranker:

    def __init__(self, upper_level=1, now_level=MAX_LEVEL, sum_height=0, max_width=0):
        self.upper_level = upper_level
        self.now_level = now_level  # current highest level in this batch
        self.sum_height = sum_height
        self.max_width = max_width
        self.highest_list = []
        self.check_list = []

        self.count = 0
        os.makedirs("/dots.ocr/test/output", exist_ok=True)
        self.parser = PageParser()

    def clear(self, upper_level=None):
        if upper_level is None:
            upper_level = self.now_level + 1
        self.upper_level = upper_level
        self.now_level = MAX_LEVEL
        self.sum_height = 0
        self.max_width = 0
        self.highest_list = []
        self.check_list = []

    def identify_highest_headers(self, cells, y_offset_list):

        print("---------------------------------------------------")

        highest_level = MAX_LEVEL
        highest_headers_idx = []
        h_idx = 0
        highest_list_is_empty = True if len(self.highest_list) == 0 else False

        for i, header in enumerate(self.check_list):
            print(y_offset_list[i], header)
        for info_block in cells["full_layout_info"]:
            print(info_block)

        for info_block in cells["full_layout_info"]:
            # print(info_block)
            now_header = SectionHeader.from_info_block(info_block)

            match_header_idx = []
            while (
                h_idx < len(y_offset_list)
                and max(0, y_offset_list[h_idx] - info_block["bbox"][3])
                / (
                    y_offset_list[h_idx]
                    - (0 if h_idx == 0 else y_offset_list[h_idx - 1])
                )
                < 0.33
            ):
                print(y_offset_list[h_idx], info_block["bbox"][3])
                match_header_idx.append(h_idx)
                h_idx += 1

            print(f"highest_level: {highest_level}     level: {now_header.level}")
            if now_header.level < highest_level:
                highest_level = now_header.level
                highest_headers_idx = []
            if now_header.level == highest_level:
                highest_headers_idx.extend(match_header_idx)

            # print(highest_headers_idx)

        if not highest_list_is_empty and 0 not in highest_headers_idx:
            self.highest_list = []

        print(highest_headers_idx)
        new_check_list = []
        SAVE_NUM = 10
        for i in range(len(highest_headers_idx)):
            if i < len(highest_headers_idx) - SAVE_NUM:
                self.highest_list.append(self.check_list[highest_headers_idx[i]])
            else:
                new_check_list.append(self.check_list[highest_headers_idx[i]])
        self.check_list = new_check_list

    async def start_rerank(self):
        merge_cops = []
        for header in self.check_list:
            merge_cops.append(header.crop_img)
            # print(header)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        cells, y_offset_list = await self.merge_crops_and_parse(
            merge_cops, save_dir=None, save_name=None
        )
        self.identify_highest_headers(cells, y_offset_list)

    def insert(self, header: SectionHeader):
        self.sum_height += header.bbox[3] - header.bbox[1]
        self.max_width = max(self.max_width, header.bbox[2] - header.bbox[0])
        if header.level < self.now_level:
            self.now_level = max(self.upper_level, header.level)

        self.check_list.append(header)

    async def try_to_insert(self, header: SectionHeader):
        h = self.sum_height + header.bbox[3] - header.bbox[1]
        w = max(self.max_width, header.bbox[2] - header.bbox[0])
        if h * w > MAX_PIXELS or h > MAX_LENGTH:
            await self.start_rerank()
            self.sum_height = 0
            self.max_width = 0
        self.insert(header)

    def assign_new_level(self):
        print(len(self.highest_list), len(self.check_list), self.now_level)
        for header in self.highest_list:
            header.new_level = self.now_level
        for header in self.check_list:
            header.new_level = self.now_level

    async def merge_crops_and_parse(self, crops, save_dir=None, save_name=None):
        if not crops:
            print("No crops provided")
            return None

        total_width = max(crop.image.width for crop in crops)
        total_height = sum(crop.image.height for crop in crops)

        merged_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

        y_offset_list = []
        y_offset = 0
        for crop in crops:
            merged_image.paste(crop.image, (0, y_offset))
            y_offset += crop.image.height
            y_offset_list.append(y_offset)

        save_dir = f"test/output/output{self.count}"
        save_name = f"output{self.count}"
        print(
            f"Merged {len(crops)} crops into image of size {total_width}x{total_height}"
        )
        assert MAX_PIXELS >= total_width * total_height

        try:
            if save_dir:
                result_debug = await self.parser._parse_single_image(
                    merged_image,
                    prompt_mode="prompt_layout_all_en",
                    save_dir=save_dir,
                    save_name=save_name,
                )
            result = await self.parser._parse_single_image(
                merged_image,
                prompt_mode="prompt_layout_all_en",
                save_dir=None,
                save_name=None,
            )

            print(f"OCR parsing completed for merged crops")
            merged_image.save(f"test/output/output{self.count}/merged_image.jpg")
            self.count += 1
            return result, y_offset_list

        except Exception as e:
            print(f"Error during OCR parsing: {e}")
            return None


class DirectoryCleaner:
    def __init__(self):
        self.reranker = Reranker()

    async def reset_header_level(self, cells_list, images_origin):

        assert len(cells_list) == len(images_origin)
        directorys = []
        all_sorted_indices = []
        for i, page in enumerate(cells_list):
            dir_structure = DirectoryStructure()
            dir_structure.load_from_json(page["full_layout_info"])
            dir_structure.extract_all_header_crops(image=images_origin[i])
            directorys.append(dir_structure)

            sorted_indices = sorted(
                range(len(dir_structure.headers)),
                key=lambda i: dir_structure.headers[i].level,
            )
            all_sorted_indices.append(sorted_indices)

        begin = [0] * len(directorys)

        self.reranker.clear(upper_level=1)
        LEVEL_NUM = 8
        while self.reranker.upper_level <= LEVEL_NUM:
            print(f"llalalalal  {self.reranker.upper_level}")
            for i, sorted_indices in enumerate(all_sorted_indices):
                dir_structure = directorys[i]
                j = begin[i]
                headers = dir_structure.get_all_headers()
                now_level_in_this_page = -1

                allow = 2
                while j < len(sorted_indices):
                    if now_level_in_this_page == -1:
                        now_level_in_this_page = headers[sorted_indices[j]].level

                    if headers[sorted_indices[j]].level < now_level_in_this_page:
                        allow -= 1
                        now_level_in_this_page = headers[sorted_indices[j]].level
                        if allow < 0:
                            break
                    # print(headers[sorted_indices[j]])
                    await self.reranker.try_to_insert(headers[sorted_indices[j]])

                    j += 1
                print(f"========================{i}")

            if len(self.reranker.check_list) == 0:
                break

            await self.reranker.start_rerank()
            self.reranker.assign_new_level()
            self.reranker.clear()

            # elimite the headers that have been assigned new_level (might not be a continuous subsequence)
            for i, sorted_indices in enumerate(all_sorted_indices):
                dir_structure = directorys[i]
                begin[i] = 0
                new_sorted_indices = []
                for j in range(len(sorted_indices)):
                    if dir_structure.headers[sorted_indices[j]].new_level is None:
                        new_sorted_indices.append(sorted_indices[j])
                all_sorted_indices[i] = new_sorted_indices

        for dir_structure in directorys:
            for header in dir_structure.get_all_headers():
                if header.new_level is None:
                    header.new_level = 8
                header._reset_text_and_update()

                print(f"{header.text}")
                # print(f"{'#'*header.new_level} {header.clean_text}")
                # print(f"  {header}")
            print("----------------------")
