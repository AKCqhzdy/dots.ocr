import json
import os
import re

from PIL import Image
from loguru import logger
from rapidfuzz import fuzz

class CropImage:
    def __init__(self, image: Image, x_offset=0):
        self.image = image
        self.x_offset = x_offset


class SectionHeader:
    def __init__(self, text, category: str, bbox, level=None, source_block=None):
        self.text = text
        self.category = category
        self.bbox = bbox
        self.source_block = source_block

        self.level = level if level is not None else self._extract_level_from_text()
        self.new_level = None
        self.clean_text = self._clean_text()
        self.crop_img: CropImage = None

    @classmethod
    def from_info_block(cls, info_block, level=None):

        text = info_block.get("text", "")
        category = info_block["category"]
        bbox = info_block["bbox"]

        return cls(text, category, bbox, level=level, source_block=info_block)

    def _extract_level_from_text(self):
        """Extract header level from markdown-style text (# ## ### etc.)"""
        if self.category == "Title":
            return 0

        hash_match = re.match(r"^(#{1,6})\s+", self.text)
        bold_match = re.search(r"\*\*(.*?)\*\*", self.text)
        tt = 8
        if hash_match:
            tt = len(hash_match.group(1))
        elif bold_match:
            tt = 7

        if self.category == "Section-header":
            return tt
        elif self.category == "List-item":
            return 10 + tt
        else:
            return 20 + tt

    def _clean_text(self):
        """Remove markdown symbols from text"""
        self.text = re.sub(r"^#{1,6}\s+", "", self.text)
        self.text = re.sub(r"^\*\*(.*?)\*\*$", r"\1", self.text.strip())
        return self.text

    def reset_text_and_update(self):
        if self.new_level is None:
            return

        lines = self.clean_text.split("\n")
        formatted_lines = []

        for line in lines:
            if not line:
                continue

            if self.new_level == 1:
                self.category = "title"
                formatted_lines.append("# " + line)
            elif self.new_level == 7:
                self.category = "Section-header"
                formatted_lines.append("**" + line + "**")
            elif self.new_level == 8:
                self.category = "List-item"
                formatted_lines.append(line)
            else:
                self.category = "Section-header"
                formatted_lines.append("#" * self.new_level + " " + line)

        self.text = "\n".join(formatted_lines)

        if self.source_block:
            self.source_block["text"] = self.text
            self.source_block["category"] = self.category

    def crop_from_image(self, image, save_path=None):
        """Extract the bbox region from an image"""
        x1, y1, x2, y2 = self.bbox
        self.crop_img = CropImage(image.crop((x1, y1, x2, y2)), x_offset=x1)
        if save_path:
            self.crop_img.save(save_path)
        return self.crop_img

    def calc_dist(self, entry):

        def remove_prefix_number(text):
            cleaned = re.sub(r'^[\d\.\s]+', '', text)
            cleaned = cleaned.strip(' .\t\n')
            return cleaned
        ratio = fuzz.ratio(remove_prefix_number(self.clean_text), entry['text'])
        if ratio < 40:
            return float('inf')

        # Calculate the distance from a point to the bbox
        x = entry['to'][0]
        y = entry['to'][1]
        x1, y1, x2, y2 = self.bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            g_dist = 0
        else:
            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)
            g_dist = (dx ** 2 + dy ** 2) ** 0.5


        return g_dist

    def __repr__(self):
        if self.new_level is not None:
            return f"SectionHeader(new level={self.new_level}, bbox={self.bbox}, height={self.bbox[3]-self.bbox[1]}, width={self.bbox[2]-self.bbox[0]}, text='{self.clean_text}')"
        return f"SectionHeader(level={self.level}, bbox={self.bbox}, height={self.bbox[3]-self.bbox[1]}, width={self.bbox[2]-self.bbox[0]}, text='{self.clean_text}')"


class DirectoryStructure:
    def __init__(self):
        self.headers = []

    def add_header(self, info_block):
        header = SectionHeader.from_info_block(info_block)
        self.headers.append(header)

    def get_headers_by_level(self, level):
        return [h for h in self.headers if h.level == level]

    def get_all_headers(self):
        return self.headers

    def load_from_json(self, json_data):
        for info_block in json_data:
            if (
                info_block.get("category") == "Section-header"
                or info_block.get("category") == "Title"
                or info_block.get('category') == 'List-item'
            ):  #  or info_block.get('category') == 'List-item'  (Perhaps it is very long and recompute many time)
                self.add_header(info_block)

    def load_from_json_path(self, json_path):
        """Load Title, section headers and List-item from a JSON file"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.load_from_json(data["full_layout_info"])

    def extract_all_header_crops(self, image, save_dir=None):
        """Extract crops for all headers from an image"""
        crops = []

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, header in enumerate(self.headers):
            cropped_img = header.crop_from_image(image)
            crops.append(cropped_img)

            if save_dir:
                sanitized_text = re.sub(
                    r'[\\/*?:"<>|\n\r]', "_", header.clean_text[:20]
                )
                filename = f"header_{i}_level{header.level}_{sanitized_text}.png"
                save_path = os.path.join(save_dir, filename)
                cropped_img.image.save(save_path)
                print(f"Saved: {save_path}")

        return crops

    def __repr__(self):
        return f"DirectoryStructure({len(self.headers)} headers)"

    def rebuild_directory_by_toc(self, toc):
        """
        Rebuild directory structure based on pdf toc.
        Section Header levels are adjusted according to their positions in the toc.
        Other section headers or list items are assigned to list items.
        """

        max_level = 0
        for entry in toc:
            min_g_dist = float('inf')
            closest_header = None
            for header in self.headers:
                g_dist = header.calc_dist(entry)
                if g_dist < min_g_dist:
                    min_g_dist = g_dist
                    closest_header = header
            if closest_header:
                logger.debug(f"Matching TOC entry '{entry}' with header '{closest_header}' at distance g:{min_g_dist:.2f}")
                closest_header.new_level = entry["level"]
                max_level = max(max_level, closest_header.new_level)

        # Ensure that max_level is at least 5 to avoid:
        # 1. assigning too many headers to list items
        # 2. for those entries that are not matched, theirs new_level exceeds the max_level of any matched headers
        max_level = max(5, max_level)
        level_set = set()
        toc_result = []
        for header in self.headers:
            if header.new_level is None:
                level_set.add(header.level)
        for header in self.headers:
            if header.new_level is None:
                if header.category == "List-item" or max_level == 0:
                    header.new_level = 8
                else:
                    header.new_level = min(8, max_level + sorted(level_set).index(header.level) + 1)
            header.reset_text_and_update()

            entry = {
                "level": header.new_level,
                "text": header.clean_text,
                "bbox": [i for i in header.bbox],
            }
            toc_result.append(entry)
        return toc_result
            