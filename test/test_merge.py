import asyncio
import os
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dots_ocr.utils.doc_utils import load_images_from_pdf
from PIL import Image
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS

dots_parser = DotsOCRParser()

class CropImage:
    def __init__(self, image, x_offset=0):
        self.image = image
        self.x_offset = x_offset

class SectionHeader:
    def __init__(self, text, bbox, level=None):
        self.text = text
        self.bbox = bbox
        self.level = level if level is not None else self._extract_level_from_text()
        self.new_level = None
        self.clean_text = self._clean_text()
        self.crop_img: CropImage = None
    
    def _extract_level_from_text(self):
        """Extract header level from markdown-style text (# ## ### etc.)"""
        hash_match = re.match(r'^(#{1,6})\s+', self.text)
        if hash_match:
            return len(hash_match.group(1))
        
        bold_match = re.match(r'^\*\*(.*?)\*\*$', self.text.strip())
        if bold_match:
            return 7
        
        return 8
    
    def _clean_text(self):
        """Remove markdown symbols from text"""
        self.text = re.sub(r'^#{1,6}\s+', '', self.text)
        self.text = re.sub(r'^\*\*(.*?)\*\*$', r'\1', self.text.strip())
        return self.text
    
    def crop_from_image(self, image, save_path=None):
        """Extract the bbox region from an image"""
        x1, y1, x2, y2 = self.bbox
        self.crop_img = CropImage(image.crop((x1, y1, x2, y2)), x_offset=x1)
        if save_path:
            self.crop_img.save(save_path)
        return self.crop_img
    
    def __repr__(self):
        return f"SectionHeader(level={self.level}, bbox={self.bbox}, height={self.bbox[3]-self.bbox[1]}, width={self.bbox[2]-self.bbox[0]}, text='{self.clean_text}')"

class DirectoryStructure:
    def __init__(self):
        self.headers = []
    
    def add_header(self, text, bbox):
        header = SectionHeader(text, bbox)
        self.headers.append(header)
    
    def get_headers_by_level(self, level):
        return [h for h in self.headers if h.level == level]
    
    def get_all_headers(self):
        return self.headers
    
    def load_from_json(self, json_path):
        """Load section headers from a JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for info_block in data['full_layout_info']:
                if info_block.get('category') == 'Section-header':
                    self.add_header(info_block['text'], info_block['bbox'])
    
    def extract_all_header_crops(self, image, save_dir=None):
        """Extract crops for all headers from an image"""
        crops = []
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i, header in enumerate(self.headers):
            cropped_img = CropImage(header.crop_from_image(image), x_offset=header.bbox[0])
            crops.append(cropped_img)
            
            if save_dir:
                filename = f"header_{i}_level{header.level}_{header.clean_text[:20].replace(' ', '_')}.png"
                save_path = os.path.join(save_dir, filename)
                cropped_img.save(save_path)
                print(f"Saved: {save_path}")
        
        return crops
    
    def __repr__(self):
        return f"DirectoryStructure({len(self.headers)} headers)"

def extract_and_print_headers_with_bbox():
    """Extract section headers with bbox from JSON files"""
    directorys = []
    
    for i in range(14):
        json_path = f"/dots.ocr/test/test_merge/test{i}.json"
        if os.path.exists(json_path):
            dir_structure = DirectoryStructure()
            dir_structure.load_from_json(json_path)
            directorys.append(dir_structure)
            
            print(f"File {i}: {dir_structure}")
            for header in dir_structure.get_all_headers():
                print(f"  {header}")
            print('----------------------')
    
    return directorys

def extract_single_directory_headers(json_path):
    dir_structure = DirectoryStructure()
    if os.path.exists(json_path):
        dir_structure.load_from_json(json_path)
    return dir_structure

CPU_EXECUTOR = ProcessPoolExecutor(max_workers=os.cpu_count())
async def get_image(input_path):

    loop = asyncio.get_running_loop()
    
    print(f"Loading PDF: {input_path}")
    
    images = await loop.run_in_executor(CPU_EXECUTOR, load_images_from_pdf, input_path)

    return images


def concat_images(images, mode="vertical"):
    if mode == "vertical":
        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        new_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height
    elif mode == "horizontal":
        total_width = sum(img.width for img in images)
        total_height = max(img.height for img in images)
        new_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
    
    return new_img

    
async def gen_page_output():
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    images = await get_image(input_pdf_path)

    tasks = []
    for i in range(len(images)-1):
        tasks.append(
            dots_parser._parse_single_image(
                images[i],
                prompt_mode="prompt_layout_all_en",
                save_dir=f"/dots.ocr/test/test_merge",
                save_name=f"test{i}"
            )
        )

    await asyncio.gather(*tasks)


async def gen_concat_output():
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    images = await get_image(input_pdf_path)

    tasks = []
    for i in range(len(images)-1):
        new_image = concat_images([images[i], images[i+1]], mode="vertical")
        tasks.append(
            dots_parser._parse_single_image(
                new_image,
                prompt_mode="prompt_layout_all_en",
                save_dir=f"/dots.ocr/test/test_merge{i}",
                save_name="test"
            )
        )

    await asyncio.gather(*tasks)

def extract_single_page_headers(page_index, save_crops=True):
    """Extract headers from a single page"""
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    
    # Load images from PDF
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    images = loop.run_until_complete(get_image(input_pdf_path))
    loop.close()
    
    if page_index >= len(images):
        print(f"Page {page_index} not found in PDF")
        return None
    
    json_path = f"/dots.ocr/test/test_merge/test{page_index}.json"
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return None
    
    # Load directory structure
    dir_structure = extract_single_directory_headers(json_path)
    
    if save_crops:
        # Extract and save header crops
        save_dir = f"/dots.ocr/test/header_crops/page_{page_index}"
        crops = dir_structure.extract_all_header_crops(images[page_index], save_dir)
    else:
        # Just extract crops without saving
        crops = []
        for header in dir_structure.headers:
            crops.append(header.crop_from_image(images[page_index]))
    
    return dir_structure, crops


def extract_header_images_from_pdf():
    """Extract header images from PDF pages"""
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    
    # Load images from PDF
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    images = loop.run_until_complete(get_image(input_pdf_path))
    loop.close()
    
    # Process each page
    directorys = []
    for i in range(len(images)):
        json_path = f"/dots.ocr/test/test_merge/test{i}.json"
        if os.path.exists(json_path):
            print(f"Processing page {i}...")
            
            # Load directory structure
            dir_structure = extract_single_directory_headers(json_path)
            
            # Extract header crops
            save_dir = f"/dots.ocr/test/header_crops/page_{i}"
            dir_structure.extract_all_header_crops(images[i], save_dir)
            directorys.append(dir_structure)
            
            print(f"Extracted {len(dir_structure.headers)} headers from page {i}")
            print('----------------------')

    return directorys

def merge_crops_and_parse(crops, save_dir=None, save_name=None):
    if not crops:
        print("No crops provided")
        return None
    
    total_width = max(crop.image.width + crop.x_offset for crop in crops)
    total_height = sum(crop.image.height for crop in crops)
    
    merged_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    
    y_offset = 0
    for crop in crops:
        merged_image.paste(crop.image, (crop.x_offset, y_offset))
        y_offset += crop.image.height
    
    print(f"Merged {len(crops)} crops into image of size {total_width}x{total_height}")

    if MAX_PIXELS < total_width * total_height:
        raise ValueError("Merged image exceeds size limit of 11289600 pixels")
    
    try:
        if save_dir:
            result = asyncio.run(dots_parser._parse_single_image(
                merged_image,
                prompt_mode="prompt_layout_all_en",
                save_dir=save_dir,
                save_name=save_name
            ))
        else:
            result = asyncio.run(dots_parser._parse_single_image_do_not_save(
                merged_image,
                prompt_mode="prompt_layout_all_en"
            ))
        
        print(f"OCR parsing completed for merged crops")
        return result
        
    except Exception as e:
        print(f"Error during OCR parsing: {e}")
        return None



class Reranker():

    def __init__(now_level=9, sum_height=0, max_width=0):
        now_level = 9
        sum_height = 0
        max_width = 0
        highest_list = []
        check_list = []
    def clear(self):
        self.now_level = 9
        self.sum_height = 0
        self.max_width = 0
        self.highest_list = []
        self.check_list = []

    def identify_highest_headers(self, layout_info):
        highest_level = 9
        idx_num = []
        check_idx = 0
        for info_block in enumerate(layout_info):
            category = info_block.get('category')
            now_header = SectionHeader(info_block['text'], info_block['bbox'])
            if category == 'Title':
                level = 1
                if level < highest_level:
                    highest_level = level
                    idx_num = []
            elif info_block == 'Section-header':
                level = now_header.level
                if level < highest_level:
                    highest_level = level
                    idx_num = []
            
            if level == highest_level:
                while check_idx < len(self.check_list):
                    if match_header(self.check_list[check_idx], now_header):
                        idx_num.append(check_idx)
                    check_idx += 1
        
        if 0 not in idx_num:
            highest_list = []

        new_check_list = []
        SAVE_NUM = 10 
        for i in range(SAVE_NUM, len(idx_num)):
            if i < len(idx_num) - SAVE_NUM:
                self.highest_list.append(self.check_list[idx_num[i]])
            else:   
                new_check_list.append(self.check_list[idx_num[i]])
        self.check_list = new_check_list
            


    def start_rerank(self):
        merge_cops = []
        for header in self.check_list:
            merge_cops.append(header.crop_img)
        cells = merge_crops_and_parse(merge_cops, save_dir=None, save_name=None)

        self.identify_highest_headers(cells["full_layout_info"])

    def insert(self, header: SectionHeader):
        self.sum_height += header.bbox[3] - header.bbox[1]
        self.max_width = max(self.max_width, header.crop_img.x_offset + header.crop_img.width)
        if header.level < self.now_level:
            self.now_level = header.level

        self.check_list.append(header)

    def try_to_insert(self, header: SectionHeader):
        if (self.sum_height + header.bbox[3] - header.bbox[1]) * max(self.max_width, header.crop_img.x_offset + header.crop_img.width) > MAX_PIXELS:
            self.start_rerank()
        self.insert(header)


def reset_header_level(directorys):

    n = len(len(directorys))

    all_sorted_indices = []
    for dir_structure in directorys:
        sorted_indices = sorted(range(len(dir_structure.headers)), key=lambda i: dir_structure.headers[i].level)
        all_sorted_indices.append(sorted_indices)
    
    begin = [0] * n

    reranker = Reranker()
    LEVEL_NUM = 8
    for _ in range(LEVEL_NUM):
        for i, sorted_indices in enumerate(all_sorted_indices):
            dir_structure = directorys[i]
            j = begin[i]
            headers = dir_structure.get_all_headers()
            max_level_in_this_page = -1
            
            while j < len(sorted_indices):
                if max_level_in_this_page == -1:
                    max_level_in_this_page == headers[sorted_indices[j]].level
                
                if headers[sorted_indices[j]].level > max_level_in_this_page:
                    reranker.try_to_insert(headers[sorted_indices[j]])
                else:
                    break
                j += 1

        reranker.start_rerank()
        reranker.assign_new_level()
        reranker.clear()

        # elimite the headers that have been assigned new_level (might not be a continuous sequence)
        for i, sorted_indices in enumerate(all_sorted_indices):
            dir_structure = directorys[i]
            begin[i] = 0
            new_sorted_indices = []
            for j in range(len(sorted_indices)):
                if dir_structure.headers[sorted_indices[j]].new_level is None:
                    new_sorted_indices.append(sorted_indices[j])
            all_sorted_indices[i] = new_sorted_indices



if __name__ == "__main__":

    # asyncio.run(gen_concat_output())

    # directorys = extract_and_print_headers_with_bbox()



    # asyncio.run(gen_page_output())
    directorys = extract_header_images_from_pdf()

    reset_header_level(directorys)

        # crops = []
        # cc = 0
        # for directory in directorys:
        #     if cc != 0:
        #         for header in directory.get_all_headers():
        #             crops.append(header.crop_img)

        # result = merge_crops_and_parse(crops, save_dir="/dots.ocr/test/merged_results", save_name="merged_crops")
        # print(result)


