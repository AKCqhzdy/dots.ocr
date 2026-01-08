from loguru import logger
from typing import Tuple, Union, List, Dict, Any
from PIL import Image
import asyncio
import cv2
import numpy as np
from paddleocr import LayoutDetection
from transformers import LayoutLMv3ForTokenClassification
from dots_ocr.model.reader_helper import boxes2inputs, prepare_inputs, parse_logits
from openvino.runtime import Core, AsyncInferQueue

_layout_detection_model_service = None

class LayoutDetectionBaseService():
    def __init__(self):
        pass
    async def _get_layout_image(
        self,
        image_input: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray, List[np.ndarray]]
    ) -> List[Dict[str, Any]]:
        pass
    async def _get_layout_pdf(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        pass
    
    @staticmethod
    def align_category(label: str) -> str:
        """
        dots_ocr supported categories:
        ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']

        From paddle's documents, PP-DocLayout_plus-L supports 23 categories:
        document title, paragraph title, text, page number, abstract, table of contents, references, footnotes, header, footer, algorithm, 
        formula, formula number, image, figure caption, table, table caption, seal, figure title, figure, header image, footer image, and sidebar text

        From paddle's documents, PP-DocLayoutV2 supports 25 categories:
        document title, section header, text, vertical text, page number, abstract, table of contents, references, footnote, image caption, 
        header, footer, header image, footer image, algorithm, inline formula, display formula, formula number, image, table, figure title
            (figure title, table title, chart title), seal, chart, aside text, and reference content.
        """

        # now contain all categories from PP-DocLayoutV2
        mapping = {
            'doc_title': 'Title',
            'paragraph_title': 'Section-header',
            'text': 'Text',
            'number': 'Text',
            'page_number': 'Page-footer',
            'header': 'Page-header',
            'footer': 'Page-footer',

            'formula': 'Formula',
            'display_formula': 'Formula',
            'inline_formula': 'Inline-Formula', # will embed into Text box later
            'formula_number': 'Text',

            'image': 'Picture',
            'header_image': 'Picture',
            'footer_image': 'Picture',
            'seal': 'Picture',

            'table': 'Table',

            'figure': 'Picture', 
            'figure_title': 'Caption',
            'chart': 'Picture',

            'abstract': 'Text',
            'algorithm': 'Text',
            'aside_text': 'Text',
            'footnote': 'Footnote',
            'vision_footnote': 'Footnote',

            'reference': 'Text',
            'reference_content': 'Text',

            'content': 'Text',
            'vertical_text': 'Text',
        }
        return mapping.get(label, label)
    
    @staticmethod
    def remove_contained_boxes(boxes: List[Dict[str, Any]], thresh: float = 0.9):
        if not boxes:
            return
            
        def area(b): return (b[2] - b[0]) * (b[3] - b[1])
        def inter(b1, b2):
            return max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * \
                max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))

        boxes.sort(key=lambda x: area(x['bbox']), reverse=True)
        
        keep = [True] * len(boxes)
        
        inline_formula_boxes = []
        for i in range(len(boxes)):
            if not keep[i]:
                continue
            bi = boxes[i]['bbox']
            for j in range(i + 1, len(boxes)):
                if not keep[j]:
                    continue
                bj = boxes[j]['bbox']
                if inter(bi, bj) / area(bj) > thresh:
                    keep[j] = False
                    if boxes[j]['category'] == "Inline-Formula":
                        inline_formula_boxes.append(boxes[j])
        
        boxes[:] = [b for b, k in zip(boxes, keep) if k]
        return inline_formula_boxes
    
class LayoutDetectionService(LayoutDetectionBaseService):
    def __init__(
        self,
        model_name="PP-DocLayoutV2",
        batch_size=1,
        cpu_executor=None
    ):
        self._model_name = model_name
        self._model_service = LayoutDetection(model_name=model_name)
        self._batch_size = batch_size
        self.cpu_executor = cpu_executor
    
    def _transform_result(
        self,
        result: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform result to keep only label and bbox with float values.
        Both single and batch results return a list.
        """
        
        def transform_single(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            transformed_boxes = [
                {
                    'category': self.align_category(bbox['label']),
                    'bbox': [float(coord) for coord in bbox['coordinate']]
                }
                for bbox in item.get('boxes', [])
            ]
            if not item.get('boxes'):
                logger.warning(f"No boxes detected on page {item.get('page_index')}. Return whole image as a single box.")
                transformed_boxes = [{
                    'category': 'Picture',
                    'bbox': [0, 0, item.img['res'].size[0], item.img['res'].size[1]]
                }]
                
            inline_formula_boxes = self.remove_contained_boxes(transformed_boxes)
            img = (item.img)['res'] # PP-DocLayout_plus-L will resize the image if parse pdf. It seems is dpi=200 but I don't find relative doc.
            width, height = img.size
            return {
                'page_no': item['page_index'],
                'width': width,
                'height': height,
                'full_layout_info': transformed_boxes
            }, {
                'page_no': item['page_index'],
                'width': width,
                'height': height,
                'full_layout_info': inline_formula_boxes
            } if inline_formula_boxes else None
        
        if isinstance(result, list):
            transformed_results = []
            inline_formula_boxes_all = []
            for item in result:
                transformed_item, inline_formula_boxes = transform_single(item)
                transformed_results.append(transformed_item)
                inline_formula_boxes_all.append(inline_formula_boxes)
        else:
            transformed_item, inline_formula_boxes = transform_single(item)
            transformed_results.append(transformed_item)
            inline_formula_boxes_all.append(inline_formula_boxes)
        return transformed_results, inline_formula_boxes_all
            
    async def _get_layout_image(
        self,
        image_input: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray, List[np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Get layout detection results.
        
        Args:
            image_path: str or list of str - single image path or list of image paths
        
        Returns:
            List[Dict]: Layout detection result(s).

            - Single image: {'input_path': str, 'page_index': None, 'boxes': List[Dict]}
            Each box in 'boxes' contains:
                - 'label': str - label name (e.g., 'paragraph_title')
                - 'bbox': List[float] - [x1, y1, x2, y2] bounding box coordinates

            - Multiple images: List of the above structure
        """
        
        if not isinstance(image_input, list):
            image_input = [image_input]
        
        # Run inference in a separate thread to avoid blocking the event loop
        def run_full_inference_sync(inputs):
            def _to_numpy(img):
                if isinstance(img, Image.Image):
                    return np.array(img)
                return img
            image_input_trans = [_to_numpy(img) for img in image_input]

            result = self._model_service.predict(
                image_input_trans,
                batch_size=self._batch_size,
                layout_nms=True
            )
            return self._transform_result(result)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.cpu_executor, run_full_inference_sync, image_input)

    async def _get_layout_pdf(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Get layout detection results for a PDF file.
        Args:
            file_path: str - path to the PDF file
        Returns:
            List of Dict: Layout detection results for each page in the PDF. format same as get_layout_image.
        """

        # Run inference in a separate thread to avoid blocking the event loop
        def run_full_inference_sync(file_path):
            result = self._model_service.predict(
                file_path,
                batch_size=self._batch_size,
                layout_nms=True
            )
            return self._transform_result(result)
    
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.cpu_executor, run_full_inference_sync, file_path)
    

class LayoutDetectionServiceONNX(LayoutDetectionBaseService):
    CATEGORY_LIST = [
        "abstract", "algorithm", "aside_text", "chart", "content",
        "display_formula", "doc_title", "figure_title", "footer",
        "footer_image", "footnote", "formula_number", "header",
        "header_image", "image", "inline_formula", "number",
        "paragraph_title", "reference", "reference_content", "seal",
        "table", "text", "vertical_text", "vision_footnote"
    ]
    def __init__(
        self,
        model_path="/app/models/pp_doclayoutv2.onnx",
        concurrency: int = 4,
        threshold: float = 0.5,
        use_cpu: bool = True,
        cpu_executor = None 
    ):
        self._model_path = model_path
        ie = Core()
        model = ie.read_model(model_path)
        model.reshape({
            "image": [1, 3, 800, 800],
            "im_shape": [1, 2],
            "scale_factor": [1, 2]
        })

        self._model = ie.compile_model(model, "CPU" if use_cpu else "GPU")
        self._concurrency = concurrency
        self._threshold = threshold
        self.cpu_executor = cpu_executor

        def infer_callback(request, userdata):
            dets = request.get_output_tensor(0).data.copy()
            loop = userdata["loop"]
            future = userdata["future"]
            loop.call_soon_threadsafe(future.set_result, dets)
        self._infer_queue = AsyncInferQueue(self._model, jobs=concurrency)
        self._infer_queue.set_callback(infer_callback)  

    def _preprocess(self, img, orig_h, orig_w):
        img_resized = cv2.resize(img, (800, 800))
        img_input = img_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img_input = (img_input - mean) / std
        img_input = img_input.transpose(2, 0, 1)[None, ...]

        im_shape = np.array([[orig_h, orig_w]], dtype=np.float32)
        scale_factor = np.array([[800 / orig_h, 800 / orig_w]], dtype=np.float32)

        return img_input, im_shape, scale_factor
    
    def _postprocess(self, dets, scale_factor, orig_w, orig_h):

        valid = dets[dets[:, 1] >= self._threshold]
        # restore to original image
        scale_y = scale_factor[0, 0]
        scale_x = scale_factor[0, 1]
        boxes = []
        for det in valid:
            cls_id = int(det[0])
            score = det[1]
            x1, y1, x2, y2 = det[2:6]

            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            box= {
                'category': self.align_category(self.CATEGORY_LIST[cls_id]),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            }
            boxes.append(box)
        if not boxes:
            logger.warning(f"No boxes detected on image. Return whole image as a single box.")
            boxes = [{
                'category': 'Picture',
                'bbox': [0, 0, orig_w, orig_h]
            }]
        inline_formula_boxes = self.remove_contained_boxes(boxes)

        result = {
            'page_no': None,
            'width': orig_w,
            'height': orig_h,
            'full_layout_info': boxes
        }
        if inline_formula_boxes:
            inline_formula_boxes = {
                'page_no': None,
                'width': orig_w,
                'height': orig_h,
                'full_layout_info': inline_formula_boxes
            }
        else:
            inline_formula_boxes = None
        return result, inline_formula_boxes 
    
    async def _get_layout_image(
        self,
        image_input: Union[str, List[str], Image.Image, List[Image.Image], 
        np.ndarray, List[np.ndarray]]
    ) -> List[Dict[str, Any]]:
        
        if not isinstance(image_input, list):
            image_input = [image_input]

        loop = asyncio.get_running_loop()
        def _convert_batch(imgs):
            def _to_numpy(img):
                if isinstance(img, Image.Image):
                    return np.array(img)
                return img
            return [_to_numpy(img) for img in imgs]

        image_input_trans = await loop.run_in_executor(self.cpu_executor, _convert_batch, image_input)

        preprocess_tasks = []
        for img in image_input_trans:
            orig_h, orig_w = img.shape[:2]
            preprocess_tasks.append(
                loop.run_in_executor(self.cpu_executor, self._preprocess, img, orig_h, orig_w)
            )
        preprocessed_data = await asyncio.gather(*preprocess_tasks)
        futures = []
        args = []
        results = []
        inline_formula_boxes_all = []
        for img_input, im_shape, scale_factor in preprocessed_data:
            future = asyncio.Future()
            userdata = {"future": future, "loop": loop}
            
            inputs_data = {
                "image": img_input,
                "im_shape": im_shape,
                "scale_factor": scale_factor
            }
            self._infer_queue.start_async(inputs_data, userdata)
            futures.append(future)
            args.append((im_shape, scale_factor))
        
        dets_list = await asyncio.gather(*futures)

        postprocess_tasks = []
        for (im_shape, scale_factor), dets in zip(args, dets_list):
            orig_h, orig_w = int(im_shape[0,0]), int(im_shape[0,1])
            postprocess_tasks.append(
                loop.run_in_executor(
                    self.cpu_executor, 
                    self._postprocess, 
                    dets, scale_factor, orig_w, orig_h
                )
            )
        postprocess_results = await asyncio.gather(*postprocess_tasks)

        for res, inline in postprocess_results:
            results.append(res)
            inline_formula_boxes_all.append(inline)
                
        return results, inline_formula_boxes_all

async def get_layout_detection_service(
    cpu_executor = None, 
    use_onnx: bool = True,
    onnx_concurrency = 4,
    onnx_threshold = 0.5,
    onnx_use_cpu = True,
) -> LayoutDetectionService:
    global _layout_detection_model_service
    if _layout_detection_model_service is None:
        logger.info("Loading layout detection ONNX model...")
        if use_onnx:
            _layout_detection_model_service = await asyncio.to_thread(
                LayoutDetectionServiceONNX,
                concurrency=onnx_concurrency,
                threshold=onnx_threshold,
                use_cpu=onnx_use_cpu,
                cpu_executor=cpu_executor)
        else:
            logger.info("Loading layout detection model...")
            _layout_detection_model_service = await asyncio.to_thread(LayoutDetectionService, cpu_executor=cpu_executor)
    return _layout_detection_model_service

async def get_layout_image(image_input) -> List[Dict[str, Any]]:
    if _layout_detection_model_service is None:
        logger.error("Layout detection model is not loaded.")
    return await _layout_detection_model_service._get_layout_image(image_input)

async def get_layout_pdf(file_path: str) -> Dict[str, Any]:
    if _layout_detection_model_service is None:
        logger.error("Layout detection model is not loaded.")
    return await _layout_detection_model_service._get_layout_pdf(file_path)




_layout_reader_model_service = None

class LayoutReaderService():
    def __init__(
        self,
        model_name="/app/models/Relation",
    ):
        self._model_name = model_name
        self._model_service = LayoutLMv3ForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_name)

    async def _sort_bboxes(
        self,
        bboxes: List[List[float]],
        width: int,
        height: int,
    ):
        """
        Sort bounding boxes in reading order (top to bottom, left to right).
        
        Args:
            bboxes: List of bounding boxes, each defined by [x1, y1, x2, y2].
        
        Returns:
            List of indices representing the sorted order of the bounding boxes.
        """
        # layoutreader model need boxes normalized to [0, 1000]
        scale_x = 1000 / width
        scale_y = 1000 / height
        # convert coordinate system
        norm_boxes = [
            [
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y),
            ]
            for box in bboxes
        ]

        try:
            def _run_model():
                inputs = boxes2inputs(norm_boxes)
                inputs = prepare_inputs(inputs, self._model_service)
                logits = self._model_service(**inputs).logits.cpu().squeeze(0)
                return parse_logits(logits, len(norm_boxes))
            
            orders = await asyncio.to_thread(_run_model)
        except Exception as e:
            logger.error(f"Error in sorting bboxes: {e}")
            orders = list(range(len(bboxes)))
        return orders

async def get_layout_reader_service() -> LayoutReaderService:
    global _layout_reader_model_service
    if (_layout_reader_model_service is None):
        logger.info("Loading layout reader model...")
        _layout_reader_model_service = await asyncio.to_thread(LayoutReaderService)
    return _layout_reader_model_service

async def sort_bboxes(bboxes: List[List[float]], width, height) -> List[int]:
    model_service = await get_layout_reader_service()
    return await model_service._sort_bboxes(bboxes, width, height)
