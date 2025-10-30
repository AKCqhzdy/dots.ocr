from loguru import logger
from typing import Union, List, Dict, Any
from paddleocr import LayoutDetection
from transformers import LayoutLMv3ForTokenClassification
from dots_ocr.model.layout_reader.helper import prepare_inputs, boxes2inputs, parse_logits

_layout_detection_model_service = None

class LayoutDetectionService():
    def __init__(
        self,
        model_name="PP-DocLayout_plus-L",
        batch_size=1
    ):
        self._model_name = model_name
        self._model_service = LayoutDetection(model_name=model_name)
        self._batch_size = batch_size

    def _transform_result(
        self,
        result: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Transform result to keep only label and bbox with float values.
        Both single and batch results return a list.
        """
        # TODO(zihao): align the category label format with dotsocr
        def transform_single(item: Dict[str, Any]) -> Dict[str, Any]:
            transformed_boxes = [
                {
                    'category': bbox['label'],
                    'bbox': [float(coord) for coord in bbox['coordinate']]
                }
                for bbox in item.get('boxes', [])
            ]
            img = (item.img)['res'] # PP-DocLayout_plus-L will resize the image but I don't find the method. So we need to recover the original size outside.
            width, height = img.size
            return {
                'page_no': item['page_index'],
                'width': width,
                'height': height,
                'full_layout_info': transformed_boxes
            }
        
        if isinstance(result, list):
            return [transform_single(item) for item in result]
        else:
            return [transform_single(result)]

    def _get_layout_image(
        self,
        image_path: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get layout detection results.
        
        Args:
            image_path: str or list of str - single image path or list of image paths
        
        Returns:
            Dict or List[Dict]: Layout detection result(s).

            - Single image: {'input_path': str, 'page_index': None, 'boxes': List[Dict]}
            Each box in 'boxes' contains:
                - 'label': str - label name (e.g., 'paragraph_title')
                - 'bbox': List[float] - [x1, y1, x2, y2] bounding box coordinates

            - Multiple images: List of the above structure
        """
        result = self._model_service.predict(image_path, batch_size=self._batch_size, layout_nms=True)
        return self._transform_result(result)

    def _get_layout_pdf(
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
        result = self._model_service.predict(file_path, batch_size=self._batch_size, layout_nms=True)
        return self._transform_result(result)

def get_layout_detection_service() -> LayoutDetectionService:
    global _layout_detection_model_service
    if _layout_detection_model_service is None:
        logger.info("Loading layout detection model...")
        _layout_detection_model_service = LayoutDetectionService()
    return _layout_detection_model_service

def get_layout_image(image_path: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    model_service = get_layout_detection_service()
    return model_service._get_layout_image(image_path)

def get_layout_pdf(file_path: str) -> Dict[str, Any]:
    model_service = get_layout_detection_service()
    return model_service._get_layout_pdf(file_path)






_layout_reader_model_service = None

class LayoutReaderService():
    def __init__(
        self,
        model_name="hantian/layoutreader",
    ):
        self._model_name = model_name
        self._model_service = LayoutLMv3ForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_name)

    def _sort_bboxes(
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
        norm_boxes = [
            [
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y),
            ]
            for box in bboxes
        ]

        inputs = boxes2inputs(norm_boxes)
        inputs = prepare_inputs(inputs, self._model_service)
        logits = self._model_service(**inputs).logits.cpu().squeeze(0)
        orders = parse_logits(logits, len(norm_boxes))
        return orders

def get_layout_reader_service() -> LayoutReaderService:
    global _layout_reader_model_service
    if _layout_reader_model_service is None:
        logger.info("Loading layout reader model...")
        _layout_reader_model_service = LayoutReaderService()
    return _layout_reader_model_service

def sort_bboxes(bboxes: List[List[float]], width, height) -> List[int]:
    model_service = get_layout_reader_service()
    return model_service._sort_bboxes(bboxes, width, height)