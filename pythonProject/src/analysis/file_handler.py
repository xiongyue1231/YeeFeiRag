
from pathlib import Path
from typing import BinaryIO, Union
import io
from content_type import ContentType
from typing import List, Tuple
import zipfile
from xml.etree import ElementTree as ET
from pdf2image import convert_from_path
import os


class FileHandler:
    # def __init__(self, file_path):
    #     self.filetype = ContentType().detect_type(file_path)

    def extract_content(self, file_path):
        filetype = ContentType().detect_type(file_path)
        if "pdf" in filetype:
            return self._extract_pdf(file_path)
        elif "document" in filetype:
            return self._extract_document(file_path)
        elif "image" in filetype:
            return self._extract_image(file_path)
        elif "text" in filetype:
            return self._extract_text(file_path)

    def _extract_text(self, file_path) -> str:
        """提取纯文本"""
        with open(file_path, "rb") as f:
            text = f.read().decode('utf-8')
        lines = text.splitlines()
        return [(line, 1.0) for line in lines if line.strip()]

    def _extract_image(self, file_path) -> str:
        """OCR提取图片文字"""
        from paddleOcr import SimpleOcr
        simpleocr = SimpleOcr()
        return simpleocr.recognize_img(file_path)

    def _extract_document(self, file_path):

        with zipfile.ZipFile(file_path) as zf:
            # 读取 document.xml
            xml_content = zf.read('word/document.xml')

        root = ET.fromstring(xml_content)

        # 提取所有 <w:t> 标签的文本
        texts = []
        for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
            if t.text:
                texts.append(t.text.strip() + " ")

        return [(line, 1.0) for line in texts]

    def _extract_pdf(self, file_path):
        # 从pdf中提取数据，程序不能直接处理pdf，需要先转换成图片
        # first_page=1, last_page=1 只读取一页

        convert_images = convert_from_path(file_path, dpi=300)  # dpi 控制清晰度
        image_names = []
        for i, image in enumerate(convert_images):
            output_path = os.path.join('./images', file_path)
            # 保存图片到本地
            image.save(f"{output_path}_{i}", "PNG")
            # 添加图片到集合，然后进行ocr
            image_names.append(f"{output_path}_{i}")

        from paddleOcr import SimpleOcr
        simpleocr = SimpleOcr()
        texts = []
        for image_name in image_names:
            texts.append(simpleocr.recognize_img(image_name))

        return texts
