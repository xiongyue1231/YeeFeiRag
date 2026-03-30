# code/file_handler.py
from pathlib import Path
from typing import BinaryIO, Union
import io
from content_type import ContentType
from typing import List, Tuple

class FileHandler:
    def extract_content(self, file_path):
        filetype = ContentType().detect_type(file_path)
        if "pdf" in filetype:
            pass
        elif "word" in filetype:
            pass
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

    def _extract_document(file: BinaryIO, filename: str) -> str:
        """提取文档内容（PDF/Word）"""

        pass

    def _extract_image(self, file_path) -> str:
        """OCR提取图片文字"""
        from paddleOcr import SimpleOcr
        simpleocr = SimpleOcr()
        return simpleocr.recognize_img(file_path)
