# code/file_handler.py
from pathlib import Path
from typing import BinaryIO, Union
import io


class FileHandler:
    """文件类型处理器"""

    SUPPORTED_TYPES = {
        # 文本类
        '.txt': 'text',
        '.md': 'text',
        '.json': 'text',
        '.csv': 'text',
        # 文档类
        '.pdf': 'document',
        '.docx': 'document',
        '.doc': 'document',
        # 图片类（需要OCR）
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.bmp': 'image',
    }

    def detect_type(self, filename: str) -> str:
        """检测文件类型"""
        ext = Path(filename).suffix.lower()
        return self.SUPPORTED_TYPES.get(ext, 'unknown')

    def extract_content(self, file_path):
        filetype = self.detect_type(file_path)
        if "pdf" in filetype:
            pass
        elif "word" in filetype:
            pass
        elif "image" in filetype:
            content = self._extract_image(file_path)

        print("提取完成", file_path)
        return content

    def _extract_text(file: BinaryIO) -> str:
        """提取纯文本"""
        return file.read().decode('utf-8')

    def _extract_document(file: BinaryIO, filename: str) -> str:
        """提取文档内容（PDF/Word）"""
        # 可以调用现有的 chuck.py 或引入 PyPDF2/python-docx
        pass

    def _extract_image(self, file_path) -> str:
        """OCR提取图片文字"""
        from paddleOcr import SimpleOcr
        simpleocr = SimpleOcr()
        return simpleocr.recognize_img(file_path)
