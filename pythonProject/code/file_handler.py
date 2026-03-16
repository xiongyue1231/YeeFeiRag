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

    def detect_type(cls, filename: str) -> str:
        """检测文件类型"""
        ext = Path(filename).suffix.lower()
        return cls.SUPPORTED_TYPES.get(ext, 'unknown')

    def extract_content(self, knowledge_id, document_id, title, file_type, file_path):
        if "pdf" in file_type:
            pass
        elif "word" in file_type:
            pass
        elif "image" in file_type:
            self._extract_image(file_path)

        print("提取完成", document_id, file_type, file_path)

    def _extract_text(file: BinaryIO) -> str:
        """提取纯文本"""
        return file.read().decode('utf-8')

    def _extract_document(file: BinaryIO, filename: str) -> str:
        """提取文档内容（PDF/Word）"""
        # 可以调用现有的 chuck.py 或引入 PyPDF2/python-docx
        pass

    def _extract_image(path) -> str:
        """OCR提取图片文字"""
        from paddleOcr import SimpleOcr
        simpleocr = SimpleOcr()
        return simpleocr.recognize_img(path)
