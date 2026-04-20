from pathlib import Path


class ContentType:
    SUPPORTED_TYPES = {
        # 文本类
        '.txt': 'text',
        '.csv': 'text',
        # 文档类
        '.pdf': 'document',
        '.docx': 'document',
        '.doc': 'document',
        '.excel': 'document',
        # 图片类（需要OCR）
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
    }

    def detect_type(self, filename: str) -> str:
        """检测文件类型"""
        ext = Path(filename).suffix.lower()
        return self.SUPPORTED_TYPES.get(ext, 'unknown')
