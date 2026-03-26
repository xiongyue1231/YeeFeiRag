# pip install paddlepaddle==3.0.0
# # 或安装 2.6.0 版本（更稳定）
# pip install paddlepaddle==2.6.2
# # 安装对应版本的 paddleocr
# pip install paddleocr==2.10.0
import torch
from paddleocr import PaddleOCR
from chuck import Chuck


class SimpleOcr:
    def __init__(self, lang='ch'):
        self.lang = lang
        self.ocr = PaddleOCR(lang=self.lang,
                             use_textline_orientation=True,
                             # show_log=False,
                             )

    def recognize_img(self, image_path):
        """
        识别图片中的文字
        :param image_path: 图片路径
        :return: 识别结果列表 [(文字, 置信度), ...]
        """
        result = self.ocr.ocr(image_path)
        texts = []
        if result[0]:
            for line in result[0]:
                text = line[1][0]  # 文字内容
                confidence = line[1][1]  # 置信度
                texts.append((text, confidence))

        return texts


