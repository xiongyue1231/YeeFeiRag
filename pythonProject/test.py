#
# #模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('PaddlePaddle/PaddleOCR-VL-1.5',cache_dir='./models/PaddlePaddle/PaddleOCR-VL-1.5')

#
# from paddleocr import PaddleOCR
#
# # 首次运行会自动下载模型（约 100MB）
# ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
#
# # 测试识别
# result = ocr.ocr('test.jpg', cls=True)
# for line in result[0]:
#     print(f"文字: {line[1][0]}, 置信度: {line[1][1]:.2f}")

import  torch
print(torch.__version__)