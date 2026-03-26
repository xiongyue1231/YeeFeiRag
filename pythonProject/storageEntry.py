from paddleOcr import SimpleOcr
from pymilvus import MilvusClient
from chuck import Chuck
import google.protobuf
from paddleocr import PaddleOCR

# print(f"protobuf 版本: {google.protobuf.__version__}")
#
# # gRPC 方式（推荐，性能更好）
# client = MilvusClient(uri="tcp://localhost:19530")
# print(client.get_server_version())

# path = 'imgtest/test.png'
# simpleOcr = SimpleOcr()
# res = simpleOcr.recognize_img(path)
# print(res)
# chuck = Chuck()
# cleaned_data = chuck.clean_sentences(res, path)
# print(cleaned_data)
# for i in enumerate(res):
#     chuck=Chuck()
#     # 小于20 直接入库
#     if len(res[i])< 20:
#         cleaned_data=chuck.clean_text()
