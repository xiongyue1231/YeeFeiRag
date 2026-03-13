
from paddleOcr import SimpleOcr
from pymilvus import MilvusClient
from chuck import Chuck

# gRPC 方式（推荐，性能更好）
client = MilvusClient(uri="grpc://localhost:19530")
# print(client.get_server_version())

path='../imgtest/test.png'
simpleOcr = SimpleOcr()
res=simpleOcr.recognize_img(path)
print(res)


chuck=Chuck()
cleaned_data=chuck.clean_text(res)