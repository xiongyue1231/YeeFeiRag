from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import time
import datetime
import uuid
import traceback
import uvicorn
from typing_extensions import Annotated
from route_schemas import (
    DocumentResponse, KnowledgeRequest, KnowledgeResponse,RAGRequest,RAGResponse
)
from db_api import (
    KnowledgeDatabase, KnowledgeDocument, Session,
)
from pythonProject.file_handler import FileHandler
from processor import DocumentProcessor
from rag_api import Rag
app = FastAPI()
from src.app_config.loder import ConfigLoader

config_manager = ConfigLoader()


# 新增知识库
@app.post("/v1/knowledge_base")
def add_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    start_time = time.time()
    try:
        for retry_time in range(10):
            with Session() as session:
                record = KnowledgeDatabase(
                    title=req.title,
                    category=req.category,
                    create_dt=datetime.datetime.now(),
                    update_dt=datetime.datetime.now(),
                )
                session.add(record)
                session.flush()  # Flushes changes to generate primary key if using autoincrement
                knowledge_id = record.knowledge_id
                session.commit()

            return KnowledgeResponse(  # type: ignore
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                category=req.category,
                title=req.title,
                response_code=200,
                response_msg="知识库插入成功",
                process_status="completed",
                processing_time=time.time() - start_time
            )

    except Exception as e:
        print(traceback.format_exc())
        # TODO 打印日志
        pass

    return KnowledgeResponse(  # type: ignore
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        category="",
        title="",
        response_code=504,
        response_msg="知识库插入失败",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 删除知识库
@app.delete("/v1/knowledge_base")
def delete_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start_time = time.time()

    try:
        for retry_time in range(10):
            with Session() as session:
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
                if record is None:
                    break

                session.delete(record)
                session.commit()
                return KnowledgeResponse(  # type: ignore
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    category=str(record.category),
                    title=str(record.title),
                    response_code=200,
                    response_msg="知识库删除成功",
                    process_status="completed",
                    processing_time=time.time() - start_time
                )
    except Exception as e:
        # TODO 打印日志
        pass

    return KnowledgeResponse(  # type: ignore
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category="",
        title="",
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 新增文档
@app.post("/v1/document")
async def add_document(
        knowledge_id: int = Annotated[str, Form()],
        title: str = Annotated[str, Form()],
        category: str = Annotated[str, Form()],
        file: UploadFile = Annotated[str, File(...)],
        background_tasks: BackgroundTasks = Annotated[BackgroundTasks, Form()]
) -> DocumentResponse:
    start_time = time.time()
    response_msg = "新增文档失败"
    try:
        for retry_time in range(10):
            # 上传的文档，记录在关系型数据库中， orm 添加记录
            # 创建数据库连接
            with Session() as session:
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
                if record is None:
                    response_msg = "知识库不存在，请提前创建"
                    break

                record = KnowledgeDocument(
                    title=title,
                    category=category,
                    knowledge_id=knowledge_id,
                    file_path="",
                    file_type=file.content_type,
                    create_dt=datetime.datetime.now(),
                    update_dt=datetime.datetime.now(),
                )
                session.add(record)
                session.flush()  # Flushes changes to generate primary key if using autoincrement
                document_id = record.document_id
                session.commit()

                # 存储数据到文件    或者存储到oss中，比如阿里云 七牛，目前是保存到本地
                file_path = f"upload_files/document_id_{document_id}_" + file.filename
                with open(file_path, "wb") as buffer:
                    buffer.write(file.file.read())

                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()
                record.file_path = file_path
                session.commit()

            # 文档内容解析，后台执行，后台提取数据
            background_tasks.add_task(
                DocumentProcessor().process_and_store(file_path),  # 后台运行的函数名
                knowledge_id=knowledge_id,
                document_id=document_id,
                title=title,
                file_type=file.content_type,
                file_path=file_path
            )

            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                category=category,
                title=title,
                knowledge_id=knowledge_id,
                file_type=file.content_type,
                response_code=200,
                response_msg="文档添加成功",
                process_status="completed",
                processing_time=time.time() - start_time
            )
    except Exception as e:
        print(traceback.format_exc())
        pass

    return DocumentResponse(  # type: ignore
        request_id=str(uuid.uuid4()),
        document_id=0,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg=response_msg,
        process_status="completed",
        processing_time=time.time() - start_time
    )


@app.post("/chat")
def chat(req: RAGRequest) -> RAGResponse:
    start_time = time.time()
    message = Rag().chat_with_rag(req.knowledge_id, req.message)

    return RAGResponse(
        request_id=str(uuid.uuid4()),
        message=message,
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config_manager.config.rag.port, workers=1)
