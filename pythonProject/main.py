from fastapi import FastAPI,UploadFile,File,Form,BackgroundTasks
import time
import datetime
import uuid
import traceback
from typing_extensions import Annotated
from route_schemas import (
    DocumentResponse
)
from db_api import (
    KnowledgeDatabase,KnowledgeDocument,Session
)
from pythonProject.file_handler import FileHandler
app = FastAPI()


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
                record
                session.commit()

                # 存储数据到文件
                file_path = f"upload_files/document_id_{document_id}_" + file.filename
                with open(file_path, "wb") as buffer:
                    buffer.write(file.file.read())

                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()
                record.file_path = file_path
                session.commit()

            # 文档内容解析，后台执行，后台提取数据
            background_tasks.add_task(
                FileHandler().extract_content(),  # 后台运行的函数名
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