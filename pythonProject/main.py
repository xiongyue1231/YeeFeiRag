import time
import uuid
import datetime
import os.path
import traceback

from fastapi import (
    FastAPI, UploadFile, File, Form, BackgroundTasks, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing_extensions import Annotated

from route_schemas import (
    DocumentResponse, KnowledgeRequest, KnowledgeResponse,
    RAGRequest, RAGResponse,
)
from src.database.db_api import (
    KnowledgeDatabase, KnowledgeDocument, Session,
    get_all_knowledge_bases,
)
from src.analysis.processor import DocumentProcessor
from src.rag.ragsimple import create_conversational_rag
from src.app_config.loder import ConfigLoader
from src.app_config.logger import logger
from src.exceptions import BusinessException, NotFoundException

# ==================== 初始化 ====================
config_manager = ConfigLoader()

app = FastAPI(
    title="YeeFei RAG 知识库管理系统",
    description="知识库、文档管理与 RAG 对话接口",
    version="1.0.0",
)

# CORS：开发环境放开；生产环境改成具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 全局异常处理器 ====================

@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    logger.warning(f"[业务异常] {request.method} {request.url.path} | {exc.message}")
    return JSONResponse(
        status_code=exc.code,
        content={
            "request_id": str(uuid.uuid4()),
            "response_code": exc.code,
            "response_msg": exc.message,
            "data": None,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(
        f"[HTTP异常] {request.method} {request.url.path} | {exc.status_code} {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": str(uuid.uuid4()),
            "response_code": exc.status_code,
            "response_msg": str(exc.detail),
            "data": None,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"[参数校验失败] {request.method} {request.url.path} | {exc.errors()}"
    )
    return JSONResponse(
        status_code=422,
        content={
            "request_id": str(uuid.uuid4()),
            "response_code": 422,
            "response_msg": "请求参数校验失败",
            "data": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"[未捕获异常] {request.method} {request.url.path}\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "request_id": str(uuid.uuid4()),
            "response_code": 500,
            "response_msg": "服务器内部错误",
            "data": None,
        },
    )


# ==================== 工具函数 ====================
def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _elapsed(start: float) -> float:
    return round(time.time() - start, 4)


def _new_request_id() -> str:
    return str(uuid.uuid4())


# ==================== 知识库 ====================

@app.post("/v1/knowledge_base", response_model=KnowledgeResponse)
def create_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    """创建知识库"""
    start = time.time()

    with Session() as session:
        record = KnowledgeDatabase(
            title=req.title,
            create_dt=_now(),
            update_dt=_now(),
        )
        session.add(record)
        session.flush()  # 拿到自增主键
        knowledge_id = record.knowledge_id
        # category 存储 Milvus 集合名，由 knowledge_id 生成，
        # 与检索侧（rag_api.py / ragbase.py 的 kb_{knowledge_id}）保持一致
        category = f"kb_{knowledge_id}"
        record.category = category
        session.commit()

    logger.info(f"知识库创建成功: id={knowledge_id}, title={req.title}")

    return KnowledgeResponse(
        request_id=_new_request_id(),
        knowledge_id=knowledge_id,
        category=category,
        title=req.title,
        response_code=200,
        response_msg="知识库创建成功",
        process_status="completed",
        processing_time=_elapsed(start),
    )


@app.get("/v1/knowledge_base")
def list_knowledge_bases():
    """查询所有知识库"""
    start = time.time()

    # 关键修复：在 session 内就把属性全部取出为纯字典，
    # 避免返回 detached ORM 实例后再访问属性触发 DetachedInstanceError
    with Session() as session:
        records = (
            session.query(KnowledgeDatabase)
            .order_by(KnowledgeDatabase.create_dt.desc())
            .all()
        )
        data = [
            {
                "knowledge_id": r.knowledge_id,
                "title": r.title,
                "category": r.category,
                "create_dt": r.create_dt.strftime("%Y-%m-%d %H:%M:%S") if r.create_dt else None,
                "update_dt": r.update_dt.strftime("%Y-%m-%d %H:%M:%S") if r.update_dt else None,
            }
            for r in records
        ]

    return {
        "request_id": _new_request_id(),
        "response_code": 200,
        "response_msg": "查询成功",
        "data": data,
        "processing_time": _elapsed(start),
    }


@app.get("/v1/knowledge_base/{knowledge_id}")
def get_knowledge_base(knowledge_id: int):
    """查询单个知识库"""
    # 关键修复：在 session 关闭前把全部属性取出为字典，
    # 避免返回函数后再访问 record.* 触发 DetachedInstanceError
    with Session() as session:
        record = (
            session.query(KnowledgeDatabase)
            .filter(KnowledgeDatabase.knowledge_id == knowledge_id)
            .first()
        )

        if record is None:
            raise NotFoundException(f"知识库不存在: id={knowledge_id}")

        data = {
            "knowledge_id": record.knowledge_id,
            "title": record.title,
            "category": record.category,
            "create_dt": record.create_dt.strftime("%Y-%m-%d %H:%M:%S") if record.create_dt else None,
            "update_dt": record.update_dt.strftime("%Y-%m-%d %H:%M:%S") if record.update_dt else None,
        }

    return {
        "request_id": _new_request_id(),
        "response_code": 200,
        "response_msg": "查询成功",
        "data": data,
    }


@app.delete("/v1/knowledge_base/{knowledge_id}", response_model=KnowledgeResponse)
def delete_knowledge_base(knowledge_id: int) -> KnowledgeResponse:
    """删除知识库"""
    start = time.time()

    with Session() as session:
        record = (
            session.query(KnowledgeDatabase)
            .filter(KnowledgeDatabase.knowledge_id == knowledge_id)
            .first()
        )
        if record is None:
            raise NotFoundException(f"知识库不存在: id={knowledge_id}")

        title = record.title
        category = record.category
        session.delete(record)
        session.commit()

    logger.info(f"知识库删除成功: id={knowledge_id}")

    return KnowledgeResponse(
        request_id=_new_request_id(),
        knowledge_id=knowledge_id,
        category=category,
        title=title,
        response_code=200,
        response_msg="知识库删除成功",
        process_status="completed",
        processing_time=_elapsed(start),
    )


# ==================== 文档 ====================

@app.post("/v1/document", response_model=DocumentResponse)
async def upload_document(
        knowledge_id: Annotated[int, Form()],
        file: Annotated[UploadFile, File(...)],
        background_tasks: BackgroundTasks,
) -> DocumentResponse:
    """上传文档（后台异步解析），文档标题与分类由文件名及后缀推导"""
    start = time.time()

    # 文档标题 = 文件名（去后缀），文档分类 = 文件后缀
    file_name = os.path.basename(file.filename or "untitled")
    title, file_ext = os.path.splitext(file_name)
    title = file_name
    file_type = file_ext.lstrip(".").lower() or "unknown"
    with Session() as session:
        knowledgeDatabaseRes = (
            session.query(KnowledgeDatabase)
            .filter(KnowledgeDatabase.knowledge_id == knowledge_id)
            .first()
        )
        if knowledgeDatabaseRes is None:
            raise NotFoundException(f"知识库不存在: id={knowledge_id}")

        # 会话关闭前取出所需属性，避免 DetachedInstanceError
        collection_name = knowledgeDatabaseRes.category
        knowledge_title = knowledgeDatabaseRes.title

        # 落库
        record = KnowledgeDocument(
            title=title,
            knowledge_name=knowledge_title,
            knowledge_id=knowledge_id,
            file_path="",
            file_type=file_type,
            create_dt=_now(),
            update_dt=_now(),
        )
        session.add(record)
        session.flush()
        document_id = record.document_id
        session.commit()

        # 存文件（使用绝对路径 + 分块写入，避免大文件一次性读入内存）
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_files")
        os.makedirs(upload_dir, exist_ok=True)
        # 防止文件名里有路径分隔符或非法字符
        safe_name = os.path.basename(file.filename or f"document_id_{document_id}")
        file_path = os.path.join(upload_dir, f"document_id_{document_id}_{safe_name}")

        # 分块写入，避免大文件一次性读入内存引发 OOM
        chunk_size = 1024 * 1024  # 1 MB
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        record.file_path = file_path
        session.commit()

    # 后台解析入库
    background_tasks.add_task(
        DocumentProcessor().process_and_store,
        knowledge_id=knowledge_id,
        document_id=document_id,
        file_type=file_type,
        file_path=file_path,
        collection_name=collection_name,
    )

    logger.info(f"文档上传成功: id={document_id}, title={title}, kb={knowledge_id}")

    return DocumentResponse(
        request_id=_new_request_id(),
        document_id=document_id,
        title=title,
        knowledge_id=knowledge_id,
        response_code=200,
        response_msg="文档添加成功",
        process_status="completed",
        processing_time=_elapsed(start),
    )

@app.get("/v1/document")
def get_knowledge_document():
    """查询上传文档"""
    # 关键修复：.all() 返回的是列表，必须遍历取属性，
    # 原来直接 record.knowledge_id 会报 'list' object has no attribute 'knowledge_id'
    with Session() as session:
        records = (
            session.query(KnowledgeDocument)
            .all()
        )

        # 在 session 关闭前把每条记录转为纯字典，避免 DetachedInstanceError
        data = [
            {
                "document_id": r.document_id,
                "knowledge_id": r.knowledge_id,
                "title": r.title,
                "file_path": r.file_path,
                "file_type": r.file_type,
                "knowledge_name": r.knowledge_name,
                "create_dt": r.create_dt.strftime("%Y-%m-%d %H:%M:%S") if r.create_dt else None,
                "update_dt": r.update_dt.strftime("%Y-%m-%d %H:%M:%S") if r.update_dt else None,
            }
            for r in records
        ]

    return {
        "request_id": _new_request_id(),
        "response_code": 200,
        "response_msg": "查询成功",
        "data": data,
    }



# ==================== 对话 ====================

@app.post("/chat", response_model=RAGResponse)
def chat(req: RAGRequest) -> RAGResponse:
    """RAG 对话"""
    start = time.time()
    session_id = req.session_id or "default_session"
    logger.info(f"对话开始: kb={req.knowledge_id}, session={session_id}")
    conversational_rag = create_conversational_rag(req.knowledge_id)
    message = conversational_rag.invoke(
        {"input": req.message[-1]["content"]},
        config={"configurable": {"session_id": session_id}},
    )

    logger.info(f"对话完成: kb={req.knowledge_id}, session={session_id}")

    return RAGResponse(
        request_id=_new_request_id(),
        message=message,
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=_elapsed(start),
    )


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn

    logger.info(f"服务启动，端口 {config_manager.config.rag.port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config_manager.config.rag.port,
        workers=1,
    )
