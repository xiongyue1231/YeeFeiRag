from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Header
import time
import datetime
import uuid
import traceback
import uvicorn
import jwt
from passlib.context import CryptContext
from typing_extensions import Annotated
from route_schemas import (
    DocumentResponse, KnowledgeRequest, KnowledgeResponse, RAGRequest, RAGResponse,
    LoginRequest, LoginResponse
)
from src.database.db_api import (
    KnowledgeDatabase, KnowledgeDocument, Session, User,
)

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
SECRET_KEY = "your_secret_key"  # 实际应用中应该从配置中获取
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
from src.analysis.file_handler import FileHandler
from src.analysis.processor import DocumentProcessor
from src.rag.rag_api import Rag

app = FastAPI()
from src.app_config.loder import ConfigLoader

# 生成JWT令牌
def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    """生成JWT访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# JWT验证函数
def validate_jwt_token(token: str) -> dict:
    """验证JWT令牌并返回负载"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception as e:
        print(f"JWT验证失败: {str(e)}")
        return None

# 密码验证
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

# 密码加密
def get_password_hash(password: str) -> str:
    """获取密码哈希值"""
    return pwd_context.hash(password)

config_manager = ConfigLoader()


# 新增知识库
@app.post("/v1/knowledge_base")
def add_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    start_time = time.time()
    knowledge_id = 0
    
    try:
        # 验证参数
        if not req.title or not req.category:
            return KnowledgeResponse(
                request_id=str(uuid.uuid4()),
                knowledge_id=0,
                category="",
                title="",
                response_code=400,
                response_msg="标题和分类不能为空",
                process_status="completed",
                processing_time=time.time() - start_time
            )
        
        with Session() as session:
            # 检查是否存在相同名称的知识库
            existing_record = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.title == req.title,
                KnowledgeDatabase.category == req.category
            ).first()
            
            if existing_record:
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=existing_record.knowledge_id,
                    category=existing_record.category,
                    title=existing_record.title,
                    response_code=409,
                    response_msg="知识库已存在",
                    process_status="completed",
                    processing_time=time.time() - start_time
                )
            
            try:
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
                
                return KnowledgeResponse(
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
                session.rollback()
                raise
                
    except Exception as e:
        print(traceback.format_exc())
        response_msg = f"知识库插入失败: {str(e)}"

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category=req.category if req else "",
        title=req.title if req else "",
        response_code=500,
        response_msg=response_msg,
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 删除知识库
@app.delete("/v1/knowledge_base")
def delete_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start_time = time.time()
    response_msg = "知识库不存在"
    category = ""
    title = ""

    try:
        # 验证token有效性（根据实际需求实现）
        # if not validate_token(token):
        #     return KnowledgeResponse(
        #         request_id=str(uuid.uuid4()),
        #         knowledge_id=knowledge_id,
        #         category="",
        #         title="",
        #         response_code=401,
        #         response_msg="无效的token",
        #         process_status="completed",
        #         processing_time=time.time() - start_time
        #     )
        
        with Session() as session:
            record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
            if record is None:
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    category="",
                    title="",
                    response_code=404,
                    response_msg=response_msg,
                    process_status="completed",
                    processing_time=time.time() - start_time
                )

            category = str(record.category)
            title = str(record.title)
            
            try:
                # 删除关联的文档记录
                documents = session.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_id == knowledge_id).all()
                for doc in documents:
                    # 清理文件（如果需要）
                    # if doc.file_path and os.path.exists(doc.file_path):
                    #     os.remove(doc.file_path)
                    session.delete(doc)
                
                # 删除知识库记录
                session.delete(record)
                session.commit()
                
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    category=category,
                    title=title,
                    response_code=200,
                    response_msg="知识库删除成功",
                    process_status="completed",
                    processing_time=time.time() - start_time
                )
            except Exception as e:
                session.rollback()
                raise
                
    except Exception as e:
        print(traceback.format_exc())
        response_msg = f"删除知识库失败: {str(e)}"

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category=category,
        title=title,
        response_code=500,
        response_msg=response_msg,
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
    document_id = 0
    file_path = ""
    collection_name = ""
    
    try:
        # 上传的文档，记录在关系型数据库中， orm 添加记录
        # 创建数据库连接
        with Session() as session:
            knowledge_record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
            if knowledge_record is None:
                response_msg = "知识库不存在，请提前创建"
                return DocumentResponse(
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

            collection_name = knowledge_record.category
            
            try:
                # 关系型数据库 添加记录
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
                
                # 存储数据到文件    或者存储到oss中，比如阿里云 七牛，目前是保存到本地
                file_path = f"upload_files/document_id_{document_id}_" + file.filename
                # 确保上传目录存在
                import os
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, "wb") as buffer:
                    buffer.write(file.file.read())

                # 更新文件路径
                record.file_path = file_path
                session.commit()
                
                # 文档内容解析，后台执行，后台提取数据
                background_tasks.add_task(
                    # 文件处理入库函数
                    DocumentProcessor().process_and_store,
                    knowledge_id=knowledge_id,
                    document_id=document_id,
                    # title=title,
                    file_type=file.content_type,
                    file_path=file_path,
                    collection_name=collection_name,
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
                session.rollback()
                # 清理已创建的文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise
                
    except Exception as e:
        print(traceback.format_exc())
        response_msg = f"新增文档失败: {str(e)}"

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        category=category,
        title=title,
        knowledge_id=knowledge_id,
        file_type=file.content_type,
        response_code=404,
        response_msg=response_msg,
        process_status="completed",
        processing_time=time.time() - start_time
    )


@app.post("/login")
def login(req: LoginRequest) -> LoginResponse:
    start_time = time.time()
    try:
        with Session() as session:
            # 查找用户
            user = session.query(User).filter(User.username == req.username).first()
            if not user:
                return LoginResponse(
                    request_id=str(uuid.uuid4()),
                    user_id="",
                    username="",
                    token="",
                    response_code=401,
                    response_msg="用户名或密码错误",
                    process_status="completed",
                    processing_time=time.time() - start_time
                )
            
            # 验证密码
            if not verify_password(req.password, user.password):
                return LoginResponse(
                    request_id=str(uuid.uuid4()),
                    user_id="",
                    username="",
                    token="",
                    response_code=401,
                    response_msg="用户名或密码错误",
                    process_status="completed",
                    processing_time=time.time() - start_time
                )
            
            # 生成访问令牌
            access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user.username, "user_id": str(user.user_id)},
                expires_delta=access_token_expires
            )
            
            return LoginResponse(
                request_id=str(uuid.uuid4()),
                user_id=str(user.user_id),
                username=user.username,
                token=access_token,
                response_code=200,
                response_msg="登录成功",
                process_status="completed",
                processing_time=time.time() - start_time
            )
    except Exception as e:
        print(traceback.format_exc())
        return LoginResponse(
            request_id=str(uuid.uuid4()),
            user_id="",
            username="",
            token="",
            response_code=500,
            response_msg=f"登录失败: {str(e)}",
            process_status="completed",
            processing_time=time.time() - start_time
        )


@app.post("/chat")
def chat(req: RAGRequest, token: str = Header(...)) -> RAGResponse:
    start_time = time.time()
    try:
        # 验证JWT令牌
        payload = validate_jwt_token(token)
        if not payload:
            return RAGResponse(
                request_id=str(uuid.uuid4()),
                message=[],
                response_code=401,
                response_msg="无效的token",
                process_status="completed",
                processing_time=time.time() - start_time
            )
        
        # 验证用户ID是否匹配
        if payload.get("user_id") != req.user_id:
            return RAGResponse(
                request_id=str(uuid.uuid4()),
                message=[],
                response_code=403,
                response_msg="用户ID与token不匹配",
                process_status="completed",
                processing_time=time.time() - start_time
            )
        
        from src.rag.ragsimple import create_conversational_rag
        
        # 使用用户ID作为会话ID的基础
        session_id = f"user_{req.user_id}_{req.knowledge_id}"
        
        # 创建对话RAG实例
        conversational_rag = create_conversational_rag(knowledge_id=req.knowledge_id)
        
        # 调用对话方法
        response = conversational_rag.invoke(
            {"input": req.message},
            config={"configurable": {"session_id": session_id}}
        )
        
        return RAGResponse(
            request_id=str(uuid.uuid4()),
            message=response,
            response_code=200,
            response_msg="ok",
            process_status="completed",
            processing_time=time.time() - start_time
        )
    except Exception as e:
        print(traceback.format_exc())
        return RAGResponse(
            request_id=str(uuid.uuid4()),
            message=[],
            response_code=500,
            response_msg=f"对话失败: {str(e)}",
            process_status="completed",
            processing_time=time.time() - start_time
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config_manager.config.rag.port, workers=1)
