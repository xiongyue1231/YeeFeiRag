from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import pymysql
import yaml  # type: ignore
from ..app_config.loder import ConfigLoader

config_manager = ConfigLoader()

db_config = config_manager.config.database
db_type = db_config.engine

if db_type == "sqlite":
    # SQLite 使用文件路径
    db_path = db_config.path
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
else:
    # MySQL 或其他数据库使用 host, port, username, password
    host = db_config.host
    port = db_config.port
    username = db_config.username
    password = db_config.password
    database = db_config.mydb  # 数据库名

    engine = create_engine(
        f"{db_type}://{username}:{password}@{host}:{port}/{database}",
        echo=True
    )

# 创建 Base 类
Base = declarative_base()

# 每次自动创建会话
Session = sessionmaker(bind=engine)


# ORM
# 定义 knowledge_database 表
class KnowledgeDatabase(Base):
    # SQLAlchemy 会自动生成表名
    __tablename__ = 'knowledge_database'

    knowledge_id = Column(Integer, primary_key=True, autoincrement=True, comment='知识库id')  # 主键，自动递增
    title = Column(String(255), comment='知识库名称')  # 名称
    category = Column(String(255), comment='知识库类型')  # 类型
    create_dt = Column(DateTime, default=datetime.utcnow, comment='创建时间')  # 创建时间
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')  # 更新时间

    # 与 KnowledgeDocument 表的关系
    documents = relationship("KnowledgeDocument", back_populates="knowledge")

    def __str__(self):
        return (f"KnowledgeDatabase(knowledge_id={self.knowledge_id}, "
                f"title='{self.title}', category='{self.category}', "
                # f"author_id={self.author_id},"
                f"create_dt={self.create_dt}, "
                f"update_dt={self.update_dt})")


# 定义 knowledge_document 表
class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_document'

    document_id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")  # 文档主键，自动递增
    title = Column(String(255), comment="文档名称")  # 文档名称
    category = Column(String(255), comment="文档类型")  # 文档类型
    knowledge_id = Column(Integer, ForeignKey('knowledge_database.knowledge_id'), comment="知识库id")  # 知识库主键（外键）
    file_path = Column(String(255), comment='文件存储地址')  # 储存地址
    file_type = Column(String(255), comment='数据类型')  # 数据类型
    create_dt = Column(DateTime, default=datetime.utcnow, comment='创建时间')  # 创建时间
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')  # 更新时间

    # 与 KnowledgeDatabase 表的关系
    knowledge = relationship("KnowledgeDatabase", back_populates="documents")


# 定义 user 表
class User(Base):
    __tablename__ = 'user'

    user_id = Column(Integer, primary_key=True, autoincrement=True, comment='用户id')  # 主键，自动递增
    username = Column(String(255), unique=True, nullable=False, comment='用户名')  # 用户名
    password = Column(String(255), nullable=False, comment='密码')  # 密码
    email = Column(String(255), unique=True, nullable=False, comment='邮箱')  # 邮箱
    create_dt = Column(DateTime, default=datetime.utcnow, comment='创建时间')  # 创建时间
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')  # 更新时间

    def __str__(self):
        return (f"User(user_id={self.user_id}, "
                f"username='{self.username}', email='{self.email}', "
                f"create_dt={self.create_dt}, "
                f"update_dt={self.update_dt})")

# 自动创建表，如果不存在
Base.metadata.create_all(engine)
# 每次自动创建会话
Session = sessionmaker(bind=engine)
