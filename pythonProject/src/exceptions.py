# src/exceptions.py
class BusinessException(Exception):
    """业务异常，由路由主动抛出"""
    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code
        super().__init__(message)


class NotFoundException(BusinessException):
    def __init__(self, message: str = "资源不存在"):
        super().__init__(message, code=404)


class BadRequestException(BusinessException):
    def __init__(self, message: str = "请求参数错误"):
        super().__init__(message, code=400)