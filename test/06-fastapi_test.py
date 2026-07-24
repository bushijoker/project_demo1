import asyncio
import mimetypes

import uvicorn
from fastapi import FastAPI, Header
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, \
    StreamingResponse
from starlette.staticfiles import StaticFiles

app=FastAPI()

# app.mount(
#     "/static",
#     StaticFiles(directory="static"),
#     name="static",
# )

#跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#允许任意来源域名发起跨域请求。
    allow_credentials=True,#允许跨域请求携带凭证信息，凭证包含：Cookie、HTTP Session，客户端证书。
    allow_methods=["*"],#允许所有 HTTP 请求方法：GET、POST、PUT、DELETE、OPTIONS 等。
    allow_headers=["*"],#允许前端携带任意请求头（比如 Authorization、自定义 Token 头部）。不加这个配置，前端无法传递认证 Token。
)
#处理/main的路径处理函数
@app.get("/main")
def main():
    #自动转换为json
    return {"message": "Hello World"}
#测试路径参数
@app.get("/params/path/{username}/{password}/{age}")
def params_path(username:str,password:str,age:int):
    return {"username":username,"password":password,"age":age}
#测试查询参数
@app.get("/params/query")
def params_query(
        username:str | None = None,
        password:str | None = None,
        age:int = 20
):
    return {"username":username,"password":password,"age":age}
#测试请求体参数
class ParamsModel(BaseModel):
    username:str
    password:str
    age:int
@app.post("/params/body")
def params_body(paramsmodel:ParamsModel):
    print(paramsmodel)
    return "helloword"

# 测试获取请求头信息
@app.get("/params/headers")
def test_header(user_agent:str=Header(None)):
    return {"user_agent":user_agent}

# 测试JSONResponse
@app.get("/test/json/response")
def test_json_response():
    return JSONResponse(
        content={"message": "Hello World"},#响应体
        status_code=200,#响应状态码
        headers={"test_header": "test_header_value"}#自定义响应头
    )
#测试FileResponse
@app.get("/test/file/response")
def test_file_response():
    return FileResponse(
        path="04-test_graph_flow.py",
        filename="test.py",
        status_code=200,
        media_type=mimetypes.guess_type("04-test_graph_flow.py")[0],
    )
# 测试HTMLResponse
@app.get("/test/html/response")
def test_html_response():
    return HTMLResponse(
        content="""
        <html>
                <head>
                    <title>首页</title>
                </head>
                <body>
                    <h1>Hello World</h1>
                </body>
            </html>
        """
    )
# 测试PlainTextResponse
@app.get("/test/plaintext/response")
def test_plaintext_response():
    return PlainTextResponse(
        content="helloworld",
    )
# 测试RedirectResponse
@app.get("/test/redirect/response")
def test_redirect_response():
    return RedirectResponse(
        url="/test/html/response",
    )
# 测试StreamingResponse
async def generate_stream():
    #模拟流式输出
    words=["hello","world","!"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")
@app.get("/test/streaming/response")
def test_streaming_response():
    return StreamingResponse(
        content=generate_stream(),
        media_type="text/plain",
    )
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)