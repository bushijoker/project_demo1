import shutil
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
#定义fastapi对象
app=FastAPI(title="query service", description="掌柜智库查询服务！")

#跨域配置
app.add_middleware(
    CORSMiddleware,#跨域资源共享中间件,自动在 HTTP 响应里添加跨域相关响应头，告诉浏览器**允许这个跨域请求放行**。
    allow_origins=["*"],#允许任意来源域名发起跨域请求。
    allow_methods=["*"],#允许所有 HTTP 请求方法：GET、POST、PUT、DELETE、OPTIONS 等。
    allow_headers=["*"],#允许前端携带任意请求头（比如 Authorization、自定义 Token 头部）。不加这个配置，前端无法传递认证 Token。
)

# --------------------------
# 静态页面路由：返回文件导入前端页面import.html
# 访问地址：http://localhost:8000/import.html
# --------------------------
@app.get("/import.html",response_class=FileResponse)
async def get_import_page():
    """返回文件导入前端页面：import.html"""
    # 拼接HTML文件绝对路径，基于项目根目录定位
    html_abs_path=PROJECT_ROOT / "app/import_process/page/import.html"
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")

    # 校验文件是否存在，不存在则抛出404异常
    #if not os.path.exists(html_abs_path):
    if not html_abs_path.exists():
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")
    # 以FileResponse返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # 显式指定媒体类型为HTML，确保浏览器正确解析
    )
# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
       LangGraph全流程执行后台任务
       核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
       任务状态更新：pending → processing → completed/failed
       节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

       :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
       :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
       :param local_file_path: 上传文件的本地绝对路径
       """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id,"processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")
        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state=get_default_state()
        init_state["task_id"]=task_id#任务ID关联
        init_state["local_dir"]=local_dir#任务本地目录
        init_state["local_file_path"]=local_file_path#上传文件本地路径
        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                add_done_task(task_id, node_name)

            # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")
    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)

@app.post("/upload",summary="文件上传接口",description="支持多文件批量上传，自动触发知识库导入全过程")
async def upload_files(
    background_tasks:BackgroundTasks,
    files:List[UploadFile]=File(...)
):
    # 将output/当前年月日目录设置为存储上传文件的目录
    now_str=datetime.now().strftime("%y%m%d")
    date_dir=PROJECT_ROOT / "output" / now_str
    #创建记录task_id的列表
    task_ids=[]
    #循环处理上传的每个文件
    for file in files:
        task_id=str(uuid.uuid4())
        task_ids.append(task_id)
        #记录当前上传文件的任务状态
        add_running_task(task_id,"upload_file")
        #获取存储上传文件的最终目录
        local_dir_path=date_dir / task_id
        #创建最终目录
        local_dir_path.mkdir(parents=True,exist_ok=True)
        #设置上传文件的具体路径
        upload_file_path=local_dir_path / file.filename
        #保存上传文件（复制文件）
        with upload_file_path.open("wb") as f:
            shutil.copyfileobj(file.file,f)
        #记录当前上传文件的任务状态
        add_done_task(task_id,"upload_file")
        #启动后台任务：LangGraph全流程执行
        background_tasks.add_task(
            run_graph_task,
            task_id=task_id,
            local_dir=str(local_dir_path),
            local_file_path=str(upload_file_path)
        )
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }
@app.get("/status/{task_id}",summary="任务状态查询",description="根据task_id查询单个文件的处理进度和全局状态")
def get_task_progress(task_id:str):
    return {
        "code":200,
        "task_id":task_id,
        "status":get_task_status(task_id),
        "done_list":get_done_task_list(task_id),
        "running_list":get_running_task_list(task_id)
    }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)