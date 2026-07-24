import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 1.任务状态记录处理
    add_running_task(state['task_id'],'node_pdf_to_md')
    # 调用step_1_validate_paths()进行路径校验
    pdf_path_obj, output_dir_obj = step_1_validate_paths(state)
    # 调用step_2_upload_and_poll()完成和minerU的交互
    full_zip_url = step_2_upload_and_poll(pdf_path_obj)
    # 调用step_3_download_and_extract()进行下载和解压
    md_path = step_3_download_and_extract(full_zip_url, output_dir_obj, pdf_path_obj.stem)
    # 更新状态md_path和md_content
    state["md_path"]=md_path
    with open(md_path,"r",encoding="utf-8") as file:
        state["md_content"]=file.read()
    add_done_task(state["task_id"],"node_pdf_to_md")
    return state

@step_log("step_1_validate_paths")
def step_1_validate_paths(state: ImportGraphState):
    # 1.获取pdf_path和local_dir路径参数
    pdf_path=state.get("pdf_path","").strip()
    local_dir=state.get("local_dir","").strip()
    # 2.参数非空校验
    if not pdf_path:
        raise ValueError(f"pdf_path 不能为空，请提供有效的PDF文件路径")
    if not local_dir:
        local_dir=PROJECT_ROOT / "output"
        state["local_dir"]=str(local_dir)
        logger.warning(f"未指定输出目录，使用默认路径：{local_dir}")
    # 3. 统一转换为Path对象，标准化路径处理
    pdf_path_obj=Path(pdf_path)
    local_dir_obj=Path(local_dir)
    # 4. 路径有效性校验（差异化处理：输入严格校验，输出自动修复）
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"pdf_path 指定的文件不存在：{pdf_path_obj}")
    if not local_dir_obj.exists():
        logger.warning(f"输出目录不存在，自动创建：{local_dir_obj}")
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    return pdf_path_obj, local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj: Path):
    # 判断.env中MinerU相关的API_KEY和BASE_URL是否为空
    if not mineru_config.api_key or not mineru_config.base_url:
        raise ValueError("MinerU相关配置API_KEY和BASE_URL缺失，请检查之后重试")
    # 向MinerU发送第一次请求，获取batch_id和file_url(上传pdf文件的连接地址)
    # 获取访问MinerU的token和url
    token=mineru_config.api_key
    url=f"{mineru_config.base_url}/file-urls/batch"
    #设置请求的请求头
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    #设置请求的请求体
    data = {
        "files": [
            {"name": pdf_path_obj.name}
        ],
        "model_version": "vlm"
    }
    response=requests.post(url,headers=header,json=data)
    if response.status_code !=200:
        raise RuntimeError(f"获取上传地址请求失败，状态码：{response.status_code},响应内容：{response.text}")
    #获取响应体
    result = response.json()
    #通过code判断请求是否成功
    if result["code"] !=0:
        raise RuntimeError(f"接口调用失败，接口调用状态：{result['code']},接口处理信息：{result['msg']}")
    #当请求成功接口调用成功获取batch_id和file_url
    batch_id=result["data"]["batch_id"]
    file_url=result["data"]["file_urls"][0]

    #向MinerU发送第二次请求，将pdf文件上传到指定地址file_url
    #读取pdf文件中的所有内容
    pdf_file_data=pdf_path_obj.read_bytes()
    #通过Session发put请求完成pdf文件的上传
    with requests.Session() as session:
        session.trust_env = False#忽略系统所有代理环境变量
        upload_response=session.put(file_url,data=pdf_file_data,timeout=60)
        if upload_response.status_code !=200:
            raise RuntimeError(f"pdf文件上传失败，状态码：{upload_response.status_code},响应内容：{upload_response.text}")
    #进行任务解析获取任务结果
    url=f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds=600#最大超时时间
    poll_interval=3#轮询间隔
    start_time=time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"任务解析超时，请检查任务是否正常执行")
        try:
            #发送请求获取结果
            poll_response=requests.get(url,headers=header)
        except Exception as e:
            logger.warning("通过batch_id获取任务结果的请求失败，错误信息：{}", e)
            time.sleep(poll_interval)
            continue
        #判断响应状态码
        if poll_response.status_code !=200:
            if 500<=poll_response.status_code<600:
            # 若响应状态码为5xx，则表示服务器端出现问题，直接轮询继续请求
                logger.warning(f"通过batch_id获取任务结果的服务器端出现错误，请重试")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(f"通过batch_id获取任务结果请求失败，状态码：{poll_response.status_code},响应内容：{poll_response.text}")
        #通过code获取接口调用情况
        poll_result=poll_response.json()
        if poll_result["code"] !=0:
            raise RuntimeError(f"接口调用失败，接口调用状态：{poll_result['code']},接口处理信息：{poll_result['msg']}")
        #获取任务结果
        extract_result=poll_result["data"]["extract_result"][0]
        #判断extract_result是否为空
        if not extract_result:
            logger.warning(f"任务结果为空，请检查任务是否正常执行")
            time.sleep(poll_interval)
            continue
        #判断任务状态
        if extract_result["state"]=="done":
            #表示任务已完成，获取pdf转为md 文件的url
            full_zip_url=extract_result["full_zip_url"]
            #判断full_zip_url是否为空
            if not full_zip_url:
                raise RuntimeError("任务处理失败，没有获取到pdf转换为md文件的链接地址")
            return full_zip_url
        elif extract_result["state"]=="failed":
            err_msg = extract_result.get("err_msg") or extract_result.get("error_msg") or "未知错误"
            raise RuntimeError(f"MinerU 任务处理失败：{err_msg}")
        else:
            logger.warning(f"任务处理中，请稍后重试")
            time.sleep(poll_interval)
            continue

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(full_zip_url: str, output_dir_obj: Path,stem: str):
    """
        参数：zip_url , out_dir_obj , 原文件名 path.stem
        返回：解压后的.md的str地址
        1. zip下载 get    output / stem_result.zip
        2. 检查解压的文件夹地址  output / stem
        3. 检查解压的文件夹进行防重复处理
        4. 进行解压 zipFile  extractall(解压的目标文件夹)
        5. 考虑文件名字 原文件件名 还是 full 还是其他
        6. 重命名处理
        7. 路径转成字符串 获取绝对路径最终返回即可！
        """
    #发送请求，根据url下载文件
    response=requests.get(full_zip_url, timeout=60)
    #判断响应状态码
    if response.status_code !=200:
        raise RuntimeError(f"pdf文件下载失败，状态码：{response.status_code},响应内容：{response.text}")
    #设置保存zip文件的路径
    zip_save_path=output_dir_obj / f"{stem}.zip"
    #将响应的内容写出到zip_save_path所对应的文件中
    zip_save_path.write_bytes(response.content)
    #设置要解压的路径地址
    extract_path=output_dir_obj / f"{stem}"
    #判断extract_path文件夹是否存在
    # 若存在，表示之前有过相同的操作，保存的是之前的数据，需要删除，保存本次的数据
    if extract_path.exists():
        shutil.rmtree(extract_path)
    # 若不存在，则进行创建
    extract_path.mkdir(parents=True, exist_ok=True)
    #进行解压
    with zipfile.ZipFile(zip_save_path, "r") as zip_file:
        zip_file.extractall(extract_path)
    # 获取extract_path所对应目录中的所有md文件
    md_files=list(extract_path.glob("*.md"))#rglob = recursive glob，递归遍历
    # 判断md_files中是否有md文件存在
    if not md_files:
        raise RuntimeError("pdf文件转换md文件失败，没有获取到pdf文件对应的md文件")
    target_md_file=None
    # 遍历所有的md文件，找到和原pdf文件名相同的md文件
    for md_file in md_files:
        if md_file.stem==stem:
            target_md_file=md_file
            break
    if not target_md_file:
        for md_file in md_files:
            if md_file.name=="full.md":
                target_md_file=md_file
                break
    # 若md文件的名字和原pdf文件的名字不一致，修改md文件的名字为原pdf文件的名字
    if target_md_file.stem != stem:
        target_md_file=target_md_file.rename(target_md_file.with_name(f"{stem}.md"))
    # 返回最终md文件的绝对路径
    final_md_file_path=str(target_md_file.resolve())
    return final_md_file_path

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")