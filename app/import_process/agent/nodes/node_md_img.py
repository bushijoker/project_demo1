import base64
import mimetypes
import os
import re
import sys
from collections import deque
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_running_task, add_done_task

"""
1.  **Step 1：初始化校验**：读取MD路径与内容，校验文件合法性，定位同级images文件夹。
2.  **Step 2：图片扫描与上下文匹配**：筛选支持格式的图片，校验MD引用关系，截取图片前后各100字符上下文。
3.  **Step 3：VLM语义生成**：调用千文Qwen3-VL-Flash，Base64编码图片+上下文构造请求，生成规范语义描述。
4.  **Step 4：上传与替换**：清理MinIO旧资源，批量上传图片生成在线URL，替换MD本地路径并填充alt语义。
5.  **Step 5：保存与状态更新**：生成「原文件名_new.md」备份，更新流程状态，完成闭环。
"""
# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(image_name:str)->bool:
    return os.path.splitext(image_name)[1].lower() in IMAGE_EXTENSIONS

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    # 1. 进行任务和日志处理
    add_running_task(state["task_id"], "node_md_img")
    # 2. 进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # 3. 查找md中使用的图片和上下文，传入md_content和images文件夹,返回进行模型访问准备 [(图片名,图片地址,(上文,下文))]
    image_targets = step_2_scan_images(md_content, images_dir_obj)
    # 4. 进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结]
    image_summaries = step_3_image_summary(image_targets, md_path_obj.stem)
    # 5. 上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
    new_md_content =step_4_upload_images_replace(image_summaries,image_targets,md_content,md_path_obj.stem)
    # 6. 备份新的md内容,改为原名称 _new.md
    new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)
    # 7. 进行md_path和md_content内容更新(state)
    state["md_path"] = new_md_file_path_str
    state["md_content"] = new_md_content
    add_done_task(state["task_id"], "node_md_img")
    return state

@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    #获取并判断md_path是否为空
    md_path = state["md_path"]
    if not md_path:
        raise RuntimeError("md_path为空，参数异常")
    # 获取md_path所对应的Path对象，判断文件是否存在
    md_path_obj=Path(md_path)
    if not md_path_obj.exists():
        raise RuntimeError(f"md_path所对应的文件不存在: {md_path},参数异常")
    if not state.get("md_content"):
        state["md_content"]=md_path_obj.read_text(encoding="utf-8")
    # 获取md文件中图片所在的路径
    image_dir_obj=md_path_obj.parent / "images"
    return state["md_content"],md_path_obj,image_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content:str,image_dir_obj:Path):
    #创建变量存储最终结果
    image_targets=[]
    #遍历images_dir_obj下的所有图片
    for image_file in image_dir_obj.iterdir():
        image_name=image_file.name
        #判断图片是否是MinIo所支持的文件
        if not is_supported_image(image_name):
            logger.warning(f"图片格式不支持: {image_name}")
            continue
        # 设置正则表达式，匹配md_content中的图片，![](xxx.jpg)
        pattern=re.compile(r"!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)")
        items=list(pattern.finditer(md_content))
        if not items:
            logger.warning(f"图片未找到: {image_name},继续搜索下一个图片")
            continue
        # 获取匹配的内容在md_content中的开始索引和结束索引
        start,end=items[0].span()#正则匹配对象 Match 的 .span() 方法，一次性返回当前匹配内容在原字符串里的起始下标、结束下标
        # 分别获取图片的前100和后100个字符串作为上下文
        pre=md_content[max(start-100,0):start]
        post=md_content[end:min(end+100,len(md_content))]
        image_targets.append((image_name,str(image_file),(pre,post)))
    return image_targets

@step_log("step_3_image_summary")
def step_3_image_summary(image_targets:list,stem):
    #设置一个字典，存储图片所对应的描述文本
    image_summaries={}
    requests_limiter=deque()#实现滑动窗口限流
    for image_name,image_path,context in image_targets:
        #设置限流
        apply_api_rate_limit(requests_limiter,max_requests=100)
        #获取提示词
        prompt=load_prompt("image_summary",root_folder=stem,image_content=context)
        #获取模型对象
        model=get_llm_client(lm_config.lv_model)
        # 判断image_path是否为Path对象，若不是则转换为Path对象
        if isinstance(image_path,str):
            image_path=Path(image_path)
        #将图片的内容转换为base64编码
        image_base64=base64.b64encode(image_path.read_bytes()).decode("utf-8")
        #拼接提示词
        message=HumanMessage(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )
        #获取chain对象
        chain=model | StrOutputParser()
        summary=chain.invoke([message])
        # 保存图片的名称和图片的总结
        image_summaries[image_name]=summary
    return image_summaries

@step_log("step_4_upload_images_replace")
def step_4_upload_images_replace(image_summaries:dict,image_targets:list,md_content:str,stem:str):
    """
        将图片上传到minio服务器!
        同时替代原md内容中的图片地址和描述内容!
        确保任何位置可以进行访问图片和现实
        :param image_summaries: 图片 和 总结
        :param image_targets: 图片名称 地址 和上下文
        :param md_content: 原md内容
        :param stem: md文件名
        :return: 替换后的md_content
        """
    #1.获取minio客户端对象
    minio_client=get_minio_client()
    # 获取knowledge-base-files桶中，/upload-images下以md文件的名字为目录名的目录中所有的文件
    minio_list=minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
        recursive=True#递归搜索
    )
    #准备删除参数
    delete_object_list=[DeleteObject(obj.object_name) for obj in minio_list]
    # 将之前上传的md文件所对应的图片删除
    errors=minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    # 记录删除失败的图片
    for error in errors:
        logger.warning(f"删除图片失败：{error}")
    # 创建存储图片上传到minio中的地址的字典
    image_urls={}
    #上传图片到minio服务器
    for image_name,image_path,_ in image_targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir[1:]}/{stem}/{image_name}",
                file_path=image_path,
                content_type=mimetypes.guess_type(image_name)[0]#根据图片文件名自动猜出图片 MIME 类型（image/jpeg/image/png等）
            )
            image_urls[image_name]=f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
            logger.info(f"图片上传成功：{image_name},url:{image_urls[image_name]}")
        except Exception as e:
            logger.warning("图片上传失败：{},原因：{}", image_name, e)
            logger.info("继续处理下一张图片")
    image_infos={}
    #创建一个字典，准备替换数据,{image_name:(summary,url)}
    for image_name,summary in image_summaries.items():
        image_infos[image_name]=(summary,image_urls[image_name])
    #判断image_urls是否为空
    if image_urls:
        for image_name,(summary,url) in image_infos.items():
            #设置正则表达式，匹配md_content中的图片
            pattern=re.compile(r"!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)")
            #替换图片
            md_content=pattern.sub(lambda _:f"![{summary}]({url})",md_content)
    return md_content

@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj, new_md_content):
    # 保留旧文件，在新文件（原文件名_new.md）中进行替换
    new_md_path_obj=md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
    # 将替换之后的new_md_content写入到新文件中
    new_md_path_obj.write_text(new_md_content,encoding="utf-8")
    return str(new_md_path_obj)

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")