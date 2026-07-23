import json
import os
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200 # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20
@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    # 记录任务状态
    add_running_task(state["task_id"], "node_document_split")
    # 从 `state` 中提取 Markdown 内容md_content与文件标题file_title，统一换行符格式（`\r\n` / `\r` → `\n`），保证跨平台兼容。
    md_content, file_title=step_1_get_content(state)
    # 基于 Markdown 标题语法（`#` ~ `######`）进行**语义级切分**，自动跳过代码块内的标题匹配，避免误切注释，保证每个块语义完整。
    # 若文档无任何标题，自动生成默认标题 `无主题`，确保内容不丢失、流程不中断。
    sections=step_2_split_by_title(md_content, file_title)
    # 使用 `RecursiveCharacterTextSplitter` 对**超过指定长度**的语义块进行二次切割，按「段落 → 换行 → 句子 → 空格」优先级切割，**不产生碎片、不硬断句子、无需手动合并。
    final_chunks=step_3_refine_chunks(sections)
    # 将切分结果备份到本地 `chunks.json` 文件，同时将最终 chunks 存入 `state`，供后续向量入库使用。
    step_4_backup_chunks(final_chunks,state)
    state["chunks"]=final_chunks
    add_done_task(state["task_id"], "node_document_split")
    return state

@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
   md_content=state.get("md_content")
   file_title=state.get("file_title")
   #判断md_content是否为空
   if not md_content:
       logger.error("文档内容为空，请检查输入参数")
       raise ValueError("文档内容为空，请检查输入参数")
   md_content=md_content.replace("\r\n", "\n").replace("\r", "\n")
   return md_content, file_title
@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content, file_title):
    #设置匹配标题的正则表达式
    title_pattern=re.compile(r"^\s*#{1,6}\s+.+")
    #将md_content按'\n'切割
    lines=md_content.split("\n")
    #创建一个列表，用于存储每个标题数据片段
    chunks=[]
    #存储当前遍历出的标题
    current_title=""
    #存储当前标题下遍历出来的行
    current_lines=[]
    #是否为代码块
    is_code_block=False
    #遍历md_content中的每一行
    for line in lines:
        line=line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block=not is_code_block
            current_lines.append(line)
            continue
        if not is_code_block and title_pattern.match(line):
            if current_title:
                chunks.append({
                    "title": current_title,
                    "content":"/n".join(current_lines),
                    "file_title":file_title
                })
            #重置数据
            current_title=line
            current_lines=[line]
        else:
            current_lines.append(line)
    if current_title:
        chunks.append({
            "title": current_title,
            "content": "/n".join(current_lines),
            "file_title": file_title
        })
        #兜底处理，如果md_content中没有标题，则生成默认标题
    if not current_title:
        chunks.append({
            "title": "无主题",
            "content": md_content,
            "file_title": file_title
        })
    return chunks

@step_log("step_3_refine_chunks")
def step_3_refine_chunks(sections):
    #创建RecursiveCharacterTextSplitter
    splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！","?"],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    final_chunks=[]
    for section in sections:
        chunks=splitter.split_text(section["content"])
        #判断二次切分得到的数据片段的个数
        has_multi_chunks=len(chunks)>1
        #对二次切分的结果进行遍历
        for idx,chunk in enumerate(chunks):
            title=f"{section['title']}_{idx}" if has_multi_chunks else section["title"]
            final_chunks.append({
                "title": title,
                "content": chunk,
                "file_title": section["file_title"],
                "parent_title": section["title"],
                "part":idx,
            })
    return final_chunks

@step_log("step_4_backup_chunks")
def step_4_backup_chunks(final_chunks,state):
    # 获取保存数据片段的chunks.json文件的路径
    chunks_json_path = Path(state["md_path"]).parent / "chunks.json"
    # 将最终的数据片段转换为json存储到chunks.json文件中
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(
            final_chunks,
            f,
            ensure_ascii=False,  # 正常输出中文
            indent=4,
        )
    logger.debug(f"数据备份成功,备份地址:{chunks_json_path}")
if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

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
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
