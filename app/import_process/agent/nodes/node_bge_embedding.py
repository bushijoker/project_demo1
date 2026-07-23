import os
import sys

from dotenv import load_dotenv

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    """
       节点: 向量化 (node_bge_embedding)
       为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
       """
    # 日志和任务队列处理
    add_running_task(state['task_id'], "node_bge_embedding")
    # 步骤1：输入数据校验，核心chunks无效则抛出异常
    texts_to_embed = step_1_validate_input(state)
    # 步骤2：初始化BGE-M3模型（单例模式，仅加载一次）
    bge_m3_ef = step_2_init_model()
    # 步骤3：批量生成双向量，为切片绑定向量字段
    output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)
    # 步骤4: 输出数据处理
    state['chunks'] = output_data
    add_done_task(state['task_id'], "node_bge_embedding")
    return state

@step_log("step_1_validate_input")
def step_1_validate_input(state: ImportGraphState):
    chunks=state.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        logger.error("向量化输入校验失败：chunks字段为空或非有效列表")
        raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")
    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(chunks)}")
    return chunks

@step_log("step_2_init_model")
def step_2_init_model():
    try:
        ef=get_bge_m3_ef()
        if ef is None:
            logger.error("向量化模型初始化失败")
            raise ValueError("错误: 无法初始化向量化模型")
        logger.info("向量化模型初始化成功")
        return ef
    except Exception as e:
        # 包装异常信息，明确错误原因和排查方向
        error_msg = f"BGE-M3模型初始化失败：{e}，请检查模型路径/环境变量配置是否正确"
        logger.error(error_msg)
        raise ValueError(error_msg)

@step_log("step_3_generate_embeddings")
def step_3_generate_embeddings(texts_to_embed, bge_m3_ef):
    output_data=[]
    batch_size=5
    total=len(texts_to_embed)
    for i in range(0,total,batch_size):
        batch_texts=texts_to_embed[i:i+batch_size]
        #拼接商品名+切片内容增强核心特征
        try:
            input_texts=[]
            for chunk in batch_texts:
                item_name=chunk.get("item_name")
                content=chunk.get("content")
                text=f"商品：{item_name}，介绍：{content}" if item_name else content
                input_texts.append(text)
            #生成批量向量
            docs_embeddings=generate_embeddings(input_texts)
            if not docs_embeddings:
                logger.error("向量化结果为空：请检查输入文本是否为空")
                output_data.extend(batch_texts)
            # 为当前批次每个切片绑定对应向量，复制原数据避免修改上游源数据
            for j,doc in enumerate(batch_texts):
                item=doc.copy()
                item["dense_vector"]=docs_embeddings["dense"][j]#绑定稠密向量
                item["sparse_vector"]=docs_embeddings["sparse"][j]#绑定稀疏向量
                output_data.append(item)
        except Exception as e:
            # 捕获异常，记录错误信息并跳过当前批次
            logger.error(f"向量化批次处理异常：{e}")
            # 异常批次保留原切片数据，保证数据完整性，后续可人工排查
            output_data.extend(batch_texts)
            # 跳过当前批次，继续处理下一批次
            continue
    return output_data
# ==========================================
# 本地单元测试入口
# 功能：独立验证向量化节点全链路逻辑，无需启动整个LangGraph流程
# 适用场景：本地开发、调试、模型有效性验证
# ==========================================
if __name__ == '__main__':
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游商品名称识别节点产出）
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")

        # 验证向量生成结果（打印向量字段是否存在）
        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | 稀疏向量生成{'' if has_sparse else '未'}成功")

    except Exception as e:
        logger.error("=== 向量化节点本地测试失败 ===" + "错误原因：{}", str(e), exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")