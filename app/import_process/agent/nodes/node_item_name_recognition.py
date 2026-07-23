import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task

# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500
@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    # 日志和任务处理
    add_running_task(state['task_id'], 'node_item_name_recognition')
    # 1. 校验和取值 （file_title,chunks）
    # 获取前置的材料！ file_title = 为了兜底，没有识别到item_name
    chunks, file_title = step_1_get_chunks_and_file_title(state)
    # 2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本
    context = step_2_build_context(chunks)
    # # 3. 调用模型，拼接提示词，识别chunks对应item_name
    item_name = step_3_call_llm(context, file_title)
    # # 4. 修改state chunks -》 item_name -> chunks [{title parent_title context part item_name [没有值]}]
    step_4_update_chunks_and_state(state, item_name, chunks)
    # # 5. item_name生成向量（稠密/稀疏）
    dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
    # # 6. 将向量存储到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
    step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)

    add_done_task(state['task_id'], 'node_item_name_recognition')
    return state

@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state:ImportGraphState):
    chunks=state.get("chunks")
    file_title=state.get("file_title")
    if not chunks:
        raise ValueError("chunks没有值，无法正常运行")
    if not file_title:
        file_title=Path(state["md_path"]).stem
        state["file_title"]=file_title
    return chunks,file_title

@step_log("step_2_build_context")
def step_2_build_context(chunks):
    """
        构建提示词上下文环境
        根据chunks切面的content内容进行分拼接！ （2000）
        截取内容限制： 1. 最多截取前top个 （5） 2. 最多字符不能超过 CONTEXT_TOTAL_MAX_CHARS
        截取内容处理：
              切片：{1}，标题:{title},内容：{content} \n\n
              切片：{2}，标题:{title},内容：{content} \n\n
              切片：{3}，标题:{title},内容：{content} \n\n
              切片：{4}，标题:{title},内容：{content} \n\n
              切片：{5}，标题:{title},内容：{content} \n\n
        :param chunks:
        :return:
        """
    #处理后的切片列表
    parts=[]
    total_chars=0
    chunks=chunks[:DEFAULT_ITEM_NAME_CHUNK_K]
    #遍历每个切片
    for index,chunk in enumerate(chunks,start=1):
        title=chunk["title"]
        content=chunk["content"]
        #拼接
        data=f"切片：{index}，标题：{title}，内容：{content}"
        #将拼接之后的内容进行存储
        parts.append(data)
        #记录切片的总长度
        total_chars+=len(data)
        if total_chars >=CONTEXT_TOTAL_MAX_CHARS:
            logger.warning("切片总长度超过上限（2500）")
            break
    context="\n\n".join(parts)
    #兜底
    context=context[:CONTEXT_TOTAL_MAX_CHARS]
    return context

@step_log("step_3_call_llm")
def step_3_call_llm(context,file_title):
    #分别获取用户提示词和系统提示词
    human_prompt=load_prompt("item_name_recognition",file_title=file_title,context=context)
    system_prompt=load_prompt("product_recognition_system")
    message=[
        HumanMessage(human_prompt),
        SystemMessage(system_prompt)
    ]
    model=get_llm_client()
    chains=model | StrOutputParser()
    item_name=chains.invoke(message)
    if not item_name:
        item_name=file_title
    return item_name

@step_log("step_4_update_chunks_and_state")
def step_4_update_chunks_and_state(state,item_name,chunks):
    state["item_name"]=item_name
    for chunk in chunks:
        chunk["item_name"]=item_name
    state["chunks"]=chunks

@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name):
    result=generate_embeddings([item_name])
    return result["dense"][0],result["sparse"][0]

@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    milvus_client=get_milvus_client()
    if not milvus_client.has_collection(milvus_config.item_name_collection):

        schema=milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        schema.add_field(field_name="pk",datatype=DataType.INT64,is_primary=True)
        schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
        index_params=milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"{milvus_config.item_name_collection}创建成功")
    #将之前关于item_name的数据删除
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f"item_name == '{item_name}'"
    )
    #准备添加的数据(要插入的数据)
    data={
        "file_title":file_title,
        "item_name":item_name,
        "dense_vector":dense_vector,
        "sparse_vector":sparse_vector
    }
    #重新添加item_name所对应的数据
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data=data
    )
    logger.info(f"{item_name}相关向量添加成功")

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

    except Exception as e:
        logger.error("商品名称识别节点本地测试失败，原因：{}", str(e), exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()