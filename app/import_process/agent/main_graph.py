from dotenv import load_dotenv
from langgraph.graph import StateGraph

from app.import_process.agent.nodes import node_md_img, node_document_split
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState

load_dotenv()

workflow=StateGraph(ImportGraphState)

workflow.add_node(node_entry)
workflow.add_node(node_pdf_to_md)
workflow.add_node(node_md_img)
workflow.add_node(node_document_split)
workflow.add_node(node_item_name_recognition)
workflow.add_node(node_bge_embedding)
workflow.add_node(node_import_milvus)

workflow.set_entry_point("node_entry")
