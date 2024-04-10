from langchain_openai import ChatOpenAI

def get_ChatGPT():
    API_SECRET_KEY = ""  # 填入自己的APIKey
    BASE_URL = ""  # 代理
    llm = ChatOpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL, model="gpt-3.5-turbo", temperature=0)
    return llm


from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InternLM_SQL(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")
    
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any):
        # 重写调用函数
            system_prompt = """You are an expert in SQL, please generate a good SQL Query for Question based on the CREATE TABLE statement.\n"""
        
            messages = [(system_prompt, '')]
            response, history = self.model.chat(self.tokenizer, prompt , history=messages)
            return response
    @property
    def _llm_type(self) -> str:
        return "InternLM-SQL"

