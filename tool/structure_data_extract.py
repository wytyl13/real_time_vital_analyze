#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/09/04 14:04
@Author  : weiyutao
@File    : structure_data_extract.py
"""
from typing import (
    Optional,
    Dict,
    Any,
    List,
    overload
)


from agent.tool.json_processor import JsonProcessor
from agent.base.base_tool import tool
from agent.llm_api.ollama_llm import OllamaLLM


@tool
class ExtractField(JsonProcessor):

    llm: Optional[OllamaLLM] = None
    default_extract_result: Optional[Dict] = None
    
    
    @overload
    def __init__(
        self, 
        llm,
    ):
        ...
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'llm' in kwargs:
            self.llm = kwargs.get('llm')
            
    async def extract_info(self, 
        extract_content: str = None,
        extract_information: str = None,
        default_extract_result: str = None,
        temperature: float = 0.0,
        str_flag: int = 0
    ) -> Dict[str, Any]:
        prompt = self.system_prompt.replace(
            "{content}", extract_content
        )
        prompt = prompt.replace(
            "extract_information", 
            str(extract_information)
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        print(messages)
        response = await self.llm._whoami_text(messages=messages, timeout=10, use_tool=False)
        if not isinstance(response, list):
            response = [{"from": "default_key", "to": "default_value"}]
        return self.parse_json_response(json_response=response)


    async def execute(
        self, 
        str_flag: int = 0,
        extract_content: str = None,
        extract_information: str = None,
        default_extract_result: str = None
    ) -> Any:
        
        if extract_content == "" or extract_content is None:
            raise ValueError("extract_content must not be null!")
        
        if extract_information == "" or extract_information is None:
            raise ValueError("extract_information must not be null!")
        
        return await self.extract_info(
            str_flag=str_flag,
            extract_content=extract_content,
            extract_information=extract_information,
            default_extract_result=default_extract_result
        )


if __name__ == '__main__':
    from agent.config.model_config import ModelConfig
    from agent.config.llm_config import LLMConfig
    from pathlib import Path
    
    llm = OllamaLLM(config=LLMConfig.from_file(Path("/work/ai/word2html/config/yaml/ollama_config.yaml")))
    # enhance_llm = EnhanceRetrieval(llm=llm)
    # print(enhance_llm)
    extract_content = """
    运城市枫之逸养老服务中心走失风险评估表
    姓  名：   replace_name            性  别：   replace_gender            年  龄：  replace_age             档案号：   replace_file_number    
    参   数	得 分	评估标准	说明
    定向能力	1	不能说出今天的具体时间（年月日星期）	replace_description
        0	能说出今天的具体时间（年月日星期）	
        1	不能说出所处的具体位置（省市县乡镇街道）	
        0	能说出所处的具体位置（省市县乡镇街道）	
    走失史	1	之前有走失史	
        0	之前没有走失史	
    意识状态	1	有意识障碍	
        0	没有意识障碍	
    心理状态	1	情绪低落、焦虑、抑郁等	
        0	状态良好	


    疾病史	4	3种以上疾病	①心脑血管病变（脑出血、脑梗塞、脑萎缩等）、②术后认知功能障碍③定向障碍（脑炎、肝性脑病、神经性脑瘫等）④记忆或认知功能障碍（智障、老年痴呆、癫痫等）、⑤精神行为异常（精神分裂、抑郁、脑瘫、癫痫等）
        3	3种	
        2	2种	
        1	1种	
        0	无	


    用药史	4	3种以上疾病	
    ①三环类抗抑郁药（丙咪嗪、氯米帕明、阿米替林、多塞平、马普替林）②抗癫痫药物（苯巴比妥、苯妥英钠、卡马西平）
        3	3种	
        2	2种	
        1	1种	
        0	无	
    走失风险评估总分	☐	上述各项得分之和☐

    走失风险评估分级	
    □级	0能力完好
    1低风险1—3分☐
    2中风险4—9分
    3高风险 ≧10分
    备注：遇有以下情况直接评估
    1、完全麻痹、完全瘫痪的服务对象直接视为低分险；
    2、入院前6个月内有≧2次以上走失史，或入住后发生走失的视为高风险。

    评估人：   replace_evalutor      评估医生：   replace_evalution_doctor       审核人：    replace_reviewer      服务对象:   replace_service_object       服务对象家属：   replace_service_object_family      

    评估日期：     replace_evalution_date       
    """
    
    
    extract_information = [
        {"from": "姓名", "to": "name"},
        {"from": "性别", "to": "gender"},
        {"from": "年龄", "to": "age"}
    ]
    
    
    default_extract_result = [
        {"from": "name", "to": ""},
        {"from": "gender", "to": ""},
        {"from": "age", "to": ""}
    ]
    default_extract_result = {
        "name": "",
        "age": "",
        "gender": ""
    }
    
    extract_field = ExtractField(llm=llm)

    
    import asyncio
    async def main():
        response = await extract_field.execute(
            extract_content=extract_content,
            extract_information=extract_information,
            default_extract_result=default_extract_result
        )
        print(response)
    
    asyncio.run(main())
