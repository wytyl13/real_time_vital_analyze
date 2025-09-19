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
    overload,
    Type
)

from pydantic import Field, BaseModel


from agent.tool.json_processor import JsonProcessor
from agent.base.base_tool import tool
from agent.llm_api.ollama_llm import OllamaLLM



class ShuifeiSchema(BaseModel):

    name: str = Field(
        description="用户的名称"
    )
    room_id: str = Field(
        ...,
        description="户号"
    )



@tool
class shuifei():
    description: str = "查用户的水费"
    args_schema: Type[BaseModel] = ShuifeiSchema
    
    @overload
    def __init__(
        self, 
    ):
        ...
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            


    async def execute(
        self, 
        name,
        room_id
    ) -> Any:
        print("shuifei: 200元")
        
        
@tool
class ExtractField2(JsonProcessor):
    
    
    @overload
    def __init__(
        self, 
    ):
        ...
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            

    async def execute(
        self, 
    ) -> Any:
        
        print("ExtractField2")



if __name__ == '__main__':
    shuifei_ = shuifei()
    # extract_field_1 = ExtractField1()

    tools_info = [shuifei_.tool_schema]
    toos = {
        
        "type": "tools",
        "tools_info": tools_info
    }
