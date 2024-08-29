from itertools import islice
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import (BaseModel, Field)

class SearchDDGInput(BaseModel):
    """正しく文字列で検索クエリが入ってくるようにする（Validator的な）
    　　≒文字列でない検索入力のデータ型をはじく"""
    query: str = Field(description="検索したいキーワードを入力してください")

@tool(args_schema=SearchDDGInput)
def search_ddg(query, max_result_num=5):
    """
    ## Toolの説明
    本ToolはDuckDuckGoを利用し、Web検索を実行するためのツールです。
    ## Toolの動作方法
    1. userが検索したいキーワードに従ってWeb検索します
    2. assistantは以下の戻り値の形式で検索結果をuserに回答します

    ## 戻り値の形式

    Returns
    -------
    List[Dict[str, str]]:
    - title
    - snippet
    - url
    """

    # [1] Web検索を実施
    res = DDGS().text(query, region='jp-jp', safesearch='off', backend="lite")

    # [2] 結果のリストを分解して戻す
    return [
        {
            "title": r.get('title', ""),
            "snippet": r.get('body', ""),
            "url": r.get('href', "")
        }
        for r in islice(res, max_result_num)
    ]