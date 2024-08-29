import requests
import html2text
from readability import Document
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import (BaseModel, Field)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FetchPageInput(BaseModel):
    """正しく文字列や数字で入ってくるようにする（Validator的な）
    """
    url: str = Field()
    # page_num: int = Field(0, ge=0) # こちらは簡易的に進めたいので、使用しないことにします

@tool(args_schema=FetchPageInput)
def fetch_page(url, page_num=0, timeout_sec=10):
    """
    ## Toolの説明
    本Toolは指定されたURLのWebページから本文の文章を取得するツールです。
    詳細な情報を取得するのに役立ちます
    ## Toolの動作方法
    1. userがWebページのURLを入力します
    2. assistantはHTTPレスポンスステータスコードと本文の文章内容をusrに回答します

    ## 戻り値の設定
    Returns
    -------
    Dict[str, Any]:
    - status: str
    - page_content
      - title: str
      - content: str
      - has_next: bool
    """

    # [1] requestモジュールで指定URLのＷebページ全体を取得
    try:
        response = requests.get(url, timeout=timeout_sec)
        response.encoding = 'utf-8'
    except requests.exceptions.Timeout:
        return {
            "status": 500,
            "page_content": {'error_message': 'Could not download page due to Timeout Error. Please try to fetch other pages.'}
        }

    # [2] HTTPレスポンスステータスコードが200番でないときにはエラーを返す
    if response.status_code != 200:
        return {
            "status": response.status_code,
            "page_content": {'error_message': 'Could not download page. Please try to fetch other pages.'}
        }

    # [3] 本文取得の処理へ（書籍ではtry-exceptできちんとしていますが、簡易に）
    doc = Document(response.text)
    title = doc.title()
    html_content = doc.summary()
    content = html2text.html2text(html_content)

    # [4] 本文の冒頭を取得
    chunk_size = 1000*3  #【chunk_sizeを大きくしておきます】
    content = content[:chunk_size]

    # [5] return処理
    return {
        "status": 200,
        "page_content": {
            "title": title,
            "content": content,  # chunks[page_num], を文書分割をやめて、contentにします
            "has_next": False  # page_num < len(chunks) - 1
        }
    }