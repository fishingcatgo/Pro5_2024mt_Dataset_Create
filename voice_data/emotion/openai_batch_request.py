import aiohttp
import asyncio

import os
import json
import re

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

def fix_json(json_string):
    # 移除开头和结尾的空白符，并去掉多余的空格
    # json_string = json_string.strip()

    # 移除外部的多余内容，只保留花括号内的内容
    json_string = re.sub(r'^[^{]*{|}[^}]*$', '', json_string).strip()

    print('移除外部的多余内容:',json_string)

    # 如果字符串不以 '{' 开头，则添加 '{'
    if not json_string.startswith('{'):
        json_string = '{' + json_string
      
    # 如果字符串不以 '}' 结尾，则添加 '}'
    if not json_string.endswith('"}')  :
        json_string = json_string + '"}'
    
     # 如果字符串不以 '}' 结尾，则添加 '}'
    if not json_string.endswith('}'):
        json_string = json_string + '}'
    
   # 尝试将字符串修正为包含一对花括号的格式
    if not (json_string.startswith('{') and json_string.endswith('}')):
        json_string = '{' + json_string.strip('{}') + '}'
    # 尝试将单引号替换为双引号
    json_string = json_string.replace("'", '"')
    
    # 添加缺少的引号（简单示例，可能需要更复杂的逻辑）
    # json_string = re.sub(r'(\w+):', r'"\1":', json_string)

    # 移除尾部逗号
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r',\s*]', ']', json_string)
    
    return json_string

def check_and_fix_json(json_str):
    if is_valid_json(json_str):
        return json_str
    else:
        fixed_str = fix_json(json_str)
        print('修复后',fixed_str)
        if is_valid_json(fixed_str):
            return fixed_str
        else:
            raise ValueError("无法修正为有效的 JSON 格式")

# 要请求的URL
url = "https://api.openai.com/v1/chat/completions"

headers = {
  'Authorization': 'Bearer sk-xxx',
  'Content-Type': 'application/json',
  # 'Cookie': '__cf_bm=fGp5y4Qn_Qg8mF3bZFuXVofslJjFKFk9S43LcUIDXpI-1723704451-1.0.1.1-6fh0oXBorJv9RcM5QNHVtFJmDk.rxSOuaEgIGaONBvaYtTpWmeM.p.5M7ECm4ihQcJHJ.eeJgObYLzWOJqvloA; _cfuvid=b0K6Fx1KZDpXh7M4w.jzA.9m4zmITITJnkRzQU8LdrA-1723703518545-0.0.1.1-604800000'
}

# 要发送的数据
# data_list = [{"name":"urm1","data":"post请求数据1"},{"name":"urm2","data":"post请求数据2"},{"name":"urm3","data":"post请求数据3"}]

# data=['讲个故事','讲个笑话','分析股市']

data=[
  f"""构造一条快乐情感相关语音对话，一问一答，字数不大于100。并以 JSON 格式提供，其中包含以下键:questions 、 answers。例如：{{"questions":"xxx","answers":"xxx"}}""",
  f"""构造一条悲伤情感相关语音对话，一问一答，字数不大于100。并以 JSON 格式提供，其中包含以下键:questions 、 answers。例如：{{"questions":"xxx","answers":"xxx"}}""",
  f"""构造一条恐惧情感相关语音对话，一问一答，字数不大于100。并以 JSON 格式提供，其中包含以下键:questions 、 answers。例如：{{"questions":"xxx","answers":"xxx"}}"""
]
# data_list=[ item['messages'][0]['content']==i for i in data]
data_list=[ {
  "model": "gpt-4",
  "max_tokens":200,
  "temperature":0.5,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": i
    }
  ],
  "stream": False,
  # "response_format":"json"
} for i in data]

print(data_list)


#批量异步请求
async def fetch_data(session, data):
    try:
        # async with session.post(url, json=data, headers=headers) as response:
        async with session.post(url, json=data, headers=headers,timeout=aiohttp.ClientTimeout(total=300)) as response:
            response.raise_for_status() # 确保响应状态码为2xx，否则引发HTTPError
            return await response.json(),data
    except aiohttp.ClientError as e:
        print(f"请求失败: {e}")
        return None,data

async def main():
    con = aiohttp.TCPConnector(ssl=False) #ssl问题
    # async with aiohttp.ClientSession(connector=con, trust_env=True) as session:
    async with aiohttp.ClientSession(connector=con, trust_env=True) as session:
        tasks = [fetch_data(session, data) for data in data_list]
        results = await asyncio.gather(*tasks)
        column_all=[]
        column_right=[]  
        column_error=[]          
        for result,data in results:
            print('数据：',data)
            if result:
                print(result)

                #校验json格式
                try:
                    output_text=result['choices'][0]['message']['content']
                    print(output_text)
                    valid_json_str = check_and_fix_json(output_text)
                    print("有效的 JSON:", valid_json_str)
                    json_obj = json.loads(valid_json_str)
                    # 检查字典中是否包含所需的键
                    keys=['questions' ,'answers']
                    missing_keys = [key for key in keys if key not in json_obj]

                    if missing_keys:
                        print('不包含所有键')
                        column_error.append(['×','不包含所有键',valid_json_str,data['messages'],data])
                        column_all.append(['×','不包含所有键',valid_json_str,data['messages'],data])

                    column_right.append(['√',None,valid_json_str,data['messages'],data])
                    column_all.append(['√',None,valid_json_str,data['messages'],data])

                except ValueError as e:
                    print(e)
                    column_error.append(['×','校验json格式出异常',valid_json_str,data['messages'],data])
                    column_all.append(['×','校验json格式出异常',valid_json_str,data['messages'],data])
            else :
               print(f"请求失败")
               column_error.append(['×','请求失败',None,data['messages'],data])
               column_all.append(['×','请求失败',None,data['messages'],data])
        return column_all,column_right,column_error
              

import pandas as pd
import openpyxl
import shutil
import datetime
# 运行主程序
if __name__ == '__main__':
    column_all,column_right,column_error=asyncio.run(main())
    # print('处理后数据：',results)

    print('所有数据：',column_all)
    print('正常数据：',column_right)
    print('异常数据：',column_error)
    print('*'*100)

    #写入表格
    df_all = pd.DataFrame(
        data=column_all,
        columns=['yes_or_no', 'message_err','out_text','input_text','promt_data']
    )

    df_right = pd.DataFrame(
        data=column_right,
        columns=['yes_or_no', 'message_err','out_text','input_text','promt_data']
    )

    df_error = pd.DataFrame(
        data=column_error,
        columns=['yes_or_no', 'message_err','out_text','input_text','promt_data']
    )

    out_dir='./text_out/Gpt4_out1'
    # if os.path.exists(out_dir): shutil.rmtree(out_dir) #清空目录
    if not os.path.exists(out_dir): os.makedirs(out_dir) #目录不存在，创建

     #filename生成
    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化时间为 "YYYYMMDD_HHMMSS"
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"GPT_{timestamp}.xlsx"

    #写入多个sheet
    with pd.ExcelWriter(f'{out_dir}/{filename}') as writer:
        df_all.to_excel(writer, sheet_name='sheet_all',index=False)
        df_right.to_excel(writer, sheet_name='sheet_right',index=False)
        df_error.to_excel(writer, sheet_name='sheet_error',index=False)

    # df_sheet_all1 = pd.read_excel(f"{out_dir}/Descrip_xlsx.xlsx", sheet_name=None,index_col=0)
    df_sheet_all = pd.read_excel(f'{out_dir}/{filename}', sheet_name=None,index_col=False)
    print('Excel_sheet_all数据：\n',df_sheet_all)
    print('*'*100)

    # # print('数据:',df)

    # # df.to_csv(f"{out_dir}/Descrip_csv1.csv", index=False)
    # # pf_read = pd.read_csv(f"{out_dir}/Descrip_csv1.csv")
    # # print('csv_read数据：\n',pf_read)
    # # print('*'*100)

    # df.to_excel(f"{out_dir}/Descrip_xlsx.xlsx", sheet_name='处理记录',index=False)
    # # df_sheet_all1 = pd.read_excel(f"{out_dir}/Descrip_xlsx.xlsx", sheet_name=None,index_col=0)
    # df_sheet_all1 = pd.read_excel(f"{out_dir}/Descrip_xlsx.xlsx", sheet_name='处理记录',index_col=False)
    # print('Excel_sheet_all1数据：\n',df_sheet_all1)
    # print('*'*100)
