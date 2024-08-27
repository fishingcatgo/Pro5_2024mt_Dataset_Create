

#1、数美接口
import requests
import pandas as pd
from io import BytesIO
import time
from openpyxl import load_workbook
url = 'http://api-text-bj.fengkongcloud.com/text/v4'
access_key = 'rVnVfegJXYbShZL9zRxa'
uid = 'None'

def shumei_api(text):
    # print(text)
    payload = {
                'accessKey': access_key,
                'appId': 'default',
                'eventId': 'input',
                'type': 'TEXTRISK',
                'kbType': 'PKB',
                'data': {
                    'text': text,
                    'tokenId': uid,
                }
            }
    res = requests.post(url, json=payload)
    # print('.............................................\n')
    # print(res.json())
    # print(res.json()['riskLabel1'])
    # return res.json()['riskLabel1']
    return res.json()
# text = '介绍一下习近平'
# rest=shumei_api(text)
# print('打标tag:',rest)


#2、加载vllm模型
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
print("CUDA_VISIBLE_DEVICES set to:", os.environ["CUDA_VISIBLE_DEVICES"])

# GLM-4-9B-Chat-1M
# max_model_len, tp_size = 1048576, 4
# 如果遇见 OOM 现象，建议减少max_model_len，或者增加tp_size
max_model_len, tp_size = 131072, 1
model_name = "/nfs2/zhaochuan.cai/czc_data/glm-4-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    enforce_eager=True,
    # gpu_memory_utilization=0.8,
    # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
    # enable_chunked_prefill=True,
    # max_num_batched_tokens=8192
)


def vllm_data(llm,tokenizer,chunk_df,col_name,dataset_name,save_file_path):
    # user_messages=['你是谁','给我讲个故事','给我写首歌','分析最近股市行情']
    # prompt=[ [{"role": "user", "content": val},] for val in user_messages]

    # vllm_data(llm,tokenizer,chunk_df['text'].tolist(),batch_data_rowsrc)
    # vllm_data(llm,tokenizer,chunk_df)
    # print('Excel Sheet to Dict:', excel_data_df.to_dict(orient='records')[:6]) #取前6个

    batch_data_rowsrc=chunk_df.to_dict(orient='records')
    user_messages=chunk_df[col_name].tolist()
    # prompt=[[{"role": "user", "content": f"""
    #         将以下消息翻译成中文，如果是中文直接输出：
    #         ```{issue}```
    #         """},] for issue in user_messages]

    prompt=[[{"role": "user", "content": f"""将以下消息翻译成中文，如果是中文直接输出：```{issue}```"""},] for issue in user_messages]
    #  prompt= f"""将以下消息翻译成中文，如果是中文直接输出：```{issue}```"""

    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=0, max_tokens=1024, stop_token_ids=stop_token_ids)

    inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

    # print(outputs[0].outputs[0].text)
    # print(outputs)
    data=[]

    for idx,val in enumerate(outputs):
        print(f'\n{idx}原文：',user_messages[idx])
        print(f'\n{idx}原文的行：',batch_data_rowsrc[idx])
        print(f'{idx}翻译结果：',val.outputs[0].text)

        # rest=shumei_api(val.outputs[0].text)
        rest={"测试":"测试忽略"}
        print('打标tag:',rest)
        # data.append({"source":"civil_comments", "sentence":val.outputs[0].text, "shumei":rest, "meta":batch_data_rowsrc[idx].tolist()})
        data.append({"source":dataset_name, "sentence":val.outputs[0].text.lstrip("\n").replace("```", ""), "shumei":rest, "meta":batch_data_rowsrc[idx]})

        # print(f'\n\n个输出：',i)
        # print('')
    print('json数据：\n',data)
   
    import json
    # 将数据保存为JSONL文件，追加
    with open(f'{save_file_path}', 'a') as f:
        for item in data:
            f.write(json.dumps(item,ensure_ascii=False) + '\n')

# user_messages=['你是谁','给我讲个故事','给我写首歌','分析最近股市行情']
# vllm_data(llm,tokenizer,user_messages)
# quit('测试')


#遍历子目录
# path='./dataset'
# for dir in next(os.walk(path))[1]:
#     # data_dir='./data/civil_comments/data/'
#     files = os.listdir(f'{path}/{dir}') 
#     print('目录：',dir)
#     for file in files :
#         if file.endswith('.parquet') or file.endswith('.csv'):
#             print(file)


#3、vllm处理数据

if __name__ == '__main__':

    data_dir='./dataset'
    out_dir='./data_out_use2'
    data_col={"Toxic_Comment_Classification_Challenge":"comment_text","civil_comments":"text"} #数据所在的列
    batch_size=10000 #推理批大小
    for dir in next(os.walk(data_dir))[1]: ##遍历子目录
        # files = os.listdir(data_dir) 
        files = os.listdir(f'{data_dir}/{dir}') 
        print('目录：',dir)
        for file in files :  #遍历文件
            if file.endswith('.parquet'):
                print(file)
                file_name,suffix = file.split(".")
                print(file_name,suffix)

                # #清空jsonl文件
                # if os.path.exists(r'./vllm_data.jsonl'):
                #     os.remove(r'./vllm_data.jsonl')
                #清空jsonl文件
                file_outpath=f'{out_dir}/{dir}/{file_name}.jsonl'
                dir_name=f'{out_dir}/{dir}'
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                if os.path.exists(rf'{file_outpath}'):
                    os.remove(rf'{file_outpath}')
                # if os.path.exists(rf'./data_out_test/{dir}/{file_name}.jsonl'):
                #     os.remove(rf'./data_out_test/{dir}/{file_name}.jsonl')

                flag=0
                # chunk_df读取一部分
                import pyarrow.parquet as pq
                # parquet_file = pq.ParquetFile("./data/civil_comments/data/train-00000-of-00002.parquet") 
                parquet_file = pq.ParquetFile(f'{data_dir}/{dir}/{file}') 
                for i in parquet_file.iter_batches(batch_size=batch_size):
                    chunk_df = i.to_pandas()

                    # print('chunk_df读取一部分:\n',chunk_df)
                    # print('列名：',chunk_df.columns.ravel())
                    # # print('某列的数据：',excel_data_df['FirstName'].tolist())
                    # print('某列的前几个数据：',chunk_df['text'].tolist())
                    # 将DataFrame对象转换为numpy数组
                    # batch_data_rowsrc = chunk_df.values

                    # 选多少条输出 
                    # flag +=1
                    # if  flag==5: break

                    # vllm_data(llm,tokenizer,chunk_df['text'].tolist(),batch_data_rowsrc)
                    col_name=data_col[dir]
                    dataset_name=dir
                    vllm_data(llm,tokenizer,chunk_df,col_name,dataset_name,file_outpath)

            if file.endswith('.csv'):
                print(file)
                file_name,suffix = file.split(".")
                print(file_name,suffix)
                #清空jsonl文件
                file_outpath=f'{out_dir}/{dir}/{file_name}.jsonl'
                dir_name=f'{out_dir}/{dir}'
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                if os.path.exists(rf'{file_outpath}'):
                    os.remove(rf'{file_outpath}')

                flag=0
                # chunk_df读取一部分
                csv_iter=pd.read_csv(f'{data_dir}/{dir}/{file}', chunksize=batch_size)
                for chunk_df in csv_iter:
                    print('chunk数：',chunk_df)
                    print('列名：',chunk_df.columns.ravel())
                    # print('某列的数据：',chunk['1'].tolist())
                    # print('某列的前几个数据：',chunk['text'].tolist())

                    # 选多少条输出 
                    # flag +=1
                    # if  flag==5: break
                    #数据所在的列
                    # col_name='comment_text' 
                    col_name=data_col[dir]
                    dataset_name=dir
                    vllm_data(llm,tokenizer,chunk_df,col_name,dataset_name,file_outpath)









