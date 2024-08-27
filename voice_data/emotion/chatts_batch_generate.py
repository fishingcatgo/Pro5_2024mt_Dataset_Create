import requests
import json
import os
import datetime

# url = "http://172.31.208.6:8080/tts"

# url = "http://172.31.208.3:8180/tts"

url = "http://172.31.208.5:9080/tts"

headers = {
  'Content-Type': 'application/json'
}

# {"sex": "女",   "age":"成年"},
# {"sex": "男",   "age":"成年"},
# {"sex": "女",   "age":"成年"},
# {"sex": "女",   "age":"成年"},
# {"sex": "女",   "age":"成年"},
# {"sex": "男",   "age":"成年"},
# {"sex": "男",   "age":"成年"},
# {"sex": "女",   "age":"成年"},
# {"sex": "男",   "age":"成年"},

# 音色选项
voices = {
    "音色1": {"seed": 1111},  "1111": {"age":"成年", "sex": "female"}, 
    "音色2": {"seed": 2222},  "2222": {"age":"成年", "sex": "male"}, 
    "音色3": {"seed": 3333},  "3333": {"age":"成年", "sex": "female"}, 
    "音色4": {"seed": 4444},  "4444": {"age":"成年", "sex": "female"}, 
    "音色5": {"seed": 5555},  "5555": {"age":"成年", "sex": "female"}, 
    "音色6": {"seed": 6666},  "6666": {"age":"成年", "sex": "male"}, 
    "音色7": {"seed": 7777},  "7777": {"age":"成年", "sex": "male"}, 
    "音色8": {"seed": 8888},  "8888": {"age":"成年", "sex": "female"}, 
    "音色9": {"seed": 9999},  "9999": {"age":"成年", "sex": "male"}, 
}

#数据格式
item_data={
    "instruction": "这是一个语音对话，回答为语音。",
    "input": {
      "type": "audio",
      "content": "", #path/to/input_audio1.wav 路径
      "format": "mp3",
      "language": "zh",
      "seed":"",
      "transcription": "",  #我今天真的很开心，因为我通过了考试。text内容
      "emotion": "",   #情绪 开心
      "gender": "",    #female
      "age": "",         #成人
      "speed": "",       #正常
      "volume": "",      #中等
      "tone": ""         #愉快
    },
    "output": {
      "type": "audio",
      "content": "",    #path/to/output_audio1.wav
      "format": "mp3",
      "language": "zh",
      "seed":"",
      "transcription": "", #太好了，祝贺你！我们应该好好庆祝一下。
      "emotion": "",       #兴奋
      "gender": "",        #male
      "age": "",           #成人
      "speed": "",         #正常
      "volume": "",        #中等
      "tone": ""           #明亮
    }
  },


import time
def save_voice(data_zip,out_dir,gpt_gene):
  questions,answers=data_zip
  
  res_questions = requests.request("POST", url, headers=headers, data=json.dumps(questions),stream=True,timeout=600)
  # 暂停三秒
  time.sleep(3)
  res_answers = requests.request("POST", url, headers=headers, data=json.dumps(answers),stream=True,timeout=600)
  # 暂停三秒
  time.sleep(3)


  # 检查请求是否成功
  if res_questions.status_code == 200 and res_answers.status_code == 200 :
    # 确保目录存在
    try:
        # out_dir='./out/test1'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 获取文件类型（如果响应头中包含Content-Type）
        content_type = res_questions.headers.get('Content-Type', 'application/octet-stream')
        file_extension = content_type.split('/')[-1]  # 获取文件扩展名

        print('content_type:',content_type)
        print('file_extension:',file_extension)
        #filename生成
        # 获取当前时间
        now = datetime.datetime.now()
        # 格式化时间为 "YYYYMMDD_HHMMSS"
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        # 生成一个随机数
        # random_number = random.randint(0, 100)  # 生成一个随机数
        # 组合时间戳和随机数
        # filename = f"{timestamp}_see{data['seed']}"
        # filename = f"{key}_see{data['seed']}"

        que_filename=f"{timestamp}_queseed{questions['seed']}"
        ans_filename=f"{timestamp}_ansseed{answers['seed']}"
        

        # 保存文件
        # file_path = os.path.join(out_dir, f"{filename}.{file_extension}")
        que_path = os.path.join(out_dir, f"{que_filename}.{file_extension}")
        ans_path = os.path.join(out_dir, f"{ans_filename}.{file_extension}")
        

        # 保存文件
        with open(que_path, 'wb') as file:
            for chunk in res_questions.iter_content(chunk_size=8192):
            # for chunk in res_questions.iter_content(chunk_size=10):
            #     print('写入que')
                file.write(chunk)
        print(f"文件已保存为 {que_path}.")

        # 保存文件
        with open(ans_path, 'wb') as file:
            for chunk in res_answers.iter_content(chunk_size=8192):
            # for chunk in res_answers.iter_content(chunk_size=10):
            #     print('写入ans')
                file.write(chunk)  
        print(f"文件已保存为 {ans_path}.")

        result={
                  "instruction": "这是一个语音对话，回答为语音。",
                  "input": {
                    "type": "audio",
                    "content": que_path, #path/to/input_audio1.wav 路径
                    "format": "mp3",
                    "language": "zh",
                    "seed":questions['seed'],
                    "transcription": questions['text'],  #我今天真的很开心，因为我通过了考试。text内容
                    "emotion": gpt_gene['emotion'],   #情绪 开心
                    "gender": voices[str(questions['seed'])]['sex'],    #female
                    "age": voices[str(questions['seed'])]['age'],         #成人
                    "speed": "",       #正常
                    "volume": "",      #中等
                    "tone": ""         #愉快
                  },
                  "output": {
                    "type": "audio",
                    "content": ans_path,    #path/to/output_audio1.wav
                    "format": "mp3",
                    "language": "zh",
                    "seed":answers['seed'],
                    "transcription": answers['text'], #太好了，祝贺你！我们应该好好庆祝一下。
                    "emotion": gpt_gene['emotion'],       #兴奋
                    "gender": voices[str(answers['seed'])]['sex'],        #male
                    "age": voices[str(answers['seed'])]['age'],           #成人
                    "speed": "",         #正常
                    "volume": "",        #中等
                    "tone": ""           #明亮
                  }
                },
       
        # column_right.append(['√',None,gpt_gene,result])
        # column_all.append(['√',None,gpt_gene,result])

        return result,['√',None,gpt_gene,result]
    except IOError as e:
        # column_right.append(['√',None,gpt_gene,result])
        # column_all.append(['√',None,gpt_gene,result])
        # return ['×','数据保存失败',gpt_gene['emotion']]
        print('数据保存失败:',res_questions,res_answers)
        return None,['×','数据保存失败',gpt_gene,result]
  else:
      # print(f"请求失败，状态码: {response.status_code}")
      print('失败内容：',res_questions,res_answers)
      # return ['×','请求失败',gpt_gene['emotion']]
      return None,['×','请求失败',gpt_gene,None]

#list中随机选取两个元素
def random_pairs(lst, n):
    pairs = []
    for _ in range(n):
        # 从列表中随机选取两个元素
        pair = random.sample(lst, 2)
        # 使用zip将这两个元素组合成一对
        pairs.append(tuple(pair))
    return pairs


import os
import json
import pandas as pd
import shutil
import random
if __name__ == '__main__':
  data_dir='./text_out/Gpt4_emo' #文本数据集路径 
  out_dir='./voice_out/chatts_emo'  #输出的路径 
  sel_sheet={'sheet_right':'out_text'} #取某个sheet的列
  if os.path.exists(out_dir): shutil.rmtree(out_dir) #清空目录

  files = os.listdir(f'{data_dir}')    
  print('目录：',data_dir)
  print('文件：',files)
  for file in files :  #遍历文件
      if file.endswith('.xlsx'):
          
          column_all=[]
          column_right=[]  
          column_error=[]  
          print('文件路径：',f'{data_dir}/{file}')
          
          file_name,suffix = file.split(".")
          print(file_name,suffix)

          #清空jsonl文件
          file_outpath=f'{out_dir}/json/{file_name}.jsonl'
          dir_name=f'{out_dir}/json'
          if not os.path.isdir(dir_name):
              os.makedirs(dir_name)
          if os.path.exists(rf'{file_outpath}'):
              os.remove(rf'{file_outpath}')
          

          df_sheet_all = pd.read_excel(f'{data_dir}/{file}', sheet_name=None,index_col=None, header=0) 
          print('所有sheet：',pd.ExcelFile(f'{data_dir}/{file}').sheet_names)
          # print('df_sheet_all数据：\n',df_sheet_all)
          for sheet_name, sheet_data in df_sheet_all.items():
              # 将当前工作表数据添加到合并的DataFrame
              # merged_df = pd.concat([merged_df, sheet_data], ignore_index=True)

              # print('sheet_name名：',sheet_name)
              # # print('sheet_data数据：\n',sheet_data)
              # print('列名：',sheet_data.columns.ravel())

              if sheet_name in sel_sheet :
                  print('sheet_name名：',sheet_name)
                  # print('sheet_data数据：\n',sheet_data)
                  print('列名：',sheet_data.columns.ravel())                
                  flag=1
                  chunksize=3
                  for start in range(0, len(sheet_data), chunksize):
                          df_chunk = sheet_data.iloc[start:start + chunksize]
                          # process_chunk(df_chunk)
                          # 选多少条输出 
                          flag -=1
                          if  flag<0:  break

                          print('chunk数据：')
                          print(df_chunk)

                          print('列数据：')
                          print(df_chunk[sel_sheet[sheet_name]])                       

                          chunk_jsons=[json.loads(json_string) for json_string in df_chunk[sel_sheet[sheet_name]].tolist()]

                          # seeds_all=[1111,2222,]
                          seeds_all=[1111,2222,3333,4444,5555,6666,7777,8888,9999,]

                          seeds=random_pairs(seeds_all, len(chunk_jsons)) #随机选取两个音色作为问答

                          data_questions=[{
                                        "text": text['questions'],
                                        "seed": seed[0],
                                        "top_P": 0.7,
                                        "top_K": 20,
                                        "temperature": 0.3,
                                        "skip_refine_text": False,
                                        "refine_text_prompt": "[oral_2][laugh_0][break_6]"
                                      }  for seed,text in zip(seeds,chunk_jsons)]

                          data_answers=[{
                                        "text": text['answers'],
                                        "seed": seed[1],
                                        "top_P": 0.7,
                                        "top_K": 20,
                                        "temperature": 0.3,
                                        "skip_refine_text": False,
                                        "refine_text_prompt": "[oral_2][laugh_0][break_6]"
                                      }  for seed,text in zip(seeds,chunk_jsons)] #for text in chunk_jsons for seed in seeds]

                          gpt_genes=[text for text in chunk_jsons]

                          data_zips=zip(data_questions,data_answers)
                          # 转换为列表并打印结果
                          # print(list(data_zips))  # 输出: [(1, 'a'), (2, 'b'), (3, 'c')]
                          data_zips_list=list(data_zips)

                          for key,data_zip in enumerate(data_zips_list):
                            print('TTS输入数据：',data_zip)
                            print('GPT数据：',gpt_genes[key])
                            result,record=save_voice(data_zip,out_dir+'/voice',gpt_genes[key])
                            print('返回：',result,record)

                            print('结果：',type(result),result)

                            if result:
                              # # 将数据保存为JSONL文件，追加
                              with open(f'{file_outpath}', 'a') as f:
                                  for item in result:
                                      f.write(json.dumps(item,ensure_ascii=False) + '\n')
                              column_right.append(record)
                              column_all.append(record)
                            else:
                              column_error.append(record)
                              column_all.append(record)

              # ld    
          # print('所有数据：',column_all)
          # print('正常数据：',column_right)
          # print('异常数据：',column_error)
          print('*'*100) # None,['×','请求失败',gpt_gene,None]

          #写入表格
          df_all = pd.DataFrame(
              data=column_all,
              columns=['yes_or_no', 'message_err','gpt_text','voice_out']
          )

          df_right = pd.DataFrame(
              data=column_right,
              columns=['yes_or_no', 'message_err','gpt_text','voice_out']
          )

          df_error = pd.DataFrame(
              data=column_error,
              columns=['yes_or_no', 'message_err','gpt_text','voice_out']
          )

          # outdf_dir='./out/Gpt4_out'
          outdf_dir=out_dir+'/log/'
          
          # out_dir='./out_use/Gpt4_emo'
          # if os.path.exists(out_dir): shutil.rmtree(out_dir) #清空目录
          if not os.path.exists(outdf_dir): os.makedirs(outdf_dir) #目录不存在，创建

          #filename生成
          # 获取当前时间
          now = datetime.datetime.now()
          # 格式化时间为 "YYYYMMDD_HHMMSS"
          timestamp = now.strftime("%Y%m%d_%H%M%S")
          # filename = f"GPT_{timestamp}.xlsx"
          # filename = f"GPT_{timestamp}.xlsx"
          filename = f'{file_name}.xlsx'

          #写入多个sheet
          with pd.ExcelWriter(f'{outdf_dir}/{filename}') as writer:
              df_all.to_excel(writer, sheet_name='sheet_all',index=False)
              df_right.to_excel(writer, sheet_name='sheet_right',index=False)
              df_error.to_excel(writer, sheet_name='sheet_error',index=False)

          # df_sheet_all1 = pd.read_excel(f"{out_dir}/Descrip_xlsx.xlsx", sheet_name=None,index_col=0)
          df_sheet_all = pd.read_excel(f'{outdf_dir}/{filename}', sheet_name=None,index_col=False)
          print('Excel_sheet_all数据：\n',df_sheet_all)
          print('*'*100)   
          print('写入成功')         
                          # # 将数据保存为JSONL文件，追加
                          # with open(f'{file_outpath}', 'a') as f:
                          #     for item in chunk_json:
                          #         f.write(json.dumps(item,ensure_ascii=False) + '\n')

# 合并目录下的json
