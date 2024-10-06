import json
import os
import re
import tqdm

def remove_text_in_parentheses(text):
    # 正则表达式匹配括号内的文本
    pattern = r'（.*?）'
    # 使用re.sub替换括号及内容为空字符串
    return re.sub(pattern, '', text)

def get_poetry_paths():
    file_list = []
    for i in range(0, 57000, 1000):
        file_list.append(f'../chinese-poetry/全唐诗/poet.tang.{i}.json')
    return file_list

file_list = get_poetry_paths()
data = open('data.txt', 'w', encoding='utf-8')
pretrain_data = open('pretrain_data.txt', 'w', encoding='utf-8')
finetune_data = open('finetune_data.txt', 'w', encoding='utf-8')
instruction = '<instruction>请用以下题目写一首诗</instruction>'
instruction_data_list = []
all_poems = []

for file in tqdm.tqdm(file_list):
    poems = json.load(open(file, 'r', encoding='utf-8'))
    all_poems.extend(poems)
    for poetry in poems:
        content = '\n'.join(poetry['paragraphs'])
        pretrain_data.write(remove_text_in_parentheses(f'{content}\n\n'))
        input = f'<input>{poetry['title']}</input>'
        response = f'<response>{content}</response>'
        instruction_data = {'instruction': instruction, 'input': input, 'response': response}
        instruction_data_list.append(instruction_data)

json.dump(all_poems, data, ensure_ascii=False)
pretrain_data.close()
json.dump(instruction_data_list, finetune_data, ensure_ascii=False)
finetune_data.close()
