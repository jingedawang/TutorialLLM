import os
import tqdm

def find_text_files(directory):
    """
    遍历指定目录及其子目录，找出所有名为 'text.txt' 的文件。

    :param directory: 要搜索的目录路径
    :return: 包含所有找到的 'text.txt' 文件路径的列表
    """
    text_files = []  # 存储找到的文件路径
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'text.txt':
                text_files.append(os.path.join(root, file))
    return text_files

# 使用示例
# 假设我们要搜索的目录是 '/path/to/directory'
directory_path = '../Classical-Modern'
found_files = find_text_files(directory_path)
print("找到的 'text.txt' 文件路径如下：")
for file_path in found_files:
    print(file_path)

with open('data.txt', 'w', encoding='utf-8') as file:
    for found_file in tqdm.tqdm(found_files):
        file.write(open(found_file, encoding='utf-8').read())
        file.write('\n')

