import os

from langchain_community.document_loaders import UnstructuredMarkdownLoader, UnstructuredFileLoader, UnstructuredPDFLoader
from tqdm import tqdm


class ReadFiles:
    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    # 获取文件路径函数
    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list


    # 加载文件函数
    def get_text(self):
        # docs 存放加载之后的纯文本对象
        docs = []
        # 遍历所有目标文件
        for one_file in tqdm(self.file_list):
            file_type = one_file.split('.')[-1]
            if file_type == 'md':
                loader = UnstructuredMarkdownLoader(one_file)
            elif file_type == 'txt':
                loader = UnstructuredFileLoader(one_file)
            elif file_type == 'pdf':
                loader = UnstructuredPDFLoader(one_file)
            else:
                # 如果是不符合条件的文件，直接跳过
                continue
            docs.extend(loader.load())
        return docs


