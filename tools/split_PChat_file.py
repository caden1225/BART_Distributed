#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 下午1:32
# @Author  : caden1225
# @File    : split_PChat_file.py
# @Description : 说明一下

import os
class SplitFile():
    """按行分割文件"""

    def __init__(self, file_name, line_count=3000000):
        """初始化要分割的源文件名和分割后的文件行数"""
        self.file_name = file_name
        self.line_count = line_count

    def split_file(self):
        if self.file_name and os.path.exists(self.file_name):
            try:
                with open(self.file_name) as f:  # 使用with读文件
                    temp_count = 0
                    temp_content = []
                    part_num = 1
                    for line in f:
                        if temp_count < self.line_count:
                            temp_count += 1
                        else:
                            self.write_file(part_num, temp_content)
                            part_num += 1
                            temp_count = 1
                            temp_content = []
                        temp_content.append(line)
                    else:  # 正常结束循环后将剩余的内容写入新文件中
                        self.write_file(part_num, temp_content)

            except IOError as err:
                print(err)
        else:
            print("%s is not a validate file" % self.file_name)

    def get_part_file_name(self, part_num):
        """"获取分割后的文件名称：在源文件相同目录下建立临时文件夹temp_part_file，然后将分割后的文件放到该路径下"""
        temp_path = os.path.dirname(self.file_name)  # 获取文件的路径（不含文件名）
        part_file_name = temp_path + "_splited"
        if not os.path.exists(part_file_name):  # 如果临时目录不存在则创建
            os.makedirs(part_file_name)
        part_file_name += os.sep + "PChat_split_" + str(part_num) + ".txt"
        return part_file_name

    def write_file(self, part_num, *line_content):
        """将按行分割后的内容写入相应的分割文件中"""
        part_file_name = self.get_part_file_name(part_num)
        print(line_content[0][0])
        try:
            with open(part_file_name, "w") as part_file:
                part_file.writelines(line_content[0])
            print('one done')
        except IOError as err:
            print(err)


if __name__ == "__main__":
    sf = SplitFile('/data/data_hub/BART_trainset/PChatchit/dist_data.txt')
    sf.split_file()