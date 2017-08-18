
import os

def combine_contents_of_dirs(dir_list, file_suffix, save_path):
    fw = open(save_path, 'w')

    for dir in dir_list:
        contents = os.listdir(dir)
        for content in contents:
            if content.endswith(file_suffix):
                full_path = dir + '/' + content
                fr = open(full_path)
                lines = fr.readlines()
                for line in lines:
                    fw.write(line + '\n')

    fw.close()


combine_contents_of_dirs(['aclImdb/train/pos', 'aclImdb/test/pos'], 'txt', 'temp/pos.txt')
combine_contents_of_dirs(['aclImdb/train/neg', 'aclImdb/test/neg'], 'txt', 'temp/neg.txt')
combine_contents_of_dirs(['aclImdb/train/unsup'], 'txt', 'temp/unsup.txt')


