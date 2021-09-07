import os
import pandas as pd

# DATA_ROOT = "D:\daon_data\\nagoya_20210727_cutout_5ms"
DATA_ROOT = "/home/tomoko/daon/nagoya2021/nagoya_20210727_cutout_5ms/" # home linux
# DATA_ROOT = "/home/tozeki/daon/nagoya2021/nagoya_20210727_cutout_5ms/" # 大学PC


def list_of_dirs(data_root):
    print("inside list of dirs")
    print(data_root)
    print(os.listdir(data_root))
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    for curDir, dirs, files in os.walk(data_root):
        print("==========")
        print("Current Directory: " + curDir)
        print(f"Number of directories: {len(dirs)}")
        print(f"Number of files: {len(files)}")
        if len(files) != 0:
            for file in files:
                if file.endswith(".wav"):
                    dir_split = curDir.split("/")
                    print(f"length: {len(dir_split)}")
                    list1.append(curDir)
                    list2.append(file)
                    list3.append(dir_split[6])
                    list4.append(dir_split[7])
                    list5.append(dir_split[8])
                    temp = dir_split[-1]
                    if (temp == 'defect') or (temp == 'normal'):
                        list6.append(temp)
                    else:
                        list6.append('')

    # df = pd.DataFrame(list(zip(list1, list2, list3, list4, list5)),
    #                   columns=['path', 'file_name', 'place','hammer_type', 'label'])
    # print(df)

    data = {'path': list1, 'file_name': list2, 'place': list3, 'train_test': list4,
            'hammer_type': list5, 'label': list6}
    df = pd.DataFrame(data)
    df.to_csv('../annotations_home_linux.csv', encoding='utf-8')
    # df2 = df[df['place'] == 'crack_1']
    # df2.to_csv('crack_1.csv')


if __name__ == "__main__":
    print("Start")
    list_of_dirs(DATA_ROOT)