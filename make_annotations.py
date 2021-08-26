import os
import pandas as pd

# DATA_ROOT = "D:\\nagoya2021\\nagoya_20210727_cutout_5ms"
DATA_ROOT = "/home/tozeki/daon/nagoya2021/nagoya_20210727_cutout_5ms"


def list_of_dirs(data_root):
    print("inside list of dirs")
    print(data_root)
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    for curDir, dirs, files in os.walk(data_root):
        print("==========")
        print("Current Directory: " + curDir)
        print(f"Number of directories: {len(dirs)}")
        print(f"Number of files: {len(files)}")
        if len(files) != 0:
            for file in files:
                if file.endswith(".wav"):
                    list1.append(curDir)
                    list2.append(file)
                    list3.append(curDir.split("/")[-4])
                    list4.append(curDir.split("/")[-2])
                    list5.append(curDir.split("/")[-1])

    df = pd.DataFrame(list(zip(list1, list2, list3, list4, list5)),
                      columns=['path', 'file_name', 'place','hammer_type', 'label'])
    print(df)
    df.to_csv('../annotations.csv', encoding='utf-8')
    print(df[df['place'] == 'crack_1'])


if __name__ == "__main__":
    print("Start")
    list_of_dirs(DATA_ROOT)