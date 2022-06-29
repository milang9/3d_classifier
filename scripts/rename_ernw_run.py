import os
import sys

def main():
    #if len(sys.argv) != 4:
    #    sys.exit("Number of inputs doesn't match the required amount!")

    ori_dirs = sys.argv[1]
    #target_dir = sys.argv[2]
    #name_list = sys.argv[3]

    for directory in os.scandir(ori_dirs):
        if os.path.isdir(directory.path):
            print(directory.path)



if __name__=="__main__":
    main()
