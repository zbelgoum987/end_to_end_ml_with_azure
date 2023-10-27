import os 
import argparse


def create_folder(folder_name):
    path = os.getcwd()
    parent = os.path.dirname(path)
    folder_path = os.path.join(parent,folder_name)
    os.mkdir(folder_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", dest='folder_name', type=str)
    args = parser.parse_args()
    return args


def main(args):
    folder_name = args.folder_name
    create_folder(folder_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)

