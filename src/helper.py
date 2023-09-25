import os


def build_user_dictionary():
    output = list()
    for dictionary in os.listdir("dataset/ChineseDictionary"):
        with open(os.path.join("dataset/ChineseDictionary", dictionary)) as input_file:
            output += input_file.readlines()
    output = list(set(output))

    with open("dataset/dictionary.txt", "w") as output_file:
        output_file.writelines(output)
