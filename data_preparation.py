import os
import sys

training_files_directory = "C:\\Users\\Mrak\\PycharmProjects\\rnn_syntax_error_correction\\training"
filesInDirectory = os.listdir(training_files_directory)
trainingSourceFile = open("_trainingSourceFile-200", "a")
sequences = []

def parseFile(file):
    lines = file.splitlines()

    for line in lines:
        #if not empty line and line starts with a digit then:
        if line.strip() and line[0].isdigit():
            line = line.strip()
            # no lines have empty space on end
            numberOfTokensInLine = line.count(' ') + 1

            lineTokens = line.split()

            count = 0
            tokens80 = ""
            for token in lineTokens:
                count += 1
                if count == 81:
                    trainingSourceFile.write(tokens80 + "\n")
                    break
                tokens80 = tokens80 +  token + " "

for file in filesInDirectory:
    file = open(training_files_directory + "\\" + file, "r")

    parseFile(file.read())


