import sys
from github import Github
import os

# GITHUB_TOKEN
# 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
# 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
# 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
# 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

token = os.getenv('GITHUB_TOKEN', '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')
github = Github(token)

#prometheus/client_golang - 83
#go-redis/redis - 80
#authelia/authelia - 313
#WorldDbs/specs-actors - 217
#gofiber/fiber - 295
#DominicBreuker/pspy - 250
#jonas747/yagpdb - 441
#grpc-ecosystem/grpc-gateway - 208

#moby/moby - 2283
#golang/tools - 1336
#golang/go - 4194
#mitchellh/packer - 339
#GoogleCloudPlatform/kubernetes - 2913
#hashicorp/terraform - 783
#golang/net - 551

#unknwon/the-way-to-go_ZH_CN - 392
#gohugoio/hugo - 721
#gin-gonic/gin - 84
#fatedier/frp - 160
#v2ray/v2ray-core - 543
#prometheus/prometheu -352 (16500)

#gogs/gogs - 173
#etcd-io/etcd - 996
#caddyserver/caddy - 188
#ethereum/go-ethereum - 1088
#minio/minio - 694
#pingcap/tidb - 1141
#rclone/rclone - 476
#istio/istio - 1563
#beego/beego - 483
#go-gitea/gitea - 1699

api_name = "go-redis/redis"
repo = github.get_repo(api_name)
api_name = api_name.replace("/", "--")
contents = repo.get_contents("")
directory = None

f = open(api_name, "a")
h = open(api_name + "-output-sequence", "a")
e = open(api_name + "-errors", "a")

goTokensDict = {'func': '1', '{': '2', '}': '3', 'package': '4',
                '(': '5', ')': '6', '/': '7', '"': '8',
                '...': '9', 'var': '10', '!': '11', '/*': '12',
                '*/': '13', 'int': '14', '[': '15', ']': '16',
                'for': '17', 'len': '18', ':': '19', '+': '20',
                '': '21', ',': '22', ':=': '23', ';': '24',
                '.': '25', '-': '26', '=': '27', 'if': '28',
                '>': '29', 'select': '30', '||': '31', 'import': '32', '*': '33',
                'interface': '34', '<': '35', 'default': '36',
                'break': '37', '^': '38', '&': '39', 'case': '40',
                'defer': '41', 'Go': '42', 'map': '43', 'struct': '44',
                'chan': '45', 'else': '46', 'goto': '47', 'switch': '48',
                'const': '49', 'fallthrough': '50', 'range': '51', 'Type': '52',
                'continue': '53', 'return': '54', 'byte': '55', 'float32': '56',
                '%': '57', '--': '58', '++': '59', '==': '60', '!=': '61',
                'Sizeof': '62', 'int32': '63', 'true': '64', 'false': '65',
                'bool': '66', 'delete': '67', 'type': '68', 'string': '69', 'nil': '70',
                'make': '71', 'float64': '72', 'go': '73', 'int64': '74',
                '&&': '75', '<<': '76', '>>': '77', '&^': '78',
                '+=': '79', '-=': '80', '*=': '81', '/=': '82',
                '%=': '83', '&=': '84', '|=': '85', '^=': '86',
                '<<=': '87', '>>=': '88', '&^=': '89', '<-': '90',
                '<=': '91', '>=': '92', '|': '93', '_': '94'
                }

goCommonFunctionsDict = {'len': '95', 'Println': '96', 'Scan': '97'}

# 101 in total(100)

#200 - 98 - literals
#300 - 99 - digit
#400 - 100 - function name
#500 - 101 - variable
#600 - type - not used

def parseFile(fileName, programSequence):
    # don't write to files is there is error in parsing
    writeToFiles = True

    tokensSequence = []
    progSeqOutput = ""
    programLength = len(programSequence)
    print(programLength)
    element = 0
    numberOfLines = 0

    while element < programLength:
        char = programSequence[element]

        # whitespace
        if char.isspace() and char != "\n":
            progSeqOutput = progSeqOutput + " "
            element += 1

        # tab
        elif char == "\t":
            progSeqOutput = progSeqOutput + "\t"
            element += 1

        # end of line
        elif char == "\n":
            numberOfLines += 1
            progSeqOutput = progSeqOutput + "\n" + str(numberOfLines)
            element += 1

        # double quotes
        elif char == "\"":
            elem = element + 1
            while elem < programLength:
                char1 = programSequence[elem]
                if programSequence[elem] == "\\" and programSequence[elem + 1] == "\"" and programSequence[elem + 2] is not " " and programSequence[elem + 2] is not ":" and programSequence[elem + 2] is not "," and programSequence[elem + 2] is not ")":
                    elem += 2
                elif programSequence[elem] == "\"":
                    element = elem + 1
                    tokensSequence.append("98")
                    progSeqOutput = progSeqOutput + " LITERAL"
                    break
                else:
                    elem += 1

        # single quote - must be separated from double quote
        elif char == "\'":
            elem = element + 1
            while elem < programLength:
                char1 = programSequence[elem]
                if programSequence[elem] == "\\" and programSequence[elem + 1] == "\'" and programSequence[elem + 2] is not " " and programSequence[elem + 2] is not ":" and programSequence[elem + 2] is not "," and programSequence[elem + 2] is not ")":
                    elem += 2
                elif programSequence[elem] == "\'":
                    element = elem + 1
                    tokensSequence.append("98")
                    progSeqOutput = progSeqOutput + " LITERAL"
                    break
                else:
                    elem += 1

        #backquotes
        elif char == "`":
            elem = element + 1
            while elem < programLength:
                if programSequence[elem] == "\\" and programSequence[elem + 1] == "\`" and programSequence[elem + 2] is not " " and programSequence[elem + 2] is not ":" and programSequence[elem + 2] is not "," and programSequence[elem + 2] is not ")":
                    elem += 2
                elif programSequence[elem] == "`":
                    element = elem + 1
                    tokensSequence.append("98")
                    progSeqOutput = progSeqOutput + " LITERAL"
                    break
                else:
                    elem += 1

        # comment //
        elif char == "/" and programSequence[element + 1] == "/":
            elem = element + 2
            while elem < programLength:
                # for elem in range(element + 2, programLength):
                if (programSequence[elem] == "\n"):
                    element = elem + 1
                    break
                # ako je zadnji red - ok
                elif (elem == programLength - 1):
                    element = elem
                    break
                elem = elem + 1

        # comment /*
        elif char == "/" and programSequence[element + 1] == "*":
            elem = element + 2
            while elem < programLength:
                if (programSequence[elem] == "*" and programSequence[elem + 1] == "/"):
                    element = elem + 2
                    break
                elem += 1

        # 2 i 3 operators
        elif char == "." or char == ">" or char == "<" or char == "&" or char == "^" or char == "|" or char == "%" or char == "/" or char == "*" or char == "-" or char == "+" or char == "!" or char == "=" or char == ":":
            # <<=
            if char == "<" and programSequence[element + 1] == "<" and programSequence[element + 2] == "=":
                tokensSequence.append(goTokensDict["<<="])
                progSeqOutput = progSeqOutput + " <<="
                element += 3

            # >>=
            elif char == ">" and programSequence[element + 1] == ">" and programSequence[element + 2] == "=":
                tokensSequence.append(goTokensDict[">>="])
                progSeqOutput = progSeqOutput + " >>="
                element += 3

            # &^=
            elif char == "&" and programSequence[element + 1] == "^" and programSequence[element + 2] == "=":
                tokensSequence.append(goTokensDict["&^="])
                progSeqOutput = progSeqOutput + " &^="
                element += 3

            # ...
            elif char == "." and programSequence[element + 1] == "." and programSequence[element + 2] == ".":
                tokensSequence.append(goTokensDict["..."])
                progSeqOutput = progSeqOutput + " ..."
                element += 3

            # ||
            elif char == "|" and programSequence[element + 1] == "|":
                tokensSequence.append(goTokensDict["||"])
                progSeqOutput = progSeqOutput + " ||"
                element += 2

            #:=
            elif char == ":" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict[":="])
                progSeqOutput = progSeqOutput + " :="
                element += 2

            # ++
            elif char == "+" and programSequence[element + 1] == "+":
                tokensSequence.append(goTokensDict["++"])
                progSeqOutput = progSeqOutput + " ++"
                element += 2

            # --
            elif char == "-" and programSequence[element + 1] == "-":
                tokensSequence.append(goTokensDict["--"])
                progSeqOutput = progSeqOutput + " --"
                element += 2

            # ==
            elif char == "=" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["=="])
                progSeqOutput = progSeqOutput + " =="
                element += 2

            # !=
            elif char == "!" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["!="])
                progSeqOutput = progSeqOutput + " !="
                element += 2

            # &&
            elif char == "&" and programSequence[element + 1] == "&":
                tokensSequence.append(goTokensDict["&&"])
                progSeqOutput = progSeqOutput + " &&"
                element += 2

            # <<
            elif char == "<" and programSequence[element + 1] == "<":
                tokensSequence.append(goTokensDict["<<"])
                progSeqOutput = progSeqOutput + " <<"
                element += 2

            # >>
            elif char == ">" and programSequence[element + 1] == ">":
                tokensSequence.append(goTokensDict[">>"])
                progSeqOutput = progSeqOutput + " >>"
                element += 2

            # &^
            elif char == "&" and programSequence[element + 1] == "^":
                tokensSequence.append(goTokensDict["&^"])
                progSeqOutput = progSeqOutput + " &^"
                element += 2

            # +=
            elif char == "+" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["+="])
                progSeqOutput = progSeqOutput + " +="
                element += 2

            # -=
            elif char == "-" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["-="])
                progSeqOutput = progSeqOutput + " -="
                element += 2

            # *=
            elif char == "*" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["*="])
                progSeqOutput = progSeqOutput + " *="
                element += 2

            # /=
            elif char == "/" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["/="])
                progSeqOutput = progSeqOutput + " /="
                element += 2

            # %=
            elif char == "%" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["%="])
                progSeqOutput = progSeqOutput + " %="
                element += 2

            # &=
            elif char == "&" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["&="])
                progSeqOutput = progSeqOutput + " &="
                element += 2

            # |=
            elif char == "|" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["|="])
                progSeqOutput = progSeqOutput + " |="
                element += 2

            # ^=
            elif char == "^" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["^="])
                progSeqOutput = progSeqOutput + " ^="
                element += 2

            # <-
            elif char == "<" and programSequence[element + 1] == "-":
                tokensSequence.append(goTokensDict["<-"])
                progSeqOutput = progSeqOutput + " <-"
                element += 2

            # <=
            elif char == "<" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict["<="])
                progSeqOutput = progSeqOutput + " <="
                element += 2

            # >=
            elif char == ">" and programSequence[element + 1] == "=":
                tokensSequence.append(goTokensDict[">="])
                progSeqOutput = progSeqOutput + " >="
                element += 2

            elif char == "." or char == "-" or char == "+" or char == "*" or char == "/" or char == "=" or char == "<" or char == ">" \
                    or char == "&" or char == "%" or char == "!" or char == "|" or char == "^" or char == ":":
                tokensSequence.append(goTokensDict[char])
                progSeqOutput = progSeqOutput + " " + char
                element += 1

        # operators braces etc.
        elif char == "(" or char == ")" or char == "[" or char == "]" or char == "{" or char == "}" or char == "," or char == ";":
            tokensSequence.append(goTokensDict[char])
            progSeqOutput = progSeqOutput + " " + char
            element += 1

        # number
        elif char.isdigit() or char == ".":
            number = char
            elem = element + 1
            while elem < programLength:
                nextChar = programSequence[elem]
                if nextChar.isspace() or nextChar == ")" or (
                        nextChar == "+" and programSequence[elem - 1] != 'E' and programSequence[elem - 1] != 'e' and
                        programSequence[elem - 1] != 'p' and programSequence[
                            elem - 1] != 'P') or nextChar == "*" or nextChar == "/" or (
                        nextChar == "-" and programSequence[elem - 1] != 'E' and programSequence[elem - 1] != 'e' and
                        programSequence[elem - 1] != 'p' and programSequence[elem - 1] != 'P') \
                        or nextChar == "<" or nextChar == ">" or nextChar == "!" or nextChar == "=" or nextChar == ";" or nextChar == "," or nextChar == "[" or nextChar == "]" \
                        or nextChar == "}" or nextChar == "{" or nextChar == "|" or nextChar == "%" or nextChar == "&" or nextChar == ":" \
                        or (nextChar == "/" and programSequence[elem + 1] == "n"):
                    element = elem
                    tokensSequence.append("99")
                    progSeqOutput = progSeqOutput + " NUMBER"
                    break
                else:
                    number = number + nextChar
                    elem += 1
        elif char == "_" and (programSequence[element + 1] == " " or programSequence[element + 1] == ","):
            element += 1
            progSeqOutput = progSeqOutput + " " + char
            tokensSequence.append(goTokensDict[char])

        # word
        elif char.isalpha() or char == "_":
            isPreviousTokenType = False
            word = char
            elem = element + 1
            while elem < programLength:
                nextChar = programSequence[elem]
                if nextChar == "(":
                    # function name
                    if word == "if" or word == "func":
                        tokensSequence.append(goTokensDict[word])
                        progSeqOutput = progSeqOutput + " " + word
                        element = elem
                    elif word in goCommonFunctionsDict.keys():
                        tokensSequence.append(goCommonFunctionsDict[word])
                        progSeqOutput = progSeqOutput + " " + word
                        element = elem
                    else:
                        progSeqOutput = progSeqOutput + " FUNCTION_NAME"
                        tokensSequence.append("100")
                        element = elem
                    break
                elif nextChar.isspace() or nextChar == ")" or nextChar == "," or nextChar == "[" or nextChar == "+" or nextChar == "-" or nextChar == "*" or nextChar == "/" \
                        or nextChar == "<" or nextChar == ">" or nextChar == "=" or nextChar == ":" or nextChar == ";" or nextChar == "}" or nextChar == "{" or nextChar == "]" or nextChar == "|" \
                        or nextChar == "%" or nextChar == "!" or nextChar == "&" or nextChar == "\n":
                    tokensSequenceLastElem = None
                    tokensSequence1BeforeLastElem = None
                    tokensSequence2BeforeLastElem = None
                    tokensSequence3BeforeLastElem = None
                    # keyword
                    if word in goTokensDict.keys():
                        if word == "type":
                            isPreviousTokenType = True
                        progSeqOutput = progSeqOutput + " " + word
                        tokensSequence.append(goTokensDict[word])
                        element = elem
                        break
                    # variable ili type(npr. Circle)
                    else:
                        if len(tokensSequence) > 3:
                            tokensSequenceLastElem = tokensSequence[-1]
                            tokensSequence1BeforeLastElem = tokensSequence[-2]
                            tokensSequence2BeforeLastElem = tokensSequence[-3]
                            tokensSequence3BeforeLastElem = tokensSequence[-4]
                        if isPreviousTokenType == True:
                            progSeqOutput = progSeqOutput + " VARIABLE"
                            tokensSequence.append("101")
                            isPreviousTokenType = False
                        elif len(
                                tokensSequence) > 3 and tokensSequenceLastElem == "101" and tokensSequence1BeforeLastElem == "5" and tokensSequence2BeforeLastElem == "1":
                            progSeqOutput = progSeqOutput + " VARIABLE"
                            tokensSequence.append("101")
                        elif len(
                                tokensSequence) > 3 and tokensSequenceLastElem == "101" and tokensSequence1BeforeLastElem == "5" and tokensSequence2BeforeLastElem == "100" and tokensSequence3BeforeLastElem == "1":
                            progSeqOutput = progSeqOutput + " VARIABLE"
                            tokensSequence.append("101")
                        else:
                            progSeqOutput = progSeqOutput + " VARIABLE"
                            tokensSequence.append("101")
                        element = elem
                        break
                # before . - object(variable)
                elif programSequence[elem] == "." and programSequence[elem + 1] == "." and programSequence[elem + 2] == "." :
                    element = elem + 3
                    # variable
                    progSeqOutput = progSeqOutput + " VARIABLE"
                    tokensSequence.append("101")
                    progSeqOutput = progSeqOutput + "..."
                    tokensSequence.append(goTokensDict["..."])
                    break
                elif programSequence[elem] == ".":
                    element = elem + 1
                    # variable
                    progSeqOutput = progSeqOutput + " VARIABLE"
                    tokensSequence.append("101")
                    progSeqOutput = progSeqOutput + " ."
                    tokensSequence.append(goTokensDict["."])
                    break
                else:
                    word = word + nextChar
                    elem += 1
                    if (elem == programLength):
                        progSeqOutput = progSeqOutput + " VARIABLE"
                        tokensSequence.append("101")
                        element = elem
                        break

        else:
            print(directory)
            print(programSequence[element - 9])
            print(programSequence[element - 8])
            print(programSequence[element - 7])
            print(programSequence[element - 6])
            print(programSequence[element - 5])
            print(programSequence[element - 4])
            print(programSequence[element - 3])
            print(programSequence[element - 2])
            print(programSequence[element - 1])
            print(char + "\n")

            e.write("\n" + directory + "\n")
            e.write("\nNumber of go files: " + str(numberOfGoFiles) + "\n")
            e.write("\n" + programSequence[element - 9])
            e.write("\n" + programSequence[element - 8])
            e.write("\n" + programSequence[element - 7])
            e.write("\n" + programSequence[element - 6])
            e.write("\n" + programSequence[element - 5])
            e.write("\n" + programSequence[element - 4])
            e.write("\n" + programSequence[element - 3])
            e.write("\n" + programSequence[element - 2])
            e.write("\n" + programSequence[element - 1])
            writeToFiles = False

            break

    if (writeToFiles):
        f.write("\n" + directory + "\n")
        f.write("\nNumber of go files: " + str(numberOfGoFiles) + "\n")
        for token in tokensSequence:
            f.write(token + " ")

        h.write("\n\n" + directory)
        h.write("\nProgram length" + str(programLength) + " = " + str(element))
        h.write("\nNumber of go files: " + str(numberOfGoFiles))
        h.write("\n" + progSeqOutput + "\n")

numberOfGoFiles = 0

filePassed = False
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        if file_content.name.endswith('.go'):
            numberOfGoFiles = numberOfGoFiles + 1
            directory = file_content.path
            print(numberOfGoFiles)
            print(file_content.path)
            if (file_content.name == "json.go" or file_content.path == "api/prometheus/v1/example_test.go" or file_content.path == "moderation/commands.go" or "/models/" in file_content.path or file_content.path == "builder/dockerfile/internals.go" or file_content.path == "integration-cli/daemon/daemon.go" or file_content.path == "godoc/static/makestatic.go" or file_content.path == "godoc/static/static.go" or file_content.path == "internal/lsp/tests/tests.go" or file_content.path == "test/eof1.go"
                    or file_content.path == "src/html/escape_test.go" or file_content.path == "fix/fixer_amazon_private_ip.go" or file_content.path == "fix/fixer_amazon_private_ip.go" or file_content.path == "internal/initwd/load_config.go" or file_content.path == "internal/terraform/transform_import_state.go" or file_content.path == "internal/command/arguments/apply.go" or file_content.path == "webdav/internal/xml/marshal_test.go" or file_content.path == "pkg/roachpb/api.pb.go"
                    or "test/fixedbugs" in file_content.path or file_content.path == "src/encoding/xml/marshal_test.go" or file_content.path == "src/net/http/cookie.go" or file_content.path == "test/complit1.go" or file_content.path == "commands/import_jekyll.go" or file_content.path == "assets/frps/statik/statik.go" or file_content.path == "discovery/triton/triton_test.go" or file_content.path == "internal/tool/path.go" or file_content.path == "internal/assets/conf/conf_gen.go"
                    or file_content.path == "replacer.go" or file_content.path == "signer/fourbyte/4byte.go" or file_content.path == "executor/point_get_test.go" or file_content.path == "fstest/fstests/fstests.go"):
                continue
            programSequence = file_content.decoded_content.decode()
            parseFile(file_content.name, programSequence)

print("Number of files in repo: ", numberOfGoFiles)