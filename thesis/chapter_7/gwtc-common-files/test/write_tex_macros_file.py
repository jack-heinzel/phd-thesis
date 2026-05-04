import glob
import re


def contains_optional_argument(latex_text):
    # Regex pattern to check if [n] is present in \newcommand definition, where n is any integer
    pattern = r'\\newcommand{\\\w+}\[(\d+)\]'
    return bool(re.search(pattern, latex_text))


def read_macros(filename):
    macros = []
    with open(filename, "r") as f:
        for latex_text in f:

            # Regex pattern to capture \newcommand definitions
            pattern = r'\\newcommand{\\(\w+)}'
            match = re.search(pattern, latex_text)
            if match and contains_optional_argument(latex_text) is False:
                macros.append(match.group(1))
    return macros


read_macros
files = glob.glob("macros/*")
macros = {}
for file in files:
    macros[file] = read_macros(file)

test_file_name = "test_macros.tex"
test_file = open(test_file_name, "w+")

print(r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{savetrees}
\usepackage{xcolor}
\usepackage{xspace}
\usepackage{longtable}
\begin{document}""",
      file=test_file)

for file in macros:
    print(f"\\input{{{file}}}", file=test_file)

print(r"""
\begin{longtable}{c|c|c}
file & MACRO & OUTPUT \\ \hline""",
      file=test_file)

backslash_char = "\\"
for file, macro_set in macros.items():
    for macro in macro_set:
        if macro not in ["soft"]:
            print(f"{file.replace('_', chr(92)+'_')} & {macro} & {backslash_char}{macro}{backslash_char}{backslash_char}", file=test_file)

print(r"""
\end{longtable}

\end{document}""",
      file=test_file)
