import re


def read_acros(filename):
    acros = []
    with open(filename, "r") as f:
        for latex_text in f:
            # Regex pattern to capture \newcommand definitions
            match = re.search(r"\\acrodef{(.*?)}", latex_text)
            if match:
                acros.append(match.group(1))
    return acros


files = ["acronyms.tex"]
acros = {}
for file in files:
    acros[file] = read_acros(file)

test_file_name = "test_acros.tex"
test_file = open(test_file_name, "w+")

print(r"""\documentclass{article}
\usepackage{savetrees}
\usepackage{xcolor}
\usepackage{xspace}
\usepackage{acronym}
\usepackage{longtable}""",
      file=test_file)

for file in files:
    print(f"\\input{{{file}}}", file=test_file)

print(r"\begin{document}", file=test_file)

print(r"""
\begin{longtable}{c|c|c}
file & ACRONYM & OUTPUT \\ \hline""",
      file=test_file)

backslash_char = "\\"
for file, acro_set in acros.items():
    for acro in acro_set:
        print(f"{file.replace('_', chr(92)+'_')} & {acro} & {backslash_char}ac{{{acro}}}{backslash_char}{backslash_char}", file=test_file)

print(r"""
\end{longtable}

\end{document}""",
      file=test_file)
