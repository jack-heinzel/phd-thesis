"""
Find BibTeX entries that already exist in the common-files BibTeX.
This is done by detecting duplicate DOIs or arXiv preprint IDs.
Adapted from 
https://git.ligo.org/publications/o4/cbc/gwtc-4-methods/-/blob/78ffacff77d37c955c3e2acc6d305c77e9f0d1e6/tests/check_local_bibtex.py
"""

import sys
import re


def read_dois_and_eprints(bibtex_path):
    with open(bibtex_path, 'r') as bibtex_file:
        lines = bibtex_file.readlines()
    dois = set()
    eprints = set()
    for line in lines:
        match = re.search(r'doi\s*=\s*["{](.+)["}]', line)
        if match:
            dois.add(match.group(1))
            continue
        match = re.search(r'eprint\s*=\s*["{](.+)["}]', line)
        if match:
            eprints.add(match.group(1))
            continue
    return dois, eprints


local_dois, local_eprints = read_dois_and_eprints('references.bib')
common_dois, common_eprints = read_dois_and_eprints(
    'gwtc-common-files/references/bibliography.bib'
)

dupe_dois = local_dois & common_dois
dupe_eprints = local_eprints & common_eprints

if dupe_dois:
    print('---- Duplicate DOIs found ----')
    for doi in sorted(dupe_dois):
        print(doi)

if dupe_eprints:
    print('---- Duplicate eprints found ----')
    for eprint in sorted(dupe_eprints):
        print(eprint)

if dupe_dois or dupe_eprints:
    # FAIL
    sys.exit(1)
