import sys

if sys.version_info.major < 3:
    raise Exception('This script requires Python 3 or later.')

if sys.version_info.minor < 7:
    # must make sure that dicts preserve insertion order
    from collections import OrderedDict as dict

import os
import getpass
import json
import requests
try:
    from itertools import pairwise
except ImportError:
    # itertools.pairwise added in version 3.10
    def pairwise(iterable):
        # pairwise('ABCDEFG') → AB BC CD DE EF FG
        iterator = iter(iterable)
        a = next(iterator, None)
        for b in iterator:
            yield a, b
            a = b


bibfile = 'bibliography.bib'
base_bibfile = 'base_' + bibfile
keyfile_ads = 'bibliography.ads.keys'
keyfile_doi = 'bibliography.doi.keys'
keyfile_valid = 'bibliography.keys.valid'
keyfile_alias = 'bibliography.keys.alias'
sedfile = 'updatetexkeys.sed'


# ADS bearer token
token = None
token_help_msg = '''
* An ADS API token is required. Get your API token from
* https://ui.adsabs.harvard.edu/user/settings/token
* and store your token in the file ~/.ads/token
* or save it in environment variable ADS_API_TOKEN
* or enter it when prompted.
'''


def parse_bibtex_entry(entry_string):

    '''
    Parses a string containing a bibtex entry.

    Returns the tuple (entry_key, entry_type, entry_fields)
    entry_key: string or None
    entry_type: string or None
    entry_fields: dict or None
    '''

    # technically a bibtex entry line must start with '@'
    # but sometimes leading whitespace is erroneously present
    entry_string = entry_string.strip()

    if not entry_string[0] == '@':
        # not a bibtex entry
        return None, None, None

    entry_type, _, entry_data = entry_string[1:].partition('{')
    entry_type = entry_type.strip().upper()

    # find matching '}' and strip it
    try:
        entry_data = entry_data[:entry_data.rindex('}')].rstrip()
    except ValueError:
        # there was no matching '}'
        raise ValueError('invalid entry data')

    # some bibtex entries are special commands

    if entry_type == 'STRING':
	# the field is the abbreviation and the value is the string
        abbreviation, _, string = entry_data.partition('=')
        abbreviation = abbreviation.strip()
        string = string.strip()
        return None, 'STRING', dict([(abbreviation, string)])

    if entry_type == 'PREAMBLE':
        return None, 'PREAMBLE', entry_data

    # this is a regular bibtex citation entry

    entry_key, _, entry_data = entry_data.partition(',')
    entry_key = entry_key.strip()
    entry_data = entry_data.lstrip()

    entry_fields = dict()

    while '=' in entry_data:

        field, _, entry_data = entry_data.partition('=')
        field = field.strip().lower()
        entry_data = entry_data.strip()

        value = ''
        quote_depth = 0
        brace_depth = 0
        escape = False

        for n, c in enumerate(entry_data):

            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"' and brace_depth == 0:
                quote_depth = 0 if quote_depth else 1
            elif c == '{':
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1
            elif c == ',' and quote_depth == 0 and brace_depth == 0:
                break

            value += c

        value = value.strip()
        entry_data = entry_data[n+1:].strip()
        entry_fields[field] = value

    return entry_key, entry_type, entry_fields


def parse_bibtex_lines(lines):

    '''
    Parses lines of a bibtex file into a dict representing bibtex entries.

    Returns: tuple (preamble, bibentry)
    preamble: list - preamble bibtex entries
    bibentry: dict - bibliographic bibtex entries
    '''

    # strip out comment lines (nonstandard)
    lines = [line for line in lines if not line.lstrip().startswith('#') or not line.lstrip().startswith('%')]

    # technically a bibtex entry line must start with '@'
    # but sometimes leading whitespace is erroneously present
    partitions = [idx for idx, line in enumerate(lines) if line.lstrip().startswith('@')]
    partitions += [len(lines)]
    entry_strings = ['\n'.join(lines[i:j]) for i, j in pairwise(partitions)]
    preamble = list()  # entries that are commands
    bibentry = dict()  # entries that are references
     
    for entry_string in entry_strings:
        entry_key, entry_type, entry_data = parse_bibtex_entry(entry_string)
        if entry_key is not None:
            bibentry[entry_key] = (entry_type, entry_data)
        else:
            preamble.append((entry_type, entry_data))

    return preamble, bibentry


def get_bibtex_from_doi(dois):

    '''
    Gets bibtex entry string for dois.
    '''

    # if this is just one doi, make it a list
    if isinstance(dois, str):
        dois = [dois]

    bibentry = dict()

    for doi in dois:

        url = os.path.join('https://doi.org', doi)
        headers = {'Accept': 'application/x-bibtex'}
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'

        # parse response
        entry_key, entry_type, entry_data = parse_bibtex_entry(response.text)

        # replace whatever key was returned with doi
        bibentry[doi] = (entry_type, entry_data)

    return bibentry


def get_bibtex_from_ads(bibcodes, token):

    # if this is just one bibcode, make it a list
    if isinstance(bibcodes, str):
        bibcodes = [bibcodes]

    headers = {'Authorization': f'Bearer {token}'}
    payload = {'bibcode': list(bibcodes)}
    serialized_payload = json.dumps(payload)
    results = requests.post('https://api.adsabs.harvard.edu/v1/export/bibtex', headers=headers, data=serialized_payload)
    results.encoding = 'utf-8'

    # parse results into a bibentry (preamble should be empty)
    _, bibentry = parse_bibtex_lines(results.json()['export'].split('\n'))

    return bibentry



if __name__ == "__main__":


    # Keys are either ADS bibcodes or DOIs

    bibcodes = set()
    dois = set()


    # Behavior of this program depends on the program name:
    # - create_program builds bibliography from key files
    # - update_program updates bibliography from key arguments

    create_program = "create_bibliography"
    update_program = "update_bibliography"
    program = sys.argv.pop(0)

    if create_program in program:

        usage = f"usage: {program} [-h | --help]"

        if "-h" in sys.argv or "--help" in sys.argv:
            print(usage, file=sys.stderr)
            sys.exit()

        if sys.argv:
            print(f"error: unrecognized arguments {' '.join(sys.argv)}", file=sys.stderr)
            sys.exit(usage)

    elif update_program in program:

        usage = f"usage: {program} [-h | --help] [texkey ...]"

        if "-h" in sys.argv or "--help" in sys.argv:
            print(usage, file=sys.stderr)
            sys.exit()

        # New keys to add from arguments

        for key in sys.argv:

            # figure out if this is a doi or a bibcode
            if len(key) == 19 and '/' not in key:
                # this is an ADS bibcode
                bibcodes.add(key)
            elif key.startswith('10.') and '/' in key:
                # this is a DOI
                dois.add(key)
            else:
                exit(f'Error: key "{key}" is not an ADS bibcode or a DOI')

    else:

        # Program must be create_bibliography or update_bibliography
        exit(f"Unrecognized program name: {program}")


    # Get ADS API bearer token

    if token is None:
        # try environment variable ADS_API_TOKEN first
        token = os.environ.get('ADS_API_TOKEN')

    if token is None:
        # try file ~/.ads/token
        try:
            with open(os.path.expanduser('~/.ads/token'), 'r') as file:
                token = file.read().rstrip()
        except FileNotFoundError:
            print(token_help_msg, file=sys.stderr)

    if token is None:
        # prompt user for token
        try:
            token = getpass.getpass('Enter your ADS API token (Ctrl+C to exit): ')
        except KeyboardInterrupt:
            exit('\nProgram Terminated')


    if token is None or len(token) == 0:
        exit(token_help_msg)


    # Read base bibliography

    try:
        with open(base_bibfile, 'r') as file:
            preamble, base_bib = parse_bibtex_lines(file) 
    except FileNotFoundError:
        pass


    # Read ADS bibcodes from ADS key file; preserve comments

    keyline_ads = dict()

    with open(keyfile_ads, 'r') as file:
        for line in file:
            line = line.rstrip()
            key = line.split('#')[0].rstrip()
            keyline_ads[key] = line

    bibcodes.update(keyline_ads)


    # Read DOIs from DOI key file; preserve comments

    keyline_doi = dict()

    with open(keyfile_doi, 'r') as file:
        for line in file:
            line = line.rstrip()
            key = line.split('#')[0].rstrip()
            keyline_doi[key] = line

    dois.update(keyline_doi)


    # Get bibentries from ADS bibcodes

    ads_bib = get_bibtex_from_ads(bibcodes, token)


    # Look for any updated ADS bibcodes

    alias_old = set(bibcodes) - set(ads_bib.keys())
    alias_new = set(ads_bib.keys()) - set(bibcodes)
    alias = dict()

    if len(alias_old) != len(alias_new):
        print('Error finding all bibcodes', file=sys.stderr)
        print('Set of bibcodes that need to be updated:', alias_old, file=sys.stderr)
        print('Set of updated bibcodes:', alias_new, file=sys.stderr)
        exit()
 

    for a in list(alias_old):

	# normally we expect bibcodes to update only when going from a preprint
	# to a published version
        if 'arXiv' not in a:
            exit(f'Error: bibcode {a} was not returned')

        # seek corresponding entry in alias_new
        for b in list(alias_new):

            entry = ads_bib[b][1]

            if 'eprint' not in entry:
                continue

	    # just check if last four digits of eprint number are in the
	    # original bibcode following the 'YYYYarXiv' prefix
            # note: this might give a false match...

            eprint = ads_bib[b][1]['eprint'].strip('"{}')
            if eprint[-4:] in a[9:-1]:
                alias[a] = b
                alias_old.remove(a)
                alias_new.remove(b)
                break


    if len(alias_old) != 0 or len(alias_new) != 0:
        print('Error finding all bibcodes', file=sys.stderr)
        print('Set of bibcodes that need to be updated:', alias_old, file=sys.stderr)
        print('Set of updated bibcodes:', alias_new, file=sys.stderr)
        exit()


    # See if any of the DOIs are already in the ADS bib

    for bibcode, (_, entry) in ads_bib.items():
        if 'doi' in entry:
            doi = entry['doi'].strip('"{}')
            if doi in dois:
                alias[doi] = bibcode
                dois.remove(doi)
                break


    # Get bibentries from DOIs

    doi_bib = get_bibtex_from_doi(dois)

    for bibcode, (_, entry) in ads_bib.items():

        authors = entry['author'][1:-1]  # strip off {} or ""
        authors = authors.split(' and ')

        # Note: some of the ADS entries have a collaboration as a first author.
        # Get rid of first author(s) if contains 'Collaboration'.
        if 'Collaboration' in authors[0]:
            if all(['collaboration' in author.lower() or 'others' in author.lower() for author in authors]):
                print(f"Entry {bibcode} has only Collaborations as authors: leaving as is")
            else:
                print(f"Entry {bibcode} has a 'Collaboration' as the first author.")
                print("  Removing all Colab. from the author list")

                # get rid of all initial authors who are collaborations
                while 'Collaboration' in authors[0]:
                    authors.pop(0)

                if len(authors) > 0 and 'others' not in authors[0].lower():
                    entry['author'] = '{' + ' and '.join(authors) + '}'

        # Handle special cases where trailing commas break the bibtex
        if len(authors) == 1:
            entry["author"] = '{' + authors[0].rstrip(',') + '}'

    # All bibentries 

    bib = doi_bib | ads_bib

    # add in entries from base bibliography; override if needed
    for key in base_bib.keys() & bib.keys():
        print(f'OVERRIDING {key} from {base_bibfile}', file=sys.stderr)

    bib |= base_bib


    # Write updated keyfiles, keeping original line comments

    with open(keyfile_ads, 'w') as file:
        for key in sorted(bibcodes):
            print(keyline_ads[key] if key in keyline_ads else key, file=file)

    with open(keyfile_doi, 'w') as file:
        for key in sorted(dois):
            print(keyline_doi[key] if key in keyline_doi else key, file=file)


    # Write valid keyfile

    with open(keyfile_valid, 'w') as file:
        print('\n'.join(k for k in sorted(bib)), file=file)


    # Write outdated key aliases

    with open(keyfile_alias, 'w') as file:
        print(repr(dict(sorted(alias.items()))), file=file)


    # Write sed command script to update keys

    with open(sedfile, 'w') as file:
        # note: use '%' rather than '/' since DOIs contain '/'
        print('\n'.join(f's%{k}%{v}%g' for k, v in sorted(alias.items())), file=file)


    # Write updated bibliography

    entries = list()

    for entry_type, entry_data in preamble:

        if entry_type == 'PREAMBLE':
            entry = '@PREAMBLE{' + entry_data + '}'

        elif entry_type == 'STRING':

            if len(entry_data) != 1:
                raise ValueError('string dict must have one item')

            abbreviation, string = list(entry_data.items())[0]
            entry = '@STRING{' + f'{abbreviation} = {string}' + '}'

        else:
            raise ValueError('unrecognized entry type')

        entries.append(entry)


    for entry_key, (entry_type, entry_data) in sorted(bib.items()):

        entry = ['@' + entry_type + '{' + entry_key]

        for field, value in entry_data.items():
            entry.append(f'    {field} = {value}')
    
        entry = ',\n'.join(entry)
        entry += '\n}'
        entries.append(entry)


    with open(bibfile, 'w') as file:
        print('\n\n'.join(entries), file=file)
