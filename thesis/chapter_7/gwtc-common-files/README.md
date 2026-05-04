# GWTC: Common files

This is a simple repository to collect and common files used by more than one paper in the GWTC-X focus issue.

## Sub-modules
The repository should be added as a sub-module of the main paper. In-depth help with submodules can be found here: https://git-scm.com/book/en/v2/Git-Tools-Submodules. However, here are some simple instructions:
- To add the repository as a submodule, enter the repository of the paper and run
  ```
  $ git submodule add git@git.ligo.org:publications/o4/cbc/gwtc-common-files.git
  ```
- If you have pulled the paper repository, you will need to run
  ```
  $ git submodule init
  ```
  to initiate the submodule.
- To sync the paper repository (i.e. move the submodules to the commits specified in the main repo), run:
  ```
  $ git submodule update
  ```
- To fetch upstream changes into the paper repository run
  ```
  $ git submodule update --remote --recursive
  ```
  After this, you will see modifications in the paper repository. You need to add and commit these into the paper repository then push them to origin.
- Finally, changes to the common files repository should be made by a MR. For simplicity, we recommend these be made in a fork of the common files repository (or if you have edit access) in a branch of the main repository.

## References
The references directory contains the files required to generate the bibliography file `bibliography.bib` that is automatically checked into the repository.

### Quick start
- To add a single reference, find the inspire key (Author:UID) and then run
  ```
  $ cd references/
  $ python update_bibliography.py <KEY>
  ```
  replacing `<KEY>` with the inspire key. Now add all the modified files and commit, pushing to a branch and creating a merge request
- If the reference does not have a key or you need to manually edit the bib entry. You can copy the content to `base_bibliography.bib` and then run
  ```
  $ python update_bibliography.py
  ```
  Once again you should the add and commit the edited files.

### Details
The file is built as follows:
- The `base_bibliography.bib` file is for adding any references not available in Inspires
- The `bibliography.keys` file contains the Inspire key for each item to add to the bibliography. You can also use the Inspire unique identifier as well.
- The file `bibliography.bib` is built (from scratch) by running `python create_bibliography.py`, please watch for any error messages and fix these before checking the built file into the repository.
- The file `bibliography.bib` can be *updated* by adding new Inspire keys to the file `bibliograhpy.keys` and running `python update_bibliography.py` or by running `python update_bibliography.py texkey1 [texkey2 ...]`. In the latter form, keys are also added into the file `bibliography.keys`.

Unfortunately, Inspire texkeys are many-to-one in some cases, e.g., texkeys `Aasi:2012wd` and `VIRGO:2012oxz` both refer to DOI [10.1088/0264-9381/29/15/155002](https://doi.org/10.1088/0264-9381/29/15/155002).  The bibliography entry returned by Inspire and written to `bibliography.bib` has the texkey `VIRGO:2012oxz`.  The "valid" texkeys that appear in `bibliography.bib` are saved in the file `bibliography.keys.valid` and a mapping between the "obsolete" keys (`Aasi:2012wd` in this example) and the "valid" keys (`VIRGO:2012oxz` in this example) are written to the file `bibliography.keys.alias`. This file can be used to create a python dictionary as:
```python
>>> keymap = eval(open("bibliography.keys.alias", "r").read())
>>> keymap["Aasi:2012wd"]
'VIRGO:2012oxz'
```
In addition, a sed command file `updatetexkeys.sed` is created which can be used to update any obsolete texkeys in `.tex` files:
```sh
$ for file in *.tex; do cp $file $file.bak ; sed -f updatetexkeys.sed $file.bak > $file ; done
```

A second bibtex bibliography file, `compendia.bib`, contains the bibliography entries for the (unpublished) papers in the focus issue to allow cross-referencing between the papers.

## Automatic builds
The acronyms and macros provided by the common files are built as part of the CI. You can view the files from the master branch here:
1. Acronyms: [download](../builds/artifacts/master/raw/test_acros.pdf?job=test_acros_compile), [view in gitlab](../builds/artifacts/master/file/test_acros.pdf?job=test_acros_compile).
1. Macros: [download](../builds/artifacts/master/raw/test_macros.pdf?job=test_macros_compile), [view in gitlab](../builds/artifacts/master/file/test_macros.pdf?job=test_macros_compile).
