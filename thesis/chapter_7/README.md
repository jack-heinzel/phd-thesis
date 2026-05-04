# O4a Astrophysical Distributions

## Description
This paper describes estimates of the astrophysical rates and population properties of CBCs detected through the O4a run.

## Automatic builds
The latest version of the paper: [download](../builds/artifacts/main/raw/paper.pdf?job=publish), [view in gitlab](../builds/artifacts/main/file/paper.pdf?job=publish).

## DCC
The latest DCC version of the paper can be found [here](https://dcc.ligo.org/P2400004).

## Team Meetings
Due to shifting schedules, the O4a Astro Dist team meets at differing times each quarter. During the June-August 2024 quarter, there are two meetings: 
* 7AM GMT Tuesdays 
* 8PM GMT Thursdays 

Agendas and minutes are [here](https://git.ligo.org/publications/o4/cbc/o4a-astrodist/-/wikis/home/telecons).
You can also reach us in our Mattermost [channel](https://chat.ligo.org/ligo/channels/o4a-astro-dist)

### Cloning the repository
Note: the build of this paper relies on a git submodule https://git.ligo.org/publications/o4/cbc/gwtc-common-files. To clone the repository and set this up, please run
```
$ git clone --recurse-submodules git@git.ligo.org:publications/o4/cbc/o4a-astrodist.git
```
If you have already cloned the repository without the `--recurse-submodules` flag, you can run
```
$ git submodule init
$ git submodule update
```
from inside this repository to add the submodule. The submodule is treated as an independent git repository. Help with using the submodule can be found here: https://git-scm.com/book/en/v2/Git-Tools-Submodules

### Fetching changes to the submodule
When you call `git pull`, the submodules are not (by default) automatically updated. Instead, you need to run
```
$ git submodule update --remote
```
This will pull the changes into your local directory.
