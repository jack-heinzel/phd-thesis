#!/bin/sh

# Script to create a tarball for manuscript submission to arXiv or ApJ
#
# This requires that the file <filename>.fls exists.
#
# - Per ApJ: No subdirectories are used in the archive.
# - Removal of comments are made.
#
# Warning:
# File names should only contain a-z A-Z 0-9 _ + - . , =

program="$0"
tmpdir="${TMPDIR:-"/tmp"}/tmp_$$"

usage()
{
    echo "usage: ${program} paper_filename final_filename" 1>&2
    exit ${1}
}

case ${1:-"unset"} in
    -h|--h|--help) usage 0;;
esac

if test $# -ne 2
then
    usage 1
fi

paper_filename="${1:?}"
final_filename="${2:?}"

# sanity check these filenames
fail=false
if `echo "${paper_filename}" | grep -q '[^a-zA-Z0-9_+.=-]'`
then
    echo "improper filename: ${file}" 1>&2
fi
if `echo "${final_filename}" | grep -q '[^a-zA-Z0-9_+.=-]'`
then
    echo "improper filename: ${file}" 1>&2
fi
if ${fail}
then
    echo 'filenames must only contain characters a-z A-Z 0-9 _ + - . , =' 1>&2
    exit 1
fi

# the filenames should not have extensions, so remove any
paper_filename="`echo "${paper_filename}" | cut -d. -f1`"
final_filename="`echo "${final_filename}" | cut -d. -f1`"

# make sure that the paper file list is present
if ! test -r "${paper_filename}.fls"
then
    echo "${paper_filename}.fls must be present: run \`latexmk ${paper_filename}.tex\`" 1>&2
    exit 1
fi

# sanity check the filenames so we don't encounter problems later
fail=false
save_IFS="${IFS}"
IFS="
"
for file in `sed -n '/INPUT [^/]/s/INPUT [./]*//p' ${paper_filename}.fls | sort | uniq`
do
    # allow / for now (they'll get replaced later)
    if `echo "${file}" | grep -q '[^a-zA-Z0-9_+.=/-]'`
    then
        echo "improper filename: ${file}" 1>&2
        fail=true
    fi
done
IFS="${save_IFS}"

if ${fail}
then
    echo 'filenames must only contain characters a-z A-Z 0-9 _ + - . , =' 1>&2
    exit 1
fi

# do everything within tmpdir
rm -rf "${tmpdir}"
mkdir "${tmpdir}"

# keep track of tex source files for later processing
texfiles=""

# get all input files from file list
# only include files that begin with INPUT
# only include files that have a relative path (not starting with /)
# if a file starts with "./" remove that
# make files unique
# copy all source files to tmpdir

for file in `sed -n '/INPUT [^/]/s/INPUT [./]*//p' ${paper_filename}.fls | sort | uniq`
do
    # incorporate relative path into filename by replacing / with __
    newfile="`echo "${file}" | sed -e 's%/%__%g'`"

    # identify all texfiles
    case "${file}" in
    *.tex)
        if test -n "${texfiles}"
        then
            texfiles="${texfiles} ${newfile}"
        else
            texfiles="${newfile}"
        fi
        ;;
    esac

    cp "${file}" "${tmpdir}/${newfile}"
done

# figure out what the common files directory is called
commonfilesdir="`sed -n '/\\\\newcommand{\\\\commonfiles}/p' "${paper_filename}.tex" | cut -d'{' -f3 | sed 's/}//'`"

# clean up texfiles
for file in ${texfiles}
do
    # remove comments and other markup for internal use
    # - lines "starting" with a %
    # - everything following an unescaped %
    # - % not preceded by a space
    # - colored text for reviewed / unreviewed / not applicable
    # - get rid of \FIXME \TODO and \NOTE
    cp "${tmpdir}/${file}" "${tmpdir}/${file}.tmp"
    sed \
        -e '/^[ ]*%/d' \
        -e 's/\(.*[^\]\)%.*/\1%/' \
        -e 's/ %.*//' \
        -e '/^\\providecolor{UNREVIEWED}/d' \
        -e '/^\\providecolor{REVIEWED}/d' \
        -e '/^\\providecolor{NOTAPPLICABLE}/d' \
        -e '/^\\newcommand{\\commonfiles}/d' \
        -e '/^\\bibliographystyle/d' \
        -e '/^\\bibliography/s/{.*}/{}/' \
        -e 's/\\color{UNREVIEWED}//g' \
        -e 's/\\color{REVIEWED}//g' \
        -e 's/\\color{NOTAPPLICABLE}//g' \
        -e '/^\\newcommand{\\FIXME}/d' \
        -e '/^\\newcommand{\\TODO}/d' \
        -e '/^\\newcommand{\\NOTE}/d' \
        -e 's/\\FIXME{[^}]*}//g' \
        -e 's/\\TODO{[^}]*}//g' \
        -e 's/\\NOTE{[^}]*}//g' \
        "${tmpdir}/${file}.tmp" > "${tmpdir}/${file}"
    rm -f "${tmpdir}/${file}.tmp"

    # explicitly replace \commonfiles
    cp "${tmpdir}/${file}" "${tmpdir}/${file}.tmp"
    sed -e 's%\\commonfiles/%'"${commonfilesdir}"'/%g' "${tmpdir}/${file}.tmp" > "${tmpdir}/${file}"
    rm -f "${tmpdir}/${file}.tmp"

    # replace / with __ in input and include commands
    # assume only one \input or \include command in a line and
    # replace all slashes with double underscores on that line
    # note: this also does \includegraphics

    cp "${tmpdir}/${file}" "${tmpdir}/${file}.tmp"
    sed \
        -e '/\\input.*{/s%/%__%g' \
        -e '/\\include.*{/s%/%__%g' \
        "${tmpdir}/${file}.tmp"  > "${tmpdir}/${file}"
    rm -f "${tmpdir}/${file}.tmp"
done

# rename paper filenames to final filenames
for file in ${tmpdir}/$paper_filename.*
do
    ext="`echo "${file}" | cut -d. -f2-`"
    mv "${tmpdir}/$paper_filename.$ext" "${tmpdir}/$final_filename.$ext"
done

# add a makefile
cat >"${tmpdir}/Makefile" <<EOF
all: ${final_filename}.pdf
${final_filename}.pdf:
	pdflatex ${final_filename}.tex
EOF

# create the archive and delete the temporary directory
(cd "${tmpdir}" && tar cf ${final_filename}.tar *)
mv "${tmpdir}/${final_filename}.tar" .
rm -rf "${tmpdir}"
