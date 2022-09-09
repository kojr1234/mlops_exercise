#!/bin/bash

# Building packages and uploading them to a Gemfury repository

GEMFURY_URL=$GEMFURY_PUSH_URL

set -e

DIRS="$@"
BASE_DIR=$(pwd)
SETUP="setup.py"

warn() {
    echo "$@" 1>&2 # the 1>&2 stdout to bash script stderr
}

die() {
    warn "$@"
    exit 1
}

build() {
    DIR="${1/%\//}" #lists all directories from base_dir (mlops_exercise)
    echo "Checking directory $DIR"
    cd "$BASE_DIR/$DIR"
    [ ! -e $SETUP ] && warn "No $SETUP file, skipping" && return
    PACKAGE_NAME=$(python $SETUP --fullname)
    echo "Package $PACKAGE_NAME"
    python "$SETUP" sdist bdist_wheel || die "Building package $PACKAGE_NAME failed"
    for X in $(ls dist) # capture all files inside dist (ls command lists all files =p)
    do
        # does the command provided by gemfury curl -F package=@<file> https://TOKEN@push.fury.io/kojr1234/
        curl -F package=@"dist/$X" "$GEMFURY_URL" || die "Uploading package $PACKAGE_NAME failed on file dist/$X"
    done
}

if [ -n "$DIRS" ]; then
    for dir in $DIRS; do
        build $dir
    done
else
    ls -d */ | while read dir; do
        build $dir
    done
fi