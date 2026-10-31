#!/usr/bin/env bash
# Push `git archive HEAD` of this repo to PERUN $PROJ/repo. Run from the repo root on lpc.
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
H=${LOGIN_NODE:?set LOGIN_NODE}
rev=$(git rev-parse --short HEAD)
git -c core.autocrlf=false archive --format=tar HEAD | ssh -o BatchMode=yes "$H" "rm -rf $PROJ/repo.new && mkdir -p $PROJ/repo.new && tar -x -C $PROJ/repo.new && echo $rev > $PROJ/repo.new/REV && rm -rf $PROJ/repo && mv $PROJ/repo.new $PROJ/repo"
echo "pushed $rev -> $H:$PROJ/repo"
