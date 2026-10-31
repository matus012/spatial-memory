#!/usr/bin/env bash
# Upload `git archive HEAD` to $PROJ/cache/tmp/repo_<rev> on PERUN (never touches $PROJ/repo, safe while jobs run).
# core.eol=lf: .gitattributes `* text=auto` + Windows git would export CRLF (frozen-config file sha differed on PERUN, 006 F6).
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
rev=$(git rev-parse --short HEAD); D=$PROJ/cache/tmp/repo_$rev
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD | ssh -o BatchMode=yes -o LogLevel=ERROR "${LOGIN_NODE:?set LOGIN_NODE}" "mkdir -p $D && tar -x -C $D && echo $rev > $D/REV"
echo "$D"
