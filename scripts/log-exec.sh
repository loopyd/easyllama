#!/usr/bin/env bash
set +e
log=/proc/1/fd/2
printf 'DEBUG child start:' >"$log"
printf ' %q' "$@" >"$log"
printf '\n' >"$log"
"$@" > /proc/1/fd/1 2>"$log"
status=$?
printf 'ERROR child exit: status=%d command=%q\n' "$status" "$1" >"$log"
exit "$status"
