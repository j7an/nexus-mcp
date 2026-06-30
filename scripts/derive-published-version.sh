#!/usr/bin/env bash
# derive-published-version.sh - derive and verify the version being published.
#
# Caller-copy reference: this script is a canonical reference for package
# release workflows in caller repos. It has no local inline workflow consumer in
# shared-workflows and is intentionally excluded from check-inline-sync.sh
# INLINE_PAIRS.
#
# Usage:
#   scripts/derive-published-version.sh <dist-dir> <tag>
#
# Output:
#   normalized artifact version on stdout
#
# Exit:
#   0 - exactly one wheel, exactly one sdist, and tag tail equals wheel version
#   1 - artifact count error or tag/artifact mismatch
#   2 - malformed invocation
#
# Bash 3.2 compatible.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <dist-dir> <tag>" >&2
  exit 2
fi

dist_dir="$1"
tag="$2"

if [ ! -d "$dist_dir" ]; then
  echo "dist directory does not exist: $dist_dir" >&2
  exit 1
fi

count_matches() {
  find "$1" -maxdepth 1 -type f -name "$2" | wc -l | tr -d '[:space:]'
}

first_match() {
  find "$1" -maxdepth 1 -type f -name "$2" | sed -n '1p'
}

wheel_count=$(count_matches "$dist_dir" '*.whl')
sdist_count=$(count_matches "$dist_dir" '*.tar.gz')

if [ "$wheel_count" -ne 1 ]; then
  echo "expected exactly one wheel in $dist_dir, found $wheel_count" >&2
  exit 1
fi

if [ "$sdist_count" -ne 1 ]; then
  echo "expected exactly one sdist in $dist_dir, found $sdist_count" >&2
  exit 1
fi

wheel_path=$(first_match "$dist_dir" '*.whl')
wheel_base="${wheel_path##*/}"

artifact_remainder="${wheel_base#*-}"
artifact_version="${artifact_remainder%%-*}"

if [ -z "$artifact_version" ] || [ "$artifact_version" = "$wheel_base" ]; then
  echo "could not derive version from wheel filename: $wheel_base" >&2
  exit 1
fi

tag_tail="${tag##*/}"
tag_tail="${tag_tail#v}"

if [ "$tag_tail" != "$artifact_version" ]; then
  echo "Tag tail '$tag_tail' does not equal built version '$artifact_version'." >&2
  echo "Tag the canonical normalized version, e.g. 'v${artifact_version}', or update project metadata." >&2
  exit 1
fi

printf '%s\n' "$artifact_version"
