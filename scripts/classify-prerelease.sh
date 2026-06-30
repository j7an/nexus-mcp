#!/usr/bin/env bash
# classify-prerelease.sh - classify a normalized PyPI-publishable version.
#
# Caller-copy reference: this script is a canonical reference for package
# release workflows in caller repos. It has no local inline workflow consumer in
# shared-workflows and is intentionally excluded from check-inline-sync.sh
# INLINE_PAIRS.
#
# Input contract: exactly one normalized, PyPI-publishable version. Local
# versions containing '+label' are unsupported because PyPI publish artifacts
# should not use local versions.
#
# Output:
#   true  - version is a pre-release or dev release
#   false - version is stable or post-release only
#
# Exit:
#   0 - classified successfully
#   2 - malformed or unsupported input
#
# Bash 3.2 compatible.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <normalized-pypi-version>" >&2
  exit 2
fi

version="$1"

if [ -z "$version" ]; then
  echo "version must not be empty" >&2
  exit 2
fi

case "$version" in
  *+*)
    echo "local versions are unsupported input: $version" >&2
    exit 2
    ;;
esac

if printf '%s\n' "$version" | grep -Eq '(a|b|rc)[0-9]|\.dev[0-9]'; then
  echo "true"
else
  echo "false"
fi
