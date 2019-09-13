#!/bin/bash

# --------------------------------------------------------------------
# Run this within a local respository directory to tag the latest
# commit, both locally and in the remote repository.
#
# For example:
#
#>> ./tag 1.1.2
# ++ git log --pretty=format:%H -n 1
# + latest_checksum=2cfa6a85c6d3bcfde3863f3e7417d0099782e198
# + git tag -a v1.1.2 -m 'version 1.1.2' 2cfa6a85c6d3bcfde3863f3e7417d0099782e198
# + git tag
# v0.9.7
# v0.9.7.1
# v0.9.8
# v0.9.8.1
# v0.9.8.3
# v0.9.9
# v0.9.9.1
# v1.0
# v1.0.1
# v1.0.2
# v1.0.3
# v1.0.4
# v1.1
# v1.1.1
# v1.1.2
# + git push origin v1.1.2
# Password for 'https://cfpython@bitbucket.org': 
# Counting objects: 1, done.
# Writing objects: 100% (1/1), 164 bytes | 0 bytes/s, done.
# Total 1 (delta 0), reused 0 (delta 0)
# To https://cfpython@bitbucket.org/cfpython/cf-python.git
#  * [new tag]         v1.1.2 -> v1.1.2
# + set +x
# --------------------------------------------------------------------

if [[ ! $1 ]] ; then 
  echo "No version \$1 (e.g. 2.0.1)"
  exit 1
fi

version=$1
major_version=$(echo $version | cut -c 1)

current_branch=`git rev-parse --abbrev-ref HEAD`

if [[ $major_version == 1 && $current_branch != v1 ]] ; then
  echo "Can only tag version $version in branch v1, not branch $current_branch"
  exit 2
fi

if [[ $major_version == 2 && $current_branch != master ]] ; then
  echo "Can only tag version $version in branch master, not branch $current_branch"
  echo "Can only tag branch 'master'"
  exit 2
fi

echo "New tag: v$version"
echo

echo "Existing Tags:"
git tag
echo

x=`git tag | grep v$version`
if [[ $? -eq 0 ]] ; then 
  echo "ERROR: Tag v$version already exists"
  exit 1
fi

set -x

# Find checksum of latest commit
latest_checksum=`git log --pretty=format:'%H' -n 1`

# Create tag in local repository
git tag -a v$version -m "version $version" $latest_checksum

# Look at at all my tags
git tag

# Push tag to remote repository      
git push origin v$version

set +x
