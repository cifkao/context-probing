#!/bin/bash
rm -rf dist/
yarn build
git --git-dir=.git-dist --work-tree=dist checkout www-dist
git --git-dir=.git-dist --work-tree=dist add .
git --git-dir=.git-dist --work-tree=dist commit -a -m "Deployment $(date)" --allow-empty
git --git-dir=.git-dist --work-tree=dist push origin www-dist
