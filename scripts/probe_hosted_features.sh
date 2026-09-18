#!/usr/bin/env bash
# Which graphene features does a hosted Neuroglancer build ship? Downloads the
# bundle and counts tool ids / endpoints in it. Usage:
#   bash scripts/probe_hosted_features.sh [origin]   (default: appspot demo)
set -e
ORIGIN=${1:-https://neuroglancer-demo.appspot.com}
D=$(mktemp -d)
cd "$D"
curl -sL "$ORIGIN/" -o index.html
for f in $(grep -o 'src="[^"]*\.js"' index.html | sed 's/src="//;s/"//'); do
  curl -sL "$ORIGIN/$f" -o "$f"
done
# Datasource modules are LAZY chunks: the entry scripts only carry the webpack
# runtime, whose chunk map names the rest. Fetch those too or every graphene
# string reads zero.
cat *.js > entry.js
for f in $(grep -oE '[0-9]+\.[a-f0-9]{16}\.js' entry.js | sort -u); do
  [ -f "$f" ] || curl -sfL "$ORIGIN/$f" -o "$f" || true
done
cat *.js > all.js
echo "ORIGIN $ORIGIN  chunks: $(ls *.js | wc -l)  bytes: $(wc -c < all.js)"
for pat in grapheneMulticutSegments grapheneMergeSegments grapheneFindPath \
           multicut chunkedgraph middleauth graphene segmentation_with_graph \
           "/merge" "/split" "/undo_split" grapheneTime; do
  printf '  %-28s %s\n' "$pat" "$(grep -o -- "$pat" all.js | wc -l)"
done
rm -rf "$D"
