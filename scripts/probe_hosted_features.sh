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
cat *.js > all.js
echo "ORIGIN $ORIGIN  bundle bytes: $(wc -c < all.js)"
for pat in grapheneMulticutSegments grapheneMergeSegments grapheneFindPath \
           multicut chunkedgraph middleauth graphene segmentation_with_graph \
           "/merge" "/split" "/undo_split" grapheneTime; do
  printf '  %-28s %s\n' "$pat" "$(grep -o -- "$pat" all.js | wc -l)"
done
rm -rf "$D"
