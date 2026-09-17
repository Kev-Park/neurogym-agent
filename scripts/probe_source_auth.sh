#!/usr/bin/env bash
# Which of the FlyWire sources in a viewer link need credentials, and does the
# cluster's cave-secret satisfy the graphene one. Prints status codes only —
# never the token. Usage: bash scripts/probe_source_auth.sh
S=~/.cloudvolume/secrets/cave-secret.json
TOK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['token'])" "$S")
echo "PROBE $(date +%H:%M)"
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$@"; }
echo "EM gs://flywire_em/aligned/v1 anon:      $(code https://storage.googleapis.com/flywire_em/aligned/v1/info)"
echo "EM microns-seunglab public anon:         $(code https://storage.googleapis.com/microns-seunglab/drosophila_v0/alignment/image_rechunked/info)"
G=https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public
echo "graphene flywire_public info anon:       $(code $G/info)"
echo "graphene flywire_public info bearer:     $(code -H "Authorization: Bearer $TOK" $G/info)"
echo "graphene auth_info:                      $(curl -s --max-time 20 https://prodv1.flywire-daf.com/auth_info)"
A=$(curl -s --max-time 20 https://prodv1.flywire-daf.com/auth_info | python3 -c "import json,sys;print(json.load(sys.stdin)['login_url'])")
echo "authorize bearer (app_urls?):            $(curl -s --max-time 20 -H "Authorization: Bearer $TOK" "$A/api/v1/authorize" | head -c 300)"
echo "user/me bearer:                          $(curl -s --max-time 20 -H "Authorization: Bearer $TOK" "$A/api/v1/user/me" | python3 -c "import json,sys;d=json.load(sys.stdin);print({k:d.get(k) for k in ('id','admin','pi','datasets','permissions_v2') if k in d})" 2>/dev/null || echo n/a)"
echo "old root 720575940625112137 mesh bearer: $(code -H "Authorization: Bearer $TOK" "$G/../../../meshing/1.0/flywire_public/manifest/720575940625112137:0?verify=1")"
