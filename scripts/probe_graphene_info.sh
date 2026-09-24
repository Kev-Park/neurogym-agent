#!/usr/bin/env bash
# Where does the flywire_public graphene datastack keep its chunks and meshes,
# and can they be read anonymously / with the CAVE token?  Status codes only.
S=~/.cloudvolume/secrets/cave-secret.json
TOK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['token'])" "$S")
G=https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public
echo "PROBE $(date +%H:%M)"
curl -s --max-time 20 -H "Authorization: Bearer $TOK" $G/info > /tmp/gi.json
python3 - <<'PY'
import json; i=json.load(open('/tmp/gi.json'))
for k in ('data_dir','mesh','mesh_metadata','graph','app','sharded_mesh','skeletons','chunks_start_at_voxel_offset'):
    if k in i: print(f"  {k}: {i[k]}")
print("  scales:", len(i.get('scales',[])), "mip0 res", i['scales'][0]['resolution'], "chunk", i['scales'][0]['chunk_sizes'])
PY
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$@"; }
DD=$(python3 -c "import json;print(json.load(open('/tmp/gi.json'))['data_dir'])")
MD=$(python3 -c "import json;i=json.load(open('/tmp/gi.json'));print(i.get('mesh',''))")
tohttp() { echo "$1" | sed -e 's#^gs://#https://storage.googleapis.com/#'; }
echo "data_dir info anon:   $(code "$(tohttp $DD)/info")"
echo "data_dir info bearer: $(code -H "Authorization: Bearer $TOK" "$(tohttp $DD)/info")"
echo "mesh info anon:       $(code "$(tohttp $DD)/$MD/info")   (absolute: $(code "$(tohttp $MD)/info"))"
echo "old-root manifest via server (bearer): $(code -H "Authorization: Bearer $TOK" "https://prodv1.flywire-daf.com/meshing/1.0/flywire_public/manifest/720575940625112137:0?verify=1")"
# CloudVolume view of the same
cd /scratch/kp0374/neurogym-agent && uv run --no-sync python - <<'PY' 2>&1 | tail -6
from cloudvolume import CloudVolume
cv = CloudVolume('graphene://https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public', mip=0, agglomerate=False, use_https=True, progress=False)
print("CV cloudpath:", cv.meta.cloudpath, "| res", cv.resolution.tolist(), "| bounds", cv.bounds.to_list())
import numpy as np
c = [int(215675*4/cv.resolution[0]), int(61215*4/cv.resolution[1]), 3828]
cut = cv[c[0]-32:c[0]+32, c[1]-32:c[1]+32, c[2]-2:c[2]+2]
u = np.unique(cut); print("CV cutout ok:", cut.shape, "unique svids:", len(u), "max", int(u.max()))
roots = cv.get_roots(u[u!=0][:50]); print("get_roots ok:", len(roots), "distinct roots:", len(set(roots.tolist())), "example current root:", int(roots[0]))
m = cv.mesh.get(720575940625112137, lod=2 if False else 0) if False else cv.mesh.get(720575940625112137)
mm = list(m.values())[0]; print("old-root mesh ok: verts", mm.vertices.shape[0], "faces", mm.faces.shape[0])
PY
