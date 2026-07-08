#!/bin/zsh
cd "$(dirname "$0")/.."
for E in 01 08 14 32 50 6e 8c 9f; do
  rm -f frames/*.raw
  ./capture > /dev/null 2>&1 &
  CAP=$!
  sleep 1.0
  ./uvc_ctl set 1 0x02 3 04 >/dev/null 2>&1
  ./uvc_ctl set 1 0x04 3 ${E}000000 >/dev/null 2>&1
  wait $CAP
  python3 - "$E" <<'EOF'
import numpy as np, sys
E=int(sys.argv[1],16)
ims=[np.fromfile(f"frames/frame_{i}.raw",dtype=np.uint8).reshape(482,640)[:480].astype(int) for i in range(95,119)]
a=np.stack(ims)
hd=np.abs(np.diff(a,axis=2)).mean(); vd=np.abs(np.diff(a,axis=1)).mean()
print(f"exp={E:3d} ({E*0.1:.1f}ms)  mean={a.mean():6.1f}  Hdiff={hd:5.2f}  Vdiff={vd:5.2f}  aniso={vd/max(hd,0.1):.2f}  sat%={100*(a>250).mean():.1f}")
EOF
done
