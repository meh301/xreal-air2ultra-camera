import numpy as np, json, re, glob, os
OUTD='/tmp/traj_out'; os.makedirs(OUTD,exist_ok=True)
def gt_for(seq):
    for root in ['/mnt/processing/packs736','/mnt/processing/packs']:
        p='%s/%s/gt.tum'%(root,seq)
        if os.path.exists(p): return p
    return None
def ate(est,gta):
    gt_t,gt_p=gta[:,0],gta[:,1:4]; t,p=est[:,0],est[:,1:4]
    i=np.clip(np.searchsorted(gt_t,t),0,len(gt_t)-1); ok=np.abs(gt_t[i]-t)<0.03
    if ok.sum()<10: return None,None,None
    A,B=p[ok],gt_p[i][ok]; ca,cb=A.mean(0),B.mean(0)
    U,Sv,Vt=np.linalg.svd((A-ca).T@(B-cb)); D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); R=Vt.T@D@U.T
    full=(R@(p-ca).T).T+cb
    errf=np.full(len(p),np.nan); errf[ok]=np.linalg.norm(full[ok]-B,axis=1)
    return np.sqrt(np.nanmean(errf[ok]**2)), full, errf
def emit(files, seq, key):
    """pick the run whose ATE is CLOSEST TO THE MEDIAN (representative, not
    a divergent outlier), and export that trajectory."""
    gt=gt_for(seq)
    if not gt: return 0
    try: gta=np.loadtxt(gt)
    except: return 0
    scored=[]
    for f in files:
        try: est=np.loadtxt(f)
        except: continue
        if est.ndim!=2 or len(est)<10: continue
        a,full,errf=ate(est,gta)
        if a is not None: scored.append((a,est,full,errf))
    if not scored: return 0
    med=np.median([s[0] for s in scored])
    a,est,full,errf=min(scored,key=lambda s:abs(s[0]-med))  # representative run
    t=est[:,0]; ts=t-t[0]; step=max(1,len(est)//1500)
    r3=lambda arr:[[round(float(v),3) for v in row] for row in arr]
    gt_t,gt_p=gta[:,0],gta[:,1:4]; gstep=max(1,len(gt_p)//1500); gtd_t,gtd_p=gt_t[::gstep],gt_p[::gstep]; go=[]
    for k in range(len(gtd_p)):
        if k and gtd_t[k]-gtd_t[k-1]>2.0: go.append(None)
        go.append([round(float(v),3) for v in gtd_p[k]])
    ed=(errf*100.0)[::step]
    open('%s/%s_%s.json'%(OUTD,seq,key),'w').write(json.dumps({
        "est":r3(full[::step]),"gt":go,
        "err":[None if np.isnan(e) else round(float(e),1) for e in ed],
        "t":[round(float(x),2) for x in ts[::step]]}))
    return 1
def group(pat, rx):
    g={}
    for f in glob.glob(pat):
        m=re.match(rx,os.path.basename(f))
        if m: g.setdefault(m.group(1),[]).append(f)
    return g
n=0
for seq,fs in group('/mnt/processing/bc_rebench/*__bc_r*_map.tum', r'(.+?)__bc_r\d+_map').items(): n+=emit(fs,seq,'bc')
for seq,fs in group('/mnt/processing/bc_rebench/*__bc_r*_vio.tum', r'(.+?)__bc_r\d+_vio').items(): n+=emit(fs,seq,'bc-vio')
for seq,fs in group('/mnt/processing/absorb_ab/*__absorb_r*_map.tum', r'(.+?)__absorb_r\d+_map').items(): n+=emit(fs,seq,'absorb')
for seq,fs in group('/mnt/processing/vio_floor/*__vio_r*_vio.tum', r'(.+?)__vio_r\d+_vio').items(): n+=emit(fs,seq,'vio')
print('traj files (representative run):',n)
