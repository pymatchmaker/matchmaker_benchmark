"""Re-evaluate archived CI paths after correcting repeat annotations."""
import argparse, hashlib, json, os, shutil
from pathlib import Path
import numpy as np
from matchmaker_eval.verify_tracking import check_tracking
from matchmaker_eval.beat_units import musical_beat_factors

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--data-revision',required=True);a=p.parse_args()
 manifest=json.loads((a.data/'diagnostics/manifest.json').read_text())
 case=next(c for c in manifest['cases'] if c['id']=='kv279/section_repetition')
 gt_source=a.data/'diagnostics'/case['paths']['match']
 assert sha(gt_source)==manifest['files'][case['paths']['match']]['sha256']
 gt=np.loadtxt(gt_source,skiprows=1,ndmin=2)
 assert np.all(np.diff(gt[:,0])>=0) and np.any(np.diff(gt[:,1])<0)
 corrected=[]
 for src in sorted(a.source.iterdir()):
  if not src.is_dir(): continue
  metrics=json.loads((src/'metrics.json').read_text())
  assert metrics['config'] in ('global_skip-midi-on','global_skip-midi-off')
  dest=a.output/'raw'/src.name;shutil.copytree(src,dest,dirs_exist_ok=True)
  for r in metrics['pieces']:
   if r['case_id']!=case['id']:continue
   folder=dest/r['directory'];oldgt=folder/'gt_0.tsv';wp=folder/'wp_0.tsv';wp_hash=sha(wp)
   audit=dict(original_gt_sha256=sha(oldgt),corrected_gt_sha256=sha(gt_source),original_mean_b=r['mean_b'],source_run=metrics['github_run_id'],source_commit=metrics['environment']['benchmark_commit'],correction_run=os.getenv('GITHUB_RUN_ID'),data_revision=a.data_revision)
   shutil.copy2(gt_source,oldgt)
   w=np.loadtxt(wp,skiprows=1,ndmin=2);w=w[np.argsort(w[:,0],kind='stable')];ix=np.searchsorted(w[:,0],gt[:,0],side='right')-1
   valid=ix>=0;err=abs(gt[valid,1]-w[ix[valid],1]);meter=json.loads((folder/'meter_0.json').read_text())
   tr=check_tracking(w,gt,segment_duration=30,threshold=.1,beat_factors=musical_beat_factors(gt[:,1],meter))
   r.update(mean_b=float(err.mean()),tracked=bool(tr['tracked']),max_deviation=float(tr['max_deviation']),n_events=int(valid.sum()),n_annotated_onsets=len(gt),annotation_correction=audit)
   r['input_hashes']['match']=sha(gt_source)
   # Original library metrics are retained with an explicit stale-annotation label.
   r['metrics_original_annotations']=r.pop('metrics');r['metrics']={k:v for k,v in r['metrics_original_annotations'].items() if k=='rtf'}
   r['metrics'].update(mean_b=r['mean_b'])
   (folder/'result.json').write_text(json.dumps(r,indent=2)+'\n')
   assert sha(wp)==wp_hash
   corrected.append(dict(config=metrics['config'],mean_b=r['mean_b'],tracked=r['tracked'],max_deviation=r['max_deviation'],wp_sha256=wp_hash,**audit))
  (dest/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 assert len(corrected)==2
 shutil.copy2(a.data/'diagnostics/manifest.json',a.output/'data_manifest.json')
 (a.output/'annotation_correction.json').write_text(json.dumps(dict(complete=True,corrections=corrected),indent=2)+'\n')
 print(json.dumps(corrected,indent=2))
if __name__=='__main__':main()
