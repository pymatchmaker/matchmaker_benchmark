"""Replay frozen online-AMT MIDI through symbolic followers, using audio GT.

The transcription model is not executed. Runtime therefore covers the follower
only. MIDI event timestamps are preserved; there is no latency compensation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'matchmaker_eval'):
    sys.path.insert(0,str(p))
import matchmaker_eval.run_submission as rs
from matchmaker_eval.diagnostics import download,sha
from matchmaker_eval.submission import DECLARATION_KEY
from matchmaker import Matchmaker


def load_outputs(cache):
    pin=json.loads((ROOT/'data/amt.json').read_text())
    base=f'https://raw.githubusercontent.com/{pin["repository"]}/{pin["commit"]}/amt/'
    cache.mkdir(parents=True,exist_ok=True)
    p=cache/'manifest.json'
    if not p.exists() or sha(p)!=pin['manifest_sha256']:download(base+'manifest.json',p)
    if sha(p)!=pin['manifest_sha256']:raise ValueError('AMT manifest hash mismatch')
    manifest=json.loads(p.read_text());cases=manifest['cases']
    if len(cases)!=146 or len({c['piece_id'] for c in cases})!=146:raise ValueError('Invalid AMT test census')
    return pin,base,{c['piece_id']:c for c in cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['arzt','pthmm','SLT_OLTW'],default='arzt')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('amt-data'))
    a=p.parse_args();pin,base,cases=load_outputs(a.cache)
    original=rs.run_piece
    def run_piece(method,piece,input_type,index,run_dir,save_plots,estimated_bpm=False):
        case=cases[piece.piece_id];midi=a.cache/case['path']
        if not midi.exists() or sha(midi)!=case['sha256']:download(base+case['path'],midi)
        if sha(midi)!=case['sha256']:raise ValueError('AMT MIDI hash mismatch')
        def build(**kwargs):
            kwargs.update(input_type='midi',performance_file=str(midi),method=a.method)
            return Matchmaker(**kwargs)
        previous=rs.Matchmaker;rs.Matchmaker=build
        try:
            # Keep input_type=audio here: use the original recording's GT and
            # audio TR threshold, not transcribed-note times or MIDI tolerance.
            result=original(method,piece,'audio',index,run_dir,save_plots,False)
        finally:rs.Matchmaker=previous
        result.update(amt_data_commit=pin['commit'],amt_output_sha256=case['sha256'],
                      amt_front_end_executed=False,timing_scope='symbolic follower only',
                      symbolic_method=a.method)
        (run_dir/f'{index}.json').write_text(json.dumps(result,indent=2,default=float)+'\n')
        return result
    rs.run_piece=run_piece
    rs.builtin_metadata=lambda method,input_type:dict(name='AMT + '+a.method,input_type='audio',
        authors=['Matchmaker benchmark'],description='Frozen online AMT output replayed through a symbolic follower.',
        **{rs.ESTIMATED_BPM_KEY:False,DECLARATION_KEY:True},kind='reference')
    parts=rs.scoring_parts
    rs.scoring_parts=lambda:{**parts(),'amt_inputs':pin,'amt_symbolic_method':a.method,
        'amt_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result=rs.evaluate_submission(builtin=a.method,input_type='audio',fold='eval',
        shard=a.shard,num_shards=a.num_shards,run_dir=a.output)
    if result['n_failed']:raise SystemExit('Incomplete AMT replay')

if __name__=='__main__':main()
