"""Run the frozen paired LSE/chroma Dixon diagnostic through the common evaluator."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/'matchmaker_eval'):
    sys.path.insert(0,str(p))
import matchmaker_eval.run_submission as rs
from matchmaker import Matchmaker


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--processor',choices=['lse','chroma'],required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    def build(**kwargs):
        return Matchmaker(**kwargs,processor=a.processor)
    metadata = rs.builtin_metadata
    def described(method, input_type):
        return {**metadata(method,input_type), "diagnostic_processor":a.processor}
    rs.builtin_metadata=described
    original_parts=rs.scoring_parts
    rs.scoring_parts=lambda: {**original_parts(), "diagnostic_processor":a.processor}
    rs.Matchmaker=build
    result=rs.evaluate_submission(builtin='dixon',input_type='audio',fold='dixon_features',
                                 shard=a.shard,num_shards=a.num_shards,run_dir=a.output)
    if result['n_failed']:
        raise SystemExit('Incomplete feature diagnostic')

if __name__=='__main__':main()
