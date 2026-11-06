"""A downloaded run retains enough meter information to reproduce its TR."""
import json

import numpy as np

from matchmaker_eval.export_details import export
from tests.test_merge_shards import shard_record, write_shards, merge


def test_meter_and_criterion_survive_shard_merge_and_detail_export(tmp_path):
    criterion = {'error_unit':'musical_beat','threshold':.1,'window_duration':30.}
    record = shard_record(1,fold_size=1)
    record['pieces'][0]['tracking_criterion']=criterion
    root = write_shards(tmp_path/'shards',[record])
    shard = root/'shard-0'
    meter = {
        'schema':1,'position_unit':'denominator_note','tracking_error_unit':'musical_beat',
        'assumed_time_signature':False,
        'changes':[
            {'start_score_beat':0.,'numerator':6,'denominator':8,'denominator_notes_per_beat':3},
            {'start_score_beat':6.,'numerator':4,'denominator':4,'denominator_notes_per_beat':1},
        ],
    }
    (shard/'meter_1.json').write_text(json.dumps(meter))
    # The performed path returns to compound meter after visiting simple meter.
    path = np.array([[0.,0.],[1.,5.],[2.,6.],[3.,7.],[4.,2.],[5.,3.]])
    for name in ['wp','gt']:
        np.savetxt(shard/f'{name}_1.tsv',path,delimiter='\t',header='perf_sec\tscore_beat')
    out = tmp_path/'merged'
    merged = merge(root,out)
    assert merged['pieces'][0]['tracking_criterion']==criterion
    assert json.loads((out/'meter_1.json').read_text())==meter
    details = export(out)
    row = details['pieces'][0]
    assert 'tracking_criterion' not in details['metrics']
    assert row['tracking_criterion']==criterion and row['meter']==meter
    assert np.allclose(row['tr_tolerance'], [[0.,.3],[2.,.1],[4.,.3],[5.,.3]])
    assert np.allclose(np.array(row['err'])[:,1],0.)
