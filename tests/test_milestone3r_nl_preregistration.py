import json
from pathlib import Path
import numpy as np

from actmask.experiments.milestone3r_nl import _audit, _history_offsets, config_hash


def test_preregistered_config_hash_and_required_fields_are_frozen():
    path=Path('outputs/actmask/milestone3r_nl_v1/preregistered_config.json')
    config=json.loads(path.read_text())
    assert config_hash(path)=='79d04d3204af01386d1a178a874d537a345bd2a8ca94545e750fcb4d50209b9f'
    assert config['history_frames']==6 and config['matching_tolerance']<=1e-6
    assert set(config['mechanisms'])=={'history_identifiable_damping_drive','hysteretic_mode_memory'}


def test_matched_final_audit_rejects_and_accepts_only_strict_pairs():
    history=np.zeros((2,6,3),np.float32);history[0,:4,1]=.1;history[1,:4,1]=-.1
    actions=np.zeros((2,20,3),np.float32);timestamps=np.tile(np.linspace(-.25,0,6,dtype=np.float32),(2,1));tcp=np.zeros((2,3),np.float32);static=np.zeros((2,9),np.float32)
    rows=[{'signed_group':'g'},{'signed_group':'g'}]
    audit=_audit(rows,history,timestamps,actions,tcp,static,np.array([1,0]))
    assert audit['groups']==1 and audit['max_matching_error']==0 and audit['min_earlier_history_difference']>.1


def test_early_history_multiset_is_matched_but_order_is_not():
    for mechanism in (0,1):
        positive=_history_offsets(1,mechanism,0)[0].numpy()
        negative=_history_offsets(1,mechanism,1)[0].numpy()
        assert np.array_equal(np.sort(positive),np.sort(negative))
        assert not np.array_equal(positive,negative)
