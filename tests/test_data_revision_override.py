from matchmaker_eval.fetch_data import load_config, branch_for


def test_explicit_revision_overrides_per_fold_branches(monkeypatch):
    revision='a'*40
    monkeypatch.setenv('MATCHMAKER_DATA_BRANCH',revision)
    config=load_config()
    for fold in ['eval','beyond_piano','dixon_features']:
        assert branch_for(config,fold)==revision
