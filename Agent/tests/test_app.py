import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_boots_with_fixture(cfg, monkeypatch):
    import Agent.config as c
    monkeypatch.setattr(c, "load_config", lambda path=None: cfg)
    at = st_testing.AppTest.from_file(str(c.AGENT_DIR / "app" / "app.py"), default_timeout=60).run()
    assert not at.exception


def test_train_panel_updates_estimate_with_cutoff(cfg, monkeypatch):
    import Agent.config as c
    monkeypatch.setattr(c, "load_config", lambda path=None: cfg)
    at = st_testing.AppTest.from_file(str(c.AGENT_DIR / "app" / "app.py"), default_timeout=60).run()
    at.sidebar.selectbox[0].set_value("train").run()
    first = {m.label: m.value for m in at.metric}
    assert {"Total life", "True RUL at this cycle", "Predicted RUL (XGBoost)"} <= set(first)
    at.sidebar.slider[0].set_value(60).run()
    second = {m.label: m.value for m in at.metric}
    assert second["True RUL at this cycle"] != first["True RUL at this cycle"]
    assert second["Predicted RUL (XGBoost)"] != first["Predicted RUL (XGBoost)"]


def test_default_cutoff_is_half_life_for_train_and_last_cycle_for_test(cfg, monkeypatch):
    import Agent.config as c
    monkeypatch.setattr(c, "load_config", lambda path=None: cfg)
    at = st_testing.AppTest.from_file(str(c.AGENT_DIR / "app" / "app.py"), default_timeout=60).run()
    sl = at.sidebar.slider[0]
    assert sl.value == sl.max  # test split: the last recorded cycle, the official evaluation endpoint
    assert {"Predicted RUL (XGBoost)", "True RUL at this cycle"} <= {m.label for m in at.metric}  # test engines get the official label
    at.sidebar.selectbox[0].set_value("train").run()
    sl = at.sidebar.slider[0]
    assert sl.value == max(1, sl.max // 2)  # train: half of the known total life


def test_test_engine_true_rul_follows_the_official_label(cfg, monkeypatch):
    import Agent.config as c
    monkeypatch.setattr(c, "load_config", lambda path=None: cfg)
    at = st_testing.AppTest.from_file(str(c.AGENT_DIR / "app" / "app.py"), default_timeout=60).run()
    at.sidebar.selectbox[1].set_value(3).run()
    last = at.sidebar.slider[0].max
    at.sidebar.slider[0].set_value(last - 10).run()
    truth = {m.label: m.value for m in at.metric}["True RUL at this cycle"]
    at.sidebar.slider[0].set_value(last).run()
    end_truth = {m.label: m.value for m in at.metric}["True RUL at this cycle"]
    assert int(truth.split()[0]) == int(end_truth.split()[0]) + 10  # label + cycles stepped back


def test_data_explorer_view_renders_all_three_datasets(cfg, monkeypatch):
    import Agent.config as c
    monkeypatch.setattr(c, "load_config", lambda path=None: cfg)
    at = st_testing.AppTest.from_file(str(c.AGENT_DIR / "app" / "app.py"), default_timeout=90).run()
    at.radio(key="view").set_value("Data explorer").run()
    assert not at.exception
    for label in list(at.radio(key="explorer_dataset").options):
        at.radio(key="explorer_dataset").set_value(label).run()
        assert not at.exception, label


def test_selection_survives_switching_views(cfg, monkeypatch):
    import Agent.config as c
    monkeypatch.setattr(c, "load_config", lambda path=None: cfg)
    at = st_testing.AppTest.from_file(str(c.AGENT_DIR / "app" / "app.py"), default_timeout=90).run()
    at.sidebar.selectbox[0].set_value("train").run()
    cutoff = at.sidebar.slider[0].value
    for view in ("Fleet scoring", "Data explorer", "RUL agent"):
        at.radio(key="view").set_value(view).run()
        assert not at.exception, view
    assert at.sidebar.selectbox[0].value == "train" and at.sidebar.slider[0].value == cutoff
