import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_solution_module():
    module_path = Path(__file__).resolve().parents[1] / "submission" / "solution.py"
    spec = importlib.util.spec_from_file_location("submission_solution", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_snapshot(module, snapshot_id, taken_on, rate_value):
    markdown = f"""# Rates
| Product | Term | APY |
| --- | --- | --- |
| Share Savings | 12 Month | 0.15% |
| Certificate | 6 Month | {rate_value} |"""
    page = SimpleNamespace(
        url="https://www.dupaco.com/rates",
        title="Rates",
        markdown=markdown,
        fetched_at=f"{taken_on}T00:00:00Z",
    )
    return SimpleNamespace(snapshot_id=snapshot_id, taken_on=taken_on, pages=[page])


def test_latest_snapshot_is_used_by_default():
    module = load_solution_module()
    ks = module.build()

    ks.ingest(build_snapshot(module, "01_2026-01-01", "2026-01-01", "2.50%"))
    ks.ingest(build_snapshot(module, "02_2026-05-01", "2026-05-01", "3.50%"))

    answer = ks.answer("What is the 6-month certificate APY?")

    assert "3.50%" in answer.text
    assert answer.confident is True


def test_as_of_uses_older_snapshot_value():
    module = load_solution_module()
    ks = module.build()

    ks.ingest(build_snapshot(module, "01_2026-01-01", "2026-01-01", "2.50%"))
    ks.ingest(build_snapshot(module, "02_2026-05-01", "2026-05-01", "3.50%"))

    answer = ks.answer("What is the 6-month certificate APY?", as_of="2026-01-15")

    assert "2.50%" in answer.text
    assert answer.confident is True


def test_conflicts_reports_a_value_change():
    module = load_solution_module()
    ks = module.build()

    ks.ingest(build_snapshot(module, "01_2026-01-01", "2026-01-01", "2.50%"))
    ks.ingest(build_snapshot(module, "02_2026-05-01", "2026-05-01", "3.50%"))

    conflicts = ks.conflicts()

    assert conflicts
    assert any("6 Month" in conflict.topic or "certificate" in conflict.topic.lower() for conflict in conflicts)
