from scripts import run_e2e_demo


def test_e2e_demo_summary(monkeypatch, capsys):
    def fake_demo():
        return None

    async def fake_counts(_database_url: str):
        return 2, 5

    monkeypatch.setenv("DATABASE_URL", "postgresql://awet:awet@localhost:5433/awet")

    counts = run_e2e_demo.run_e2e_demo(demo_runner=fake_demo, count_fetcher=fake_counts)
    assert counts == (2, 5)

    paper_trades, audit_events = counts
    if paper_trades > 0 and audit_events > 0:
        print(f"E2E demo OK: {paper_trades} paper trades, {audit_events} audit events.")
    out = capsys.readouterr().out
    assert "E2E demo OK" in out
