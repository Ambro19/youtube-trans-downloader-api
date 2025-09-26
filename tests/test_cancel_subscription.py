# backend/tests/test_cancel_subscription.py
# why: verify cancel endpoint behavior without real DB/Stripe
from fastapi.testclient import TestClient
from main import app
from auth_deps import get_current_user
from main import get_db as main_get_db
import types

class StubDB:
    def __init__(self):
        self._sub = types.SimpleNamespace(
            stripe_subscription_id=None,
            status="active",
            extra_data="",
            cancelled_at=None,
        )

    def query(self, _):
        class Q:
            def __init__(self, sub): self.sub = sub
            def filter(self, *a, **k): return self
            def order_by(self, *a, **k): return self
            def first(self): return self.sub
        return Q(self._sub)

    def commit(self): pass
    def refresh(self, *a, **k): pass
    def rollback(self): pass

def override_db():
    yield StubDB()

class StubUser:
    def __init__(self, tier="premium"):
        self.id = 1
        self.username = "tester"
        self.email = "tester@example.com"
        self.subscription_tier = tier

def make_client(tier="premium"):
    app.dependency_overrides[get_current_user] = lambda: StubUser(tier)
    app.dependency_overrides[main_get_db] = override_db
    return TestClient(app)

def teardown_module(_):
    app.dependency_overrides.clear()

def test_cancel_at_period_end_success():
    client = make_client("premium")
    r = client.post("/subscription/cancel", json={})  # default: at_period_end True
    assert r.status_code == 200
    data = r.json()
    assert data["at_period_end"] is True
    assert data["status"] in {"scheduled_cancellation", "cancelled"}

def test_cancel_immediate_success():
    client = make_client("premium")
    r = client.post("/subscription/cancel", json={"at_period_end": False})
    assert r.status_code == 200
    data = r.json()
    assert data["at_period_end"] is False
    assert data["status"] == "cancelled"

def test_cancel_free_plan_fails():
    client = make_client("free")
    r = client.post("/subscription/cancel", json={})
    assert r.status_code == 400
    assert "No active subscription" in r.json()["detail"]
