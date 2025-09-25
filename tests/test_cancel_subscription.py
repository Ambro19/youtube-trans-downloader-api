# backend/tests/test_cancel_subscription.py
# Minimal FastAPI tests for the cancel endpoint (no real DB/Stripe).
import types
from fastapi.testclient import TestClient
from main import app
from auth_deps import get_current_user
from main import get_db as main_get_db  # the exact dependency used in endpoints

class StubDB:
  def __init__(self):
    # fake "latest subscription" row
    self.sub = types.SimpleNamespace(
      stripe_subscription_id=None,
      status="active",
      extra_data="",
      cancelled_at=None,
    )
  def query(self, model):
    class Q:
      def __init__(self, sub): self._sub = sub
      def filter(self, *a, **k): return self
      def order_by(self, *a, **k): return self
      def first(self): return self._sub
    return Q(self.sub)
  def commit(self): pass
  def refresh(self, *a, **k): pass
  def rollback(self): pass
  def add(self, *a, **k): pass

def override_db():
  db = StubDB()
  yield db

class StubUser:
  def __init__(self, tier="premium"):
    self.id = 1
    self.username = "tester"
    self.email = "tester@example.com"
    self.subscription_tier = tier

def client_with_user(tier="premium"):
  app.dependency_overrides[get_current_user] = lambda: StubUser(tier)
  app.dependency_overrides[main_get_db] = override_db
  return TestClient(app)

def teardown_module(_m):
  app.dependency_overrides.clear()

def test_cancel_at_period_end_success():
  client = client_with_user("premium")
  r = client.post("/subscription/cancel", json={})  # default at_period_end=True
  assert r.status_code == 200
  payload = r.json()
  assert payload["at_period_end"] is True
  assert payload["status"] in ("scheduled_cancellation", "cancelled")

def test_cancel_immediate_success():
  client = client_with_user("premium")
  r = client.post("/subscription/cancel", json={"at_period_end": False})
  assert r.status_code == 200
  payload = r.json()
  assert payload["at_period_end"] is False
  assert payload["status"] == "cancelled"

def test_cancel_free_plan_fails():
  client = client_with_user("free")
  r = client.post("/subscription/cancel", json={})
  assert r.status_code == 400
  assert "No active subscription" in r.json()["detail"]
