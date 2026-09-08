from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_full_employee_crud():
    res_create = client.post("/employees/", json={"id": 101, "name": "Sarah Khan", "role": "Senior Dev", "department": "Engineering"})
    assert res_create.status_code == 201
    assert res_create.json()["name"] == "Sarah Khan"

    res_get = client.get("/employees/101")
    assert res_get.status_code == 200

    res_update = client.put("/employees/101", json={"role": "Lead Architect"})
    assert res_update.status_code == 200

    res_list = client.get("/employees/")
    assert res_list.status_code == 200

    res_delete = client.delete("/employees/101")
    assert res_delete.status_code == 200
