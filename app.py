from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(title="Employee Management REST API")
employees_db: Dict[int, dict] = {}

class Employee(BaseModel):
    id: int
    name: str
    role: str
    department: str

class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None

@app.post("/employees/", status_code=201)
def create_employee(emp: Employee):
    if emp.id in employees_db:
        raise HTTPException(status_code=400, detail="Employee ID already exists.")
    employees_db[emp.id] = emp.dict()
    return employees_db[emp.id]

@app.get("/employees/", response_model=List[dict])
def list_employees():
    return list(employees_db.values())

@app.get("/employees/{emp_id}")
def get_employee(emp_id: int):
    if emp_id not in employees_db:
        raise HTTPException(status_code=404, detail="Employee not found.")
    return employees_db[emp_id]

@app.put("/employees/{emp_id}")
def update_employee(emp_id: int, emp_data: EmployeeUpdate):
    if emp_id not in employees_db:
        raise HTTPException(status_code=404, detail="Employee not found.")
    current_data = employees_db[emp_id]
    updated_data = emp_data.dict(exclude_unset=True)
    current_data.update(updated_data)
    employees_db[emp_id] = current_data
    return current_data

@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: int):
    if emp_id not in employees_db:
        raise HTTPException(status_code=404, detail="Employee not found.")
    deleted_emp = employees_db.pop(emp_id)
    return {"message": "Employee deleted successfully", "employee": deleted_emp}
