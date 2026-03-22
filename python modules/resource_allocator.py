# resource_allocator.py
# Algorithm: 0/1 Knapsack via Dynamic Programming
# Optional: Fractional Knapsack (greedy) as fallback when DP is overkill
 
from pydantic import BaseModel, Field
from typing import List
class resource_schema(BaseModel):
    id: str = Field(..., description="Unique identifier for the resource")
    name: str = Field(..., description="Name of the resource")
    weight: float = Field(..., description="weight  of the resource in kg")
    value: float = Field(..., description="value or price of the resource ")
    quantity: float = Field(..., description="avaliable quantity of the resource")
    
class resource_allocator:
    """ This class deals with resource allocation at teh time of limit restriction on the transporations carriers. Done by  fractional knapsacks."""
    
    allocated_resources: List[resource_schema]
    def __init__(self, resources: List[dict], capacity: float):
        self.allocated_resources = []
        resource_list = [resource_schema(**res) for res in resources]
        self.fractional_knapsack(resource_list, capacity)  
        
    def fractional_knapsack(self, resourses: List[resource_schema], capacity: float) -> List[resource_schema]:
        """
        Greedy approach for fractional knapsack.
        """
        resourses.sort(key=lambda r: r.value / r.weight, reverse=True)
        
        allocated = []
        remaining_capacity = capacity
        
        for res in resourses:
            if remaining_capacity <= 0:
                break
            
            if res.weight * res.quantity <= remaining_capacity:
                allocated.append(res)
                remaining_capacity -= res.weight * res.quantity
            else:
                fraction = remaining_capacity / (res.weight * res.quantity)
                allocated.append(resource_schema(id=res.id,  name=res.name, weight=res.weight, value=res.value, quantity=float(res.quantity * fraction)))
                remaining_capacity = 0
        self.allocated_resources = allocated    
        
    def print_allocated_resources(self):
        print("Allocated Resources:")
        for res in self.allocated_resources:
            print(f"ID: {res.id}, Name: {res.name}, Weight: {res.weight} kg, Value: {res.value}, Quantity: {res.quantity}", end="\n")
            

# ------------- Quick test --------------

if __name__ == "__main__":
    resources = [
        {"id": "r1", "name": "Water", "weight": 10, "value": 100, "quantity": 5},
        {"id": "r2", "name": "Food", "weight": 20, "value": 200, "quantity": 3},
        {"id": "r3", "name": "Medicine", "weight": 5, "value": 150, "quantity": 10},
    ]
    capacity = 100
    allocator = resource_allocator(resources, capacity)
    allocator.print_allocated_resources()