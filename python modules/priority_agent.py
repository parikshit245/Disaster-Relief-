from pydantic import BaseModel
from typing import List

class priority_schema(BaseModel):
    severity: str
    people_affected: int
    deadline_hours: float
    distance_km: float

class ranked_request(BaseModel):
    id: str
    location: str
    severity: str
    people_affected: int
    deadline_hours: float
    distance_km: float
    need_type: str
    score: float = 0.0
    
class priority_agent:
    
    severity_weights = {
        'low': 1,
        'medium': 2,
        'high': 3,
        'critical': 4,
    }
    
    def compute_priority_score(self, req: priority_schema) -> float:
        """
        Compute a numeric urgency score for a single request.
        """
        severity_lower = req.severity.lower()
        sev_score = self.severity_weights.get(severity_lower, 0)
        people_score = min(req.people_affected / 100, 10)
        deadline_hours_score = max(10 - req.deadline_hours, 0)
        distance_score = max(10 - req.distance_km / 10, 0)
        
        score = (sev_score * 4) + (people_score * 3) + (deadline_hours_score * 2) + (distance_score * 1)
        return round(score, 2)
        
    def rank_requests(self, requests: List[dict]) -> List[ranked_request]:
        """
        Score and sort all requests by priority.
        """
        ranked_items = []
        for req_data in requests:
            req_schema = priority_schema(**req_data)
            score = self.compute_priority_score(req_schema)
            
            ranked_item = ranked_request(**req_data, score=score)
            ranked_items.append(ranked_item)
            
        ranked_items.sort(key=lambda r: r.score, reverse=True)
        return ranked_items
    
    def get_top_request(self, requests: List[dict]) -> ranked_request:
        """
        Get the highest priority request from the list
        """
        ranked_items = self.rank_requests(requests)
        return ranked_items[0] if ranked_items else None
    
    def print_priority_list(self, ranked_items: List[ranked_request]):
        """
        Print the ranked requests in a readable format.
        """
        print("Ranked Requests:")
        for item in ranked_items:
            print(f"ID: {item.id}, Location: {item.location}, Severity: {item.severity}, "
                  f"People Affected: {item.people_affected}, Deadline Hours: {item.deadline_hours}, "
                  f"Distance (km): {item.distance_km}, Need Type: {item.need_type}, Score: {item.score}")
            
            
# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    sample_requests = [
        {'id': 'REQ-01', 'location': 'Village A', 'severity': 'High',     'people_affected': 150, 'deadline_hours': 6,  'distance_km': 20, 'need_type': 'Medical'},
        {'id': 'REQ-02', 'location': 'Village B', 'severity': 'Medium',   'people_affected': 80,  'deadline_hours': 12, 'distance_km': 35, 'need_type': 'Food'},
        {'id': 'REQ-03', 'location': 'Village C', 'severity': 'Critical', 'people_affected': 340, 'deadline_hours': 2,  'distance_km': 15, 'need_type': 'Medical'},
        {'id': 'REQ-04', 'location': 'Hospital B','severity': 'High',     'people_affected': 210, 'deadline_hours': 4,  'distance_km': 10, 'need_type': 'Rescue'},
    ]
    priority_agent = priority_agent()
    ranked = priority_agent.rank_requests(sample_requests)
    priority_agent.print_priority_list(ranked)
 
    top = priority_agent.get_top_request(sample_requests)
    print(f"Top priority request: {top.id} at {top.location} (score: {top.score})")