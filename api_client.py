import requests
import json
import os
import base64
import dotenv

dotenv.load_dotenv()

class WeaveAPIClient:
    def __init__(self):
        self.base_url = "https://trace.wandb.ai"
        self.auth = self._get_auth()

    def _get_auth(self):
        username = os.getenv('WANDB_USERNAME')
        api_key = os.getenv('WANDB_API_KEY')
        return base64.b64encode(f"{username}:{api_key}".encode()).decode()

    def _make_request(self, endpoint, method="POST", payload=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f"Basic {self.auth}"
        }
        response = requests.request(method, url, headers=headers, data=json.dumps(payload) if payload else None)
        response.raise_for_status()
        return response.json()

    def get_sample_count(self, project_id, additional_filter=None, additional_query=None):
        base_filter = {"trace_roots_only": True}
        base_query = {
            "$expr": {
                "$and": [
                    {
                        "$eq": [
                            {"$getField": "exception"},
                            {"$literal": None}
                        ]
                    }
                ]
            }
        }

        if additional_filter:
            base_filter.update(additional_filter)
        if additional_query:
            base_query["$expr"]["$and"].extend(additional_query.get("$expr", {}).get("$and", []))

        payload = {
            "project_id": project_id,
            "filter": base_filter,
            "query": base_query
        }
        response = self._make_request("calls/query_stats", payload=payload)
        return response['count']

    def get_calls(self, project_id, start=0, end=10, additional_filter=None, additional_query=None):
        base_filter = {"trace_roots_only": True}
        base_query = {
            "$expr": {
                "$and": [
                    {
                        "$eq": [
                            {"$getField": "exception"},
                            {"$literal": None}
                        ]
                    }
                ]
            }
        }

        if additional_filter:
            base_filter.update(additional_filter)
        if additional_query:
            base_query["$expr"]["$and"].extend(additional_query.get("$expr", {}).get("$and", []))

        payload = {
            "project_id": project_id,
            "filter": base_filter,
            "limit": end - start,
            "offset": start,
            "include_costs": False,
            "query": base_query,
            "sort_by": [
                {
                    "field": "started_at",
                    "direction": "desc"
                }
            ]
        }

        try:
            response = self._make_request("calls/stream_query", payload=payload)
            print(response)
            print(type(response))
            return response 
        except requests.exceptions.RequestException as e:
            print(f"Error fetching calls: {e}")
            return []

    def get_feedback_for_call(self, project_id, call_id):
        payload = {
            "project_id": project_id,
            "query": {
                "$expr": {
                    "$and": [
                        {
                            "$eq": [
                                {
                                    "$getField": "weave_ref"
                                },
                                {
                                    "$literal": f"weave:///{project_id}/call/{call_id}"
                                }
                            ]
                        }
                    ]
                }
            }
        }
        
        response = self._make_request("feedback/query", payload=payload)
        
        return response
    
    def post_feedback(self, project_id: str, call_id: str, feedback: list[dict]):
        #iterate over feedback and post each item
        for item in feedback:
            # check if feedback has note key in it and set feedback_type to wandb.note.1 else wandb.emoji.1
            if "note" in item:
                feedback_type = f"wandb.note.1"
            if "emoji" in item:
                feedback_type = f"wandb.reaction.1"

            payload = {
                "project_id": project_id,
                "weave_ref": f"weave:///{project_id}/call/{call_id}",
                "feedback_type": feedback_type,
                "payload": item
            }
            print(payload)
            response = self._make_request("feedback/create", payload=payload)
            print(response)
        return response