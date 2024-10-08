import base64
import json
import logging
import os

import dotenv
import openai
import requests

dotenv.load_dotenv()


class WeaveAPIClient:
    def __init__(self):
        self.base_url = "https://trace.wandb.ai"
        self.auth = self._get_auth()

    def _get_auth(self):
        username = os.getenv("WANDB_USERNAME")
        api_key = os.getenv("WANDB_API_KEY")
        return base64.b64encode(f"{username}:{api_key}".encode()).decode()

    def _make_request(self, endpoint, method="POST", payload=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Basic {self.auth}",
        }
        response = requests.request(
            method, url, headers=headers, data=json.dumps(payload) if payload else None
        )
        response.raise_for_status()
        return response.json()

    def get_sample_count(
        self, project_id, additional_filter=None, additional_query=None
    ):
        base_filter = {"trace_roots_only": True}
        base_query = {
            "$expr": {
                "$and": [
                    {"$eq": [{"$getField": "exception"}, {"$literal": None}]},
                    {
                        "$not": [
                            {
                                "$contains": {
                                    "input": {"$getField": "op_name"},
                                    "substr": {"$literal": "evaluation"},
                                    "case_insensitive": True,
                                }
                            }
                        ],
                    },
                ]
            }
        }

        if additional_filter:
            base_filter.update(additional_filter)
        if additional_query:
            base_query["$expr"]["$and"].extend(
                additional_query.get("$expr", {}).get("$and", [])
            )

        payload = {"project_id": project_id, "filter": base_filter, "query": base_query}
        try:
            response = self._make_request("calls/query_stats", payload=payload)
            return response["count"]
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error occurred: {e}")
            logging.error(f"Response content: {e.response.content}")
            return 0
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return 0
        return response["count"]

    def get_calls(
        self, project_id, start=0, end=10, additional_filter=None, additional_query=None
    ):
        base_filter = {"trace_roots_only": True}
        base_query = {
            "$expr": {
                "$and": [
                    {"$eq": [{"$getField": "exception"}, {"$literal": None}]},
                    {
                        "$not": [
                            {
                                "$contains": {
                                    "input": {"$getField": "op_name"},
                                    "substr": {"$literal": "evaluation"},
                                    "case_insensitive": True,
                                }
                            }
                        ]
                    },
                ]
            }
        }

        if additional_filter:
            base_filter.update(additional_filter)
        if additional_query:
            base_query["$expr"]["$and"].extend(
                additional_query.get("$expr", {}).get("$and", [])
            )

        payload = {
            "project_id": project_id,
            "filter": base_filter,
            "limit": end - start,
            "offset": start,
            "include_costs": False,
            "query": base_query,
            "sort_by": [{"field": "started_at", "direction": "desc"}],
        }

        try:
            response = self._make_request("calls/stream_query", payload=payload)
            print(response)
            print(type(response))
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error fetching calls: {e}")
            return []

    def get_feedback_for_call(self, project_id: str, call_id: str) -> list[dict]:
        payload = {
            "project_id": project_id,
            "query": {
                "$expr": {
                    "$and": [
                        {
                            "$eq": [
                                {"$getField": "weave_ref"},
                                {"$literal": f"weave:///{project_id}/call/{call_id}"},
                            ]
                        }
                    ]
                }
            },
        }

        response = self._make_request("feedback/query", payload=payload)
        has_thumbsup = False
        has_thumbsdown = False
        feedback_notes = []
        try:
            for feedback in response.get("result", []):
                print(f"feedback: {json.dumps(feedback, indent=2)}")
                if feedback["feedback_type"] == "wandb.reaction.1":
                    has_thumbsup = feedback["payload"]["emoji"] == "👍"
                    has_thumbsdown = feedback["payload"]["emoji"] == "👎"
                if feedback["feedback_type"] == "wandb.note.1":
                    feedback_notes.append(feedback.get("payload", {}).get("note", ""))
        except Exception as e:
            print(e)
            pass

        return {
            "has_thumbsup": has_thumbsup,
            "has_thumbsdown": has_thumbsdown,
            "feedback_note": ",".join(feedback_notes),
        }

    def post_feedback(self, project_id: str, call_id: str, feedback: list[dict]):
        # iterate over feedback and post each item
        for item in feedback:
            payload = {
                "project_id": project_id,
                "weave_ref": f"weave:///{project_id}/call/{call_id}",
                "feedback_type": item["feedback_type"],
                "payload": item["payload"],
            }
            print(payload)
            response = self._make_request("feedback/create", payload=payload)
            print(response)
        return response

    def get_category_from_task(self, task: str):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "system",
                    "content": "User will provide his task for this annottation session, and you will answer with 1 word category name for the task.",
                },
                {
                    "role": "user",
                    "content": f"Classify the following task into a category: {task}",
                },
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content
