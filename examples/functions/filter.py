"""
title: Persona Rewrite Filter
author: gauri-nagavkar
version: 0.1
"""

from pydantic import BaseModel
from typing import Optional
import requests

profile_prompts = {}


def profile_rewrite(user_id, message):

    try:
        url = "http://middleware:8000/memory/store-and-search"
        response = requests.post(
            url,
            params={"user_id": str(user_id), "query": message},
            timeout=1000,
        )
        response.raise_for_status()
        data = response.json()
        return str(data)
    except Exception as e:
        print(f"[persona_filter] Exception during persona rewrite: {e}")
        return message


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        metadata = body.get("metadata", {})

        if messages and messages[-1].get("role") == "user":
            rewritten = profile_rewrite(
                metadata.get("user_id", "unknown_user"), messages[-1]["content"]
            )
            messages[-1]["content"] = rewritten
            print(f"[persona_filter] Rewritten message: {rewritten}")

            # Cache the rewritten prompt for reference in outlet
            profile_prompts[metadata.get("message_id", "default_id")] = rewritten

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        prompt = profile_prompts.pop(body.get("id", ""), "No modified prompt found")
        prompt_blockquote = "\n>" + prompt.replace("\n", "\n>") + "\n\n"

        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "assistant":
            details_markdown = f"""
<details>
  <summary>Modified Prompt</summary>
  {prompt_blockquote}
</details>
"""
            messages[-1]["content"] = f"{details_markdown}{messages[-1]['content']}"

        return body
