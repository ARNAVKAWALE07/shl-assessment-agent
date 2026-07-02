#catalog fetch from link

import httpx
from typing import List, Dict, Any
import json


class CatalogFetch:
    def __init__(self, json_url:str):
        self.catalog: List[Dict[str, Any]] = []
        self.load_and_clean_from_url(json_url)
    
    def determine_test_type(self, keys:List[str]) -> str:
        """ maps the catalog keys array into strict single-characters"""

        keys_joined = " ".join(keys).lower()
        if "personality" in keys_joined or "behavior" in keys_joined or "comptencies" in keys_joined:
            return "P"
        if "ability" in keys_joined or "aptitude" in keys_joined or "cognitive" in keys_joined:
            return "A"
        return "K" # skill/ knowledge default
    
    def load_and_clean_from_url(self, json_url: str):
        """ fetch the catalog json over Http and normalize it when server start"""

        try:
            response = httpx.get(json_url, timeout=30.0)
            response.raise_for_status()
            raw_data = json.loads(response.text, strict=False)
            
            for item in raw_data:
                cleaned_data = {
                    "name": item.get("name", "").strip(),
                    "url": item.get("link", "").strip(),
                    "test_type": self.determine_test_type(item.get("keys", [])),
                    "description": item.get("description", "").strip(),
                    "job_levels": [lvl.strip() for lvl in item.get("job_levels", []) if lvl.strip()]
                }

                # cleaned item that are valid
                if cleaned_data["name"] and cleaned_data["url"]:
                    self.catalog.append(cleaned_data)

                print(f"Successfully loaded {len(self.catalog)} assessment from catalog")
        except Exception as e:
            print(f"Critical Error URL not fetch :{e}")
            raise e
        
    def get_assessment_data(self) -> str:
        """ Format the entire dataset compactly to pass to llm """
        context_lines = []
        for index, item in enumerate(self.catalog):
            levels = " , ".join(item["job_levels"])
            line = f"[{index}] Name: {item['name']} | Type :{item['test_type']} | URL: {item['url']} | Level :{levels} | Desc :{item['description']}"
            context_lines.append(line)
        return "\n".join(context_lines)