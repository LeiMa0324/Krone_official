# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM Types."""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Protocol
from typing_extensions import Unpack

from typing_extensions import NotRequired, TypedDict
from collections.abc import Callable
import openai
from dotenv import load_dotenv
import os
load_dotenv()
import re
import time


from openai import OpenAI
import asyncio
from typing import Generic, TypeVar, Optional
from typing import Any, Dict, List
from openai import AzureOpenAI
import json


T = TypeVar('T')

@dataclass
class LLMOutput(Generic[T]):
    """The output of an LLM invocation."""
    output: Optional[T] = None
    """The output of the LLM invocation."""
    json: Optional[Dict] = field(default=None)
    """The JSON output from the LLM, if available."""
    history: Optional[List[Dict]] = field(default=None)
    """The history of the LLM invocation, if available (e.g., chat mode)."""

    def preprocess_json_string(self, json_string):
        # Regular expression to find double quotes inside a value string that are not escaped
        pattern = r'(?<!\\)"(.*?)"(?=\s*[,:}])'  # Matches unescaped double quotes inside strings

        # Replace problematic quotes by escaping them
        corrected_json = re.sub(pattern, lambda match: re.sub(r'"', r'\"', match.group()), json_string)

        return corrected_json

    def to_json(self, default_json = None):
        # preprocessed_json_string = self.preprocess_json_string(self.output)
        try:
            process_dict = json.loads(self.output)
        except json.JSONDecodeError:
            process_dict = default_json
        return process_dict


class LLM:
    def __init__(self, model: str):
        self.model = model
        self.call_details = {"seq_ids":[], "call_type":[], "path_level": [], "path_identifier":[], "logkey_seq":[],
                             "example_seq_ids":[], "example_pattern_num":[], "detection":[],"duration":[]}

    def detect_existing_calls(self, call_detail):
        call_details_df = pd.DataFrame(self.call_details)
        match =  call_details_df[(call_details_df["call_type"]==call_detail["call_type"]) &
                                 (call_details_df["path_level"]==call_detail["path_level"]) &
                                 (call_details_df["path_identifier"] == call_detail["path_identifier"]) &
                                 (call_details_df["logkey_seq"] == call_detail["logkey_seq"])
                                 ]
        if len(match) >0:
            return True
        return False

    def __call__(self,prompt: str, variables: dict | None = None, call_detail: Dict=None, if_extract = False, default_json=None) -> LLMOutput[str]:
        if call_detail is not None:
            repeated_call = self.detect_existing_calls(call_detail)
            if repeated_call:
                print("Repeated llm call detected!")

        if variables is not None:
            for key, val in variables.items():
                prompt = prompt.replace("{"+f"{key}"+"}", f'{val}')
        self.prompt = prompt
        # print(prompt)
        if self.model == "gpt-3.5-turbo":
            api_key = os.getenv('OPENAI_API_KEY_3.5')
        elif self.model == "gpt-4-0613":
            api_key = os.getenv('OPENAI_API_KEY_4')
        else:
            raise Exception(f"Unsupported model {self.model}!")

        try:

            client = OpenAI(
                api_key=api_key,
            )

            start = time.time()
            response =client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                    }
                ],
                n=1,
                stop=None,
                temperature=0.7
            )

            end = time.time()
            generated_text = response.choices[0].message.content

            output = LLMOutput(
                output= generated_text,
            )

            if call_detail is not None:
                self.call_details["detection"].append(
                    output.to_json()["prediction"] if call_detail["call_type"] == 'detect' else "")
                self.call_details["duration"].append(end - start)

                for key in call_detail.keys():
                    self.call_details[key].append(call_detail[key])
            return output

        except Exception as e:
            if default_json is not None:
                return LLMOutput(output=str(e), json=default_json)
            default_json = {}
            if call_detail is not None:
                default_json = {"prediction": "Normal", "reason": "LLM failed"}
            if if_extract:
                default_json = {
                "source_entity": "none",
                "source_entity_type": "none",
                "target_entity": "none",
                "target_entity_type": "none",
                "action": "none",
                "status": "none",
                "summary": "none"
                }
            return LLMOutput(output=str(e), json=default_json)

    def store_call_details(self, detail_path):
        call_details = pd.DataFrame(self.call_details)
        summary_duration = call_details[call_details["call_type"]=='summary']["duration"].sum()
        detect_duration = call_details[call_details["call_type"]=='detect']["duration"].sum()
        summary = len(call_details[call_details['call_type']=='summary'])
        detection = len(call_details[call_details['call_type']=='detect'])
        total = summary +detection
        print(f"Total {len(call_details)} LLM calls\n "
              f"summary: {len(call_details[call_details['call_type']=='summary'])}, duration: {summary_duration}\n"
              f"detect: {len(call_details[call_details['call_type']=='detect'])}, duration: {detect_duration}")
        call_details.to_csv(detail_path, index=False)
        return total, summary, detection



class Embedding_LLM:
    def __init__(self, model: str):
        self.model = model

    def __call__(self, sequences: List[List[str]]):

        input_text = [" ".join(input) for input in sequences]
        if self.model == "text-embedding-ada-002":
            api_key = os.getenv('OPENAI_API_KEY_3.5')
        else:
            raise Exception(f"Unsupported model {self.model}!")

        try:
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint="https://search.bytedance.net/gpt/openapi/online/v2/crawl",
                api_version="2023-03-15-preview",
            )
            response =client.embeddings.create(
                model=self.model,
                input= input_text
            )

            # Extract the embedding from the response
            embeddings = [item['embedding'] for item in response['data']]
            return np.array(embeddings)

        except Exception as e:
            raise Exception(str(e))