# Vendored from NVIDIA/NeMo-Skills@645cf567ff08c0ae9cc3fc8e1edbb975b3067816
# Source: nemo_skills/dataset/ruler2/tokenizer.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import List



def select_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type == "hf":
        return HFTokenizer(model_path=tokenizer_path)
    elif tokenizer_type == "openai":
        return OpenAITokenizer(model_path=tokenizer_path)
    elif tokenizer_type == "gemini":
        return GeminiTokenizer(model_path=tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer_type {tokenizer_type}")


class HFTokenizer:
    """
    Tokenizer from HF models
    """

    def __init__(self, model_path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text


class OpenAITokenizer:
    """
    Tokenizer from tiktoken
    """

    def __init__(self, model_path="cl100k_base") -> None:
        import tiktoken

        self.tokenizer = tiktoken.get_encoding(model_path)

    def text_to_tokens(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens)
        return text


