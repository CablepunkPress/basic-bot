"""Legacy model utilities.

Retained for summary.py until Phase 3 replaces summarization
with a provider-based or local model approach.
"""

import re

from anthropic.types import TextBlock

_SEQ_ANNOTATION = re.compile(r'<!--\s*seq:\d+\s*-->')


def extract_reply(response) -> str:
    """Extract the text reply from a Claude response.

    Strips any <!-- seq:N --> annotations the model may have echoed.
    """
    for block in response.content:
        if isinstance(block, TextBlock):
            return _SEQ_ANNOTATION.sub('', block.text).strip()
    raise ValueError("No text block found in response")
