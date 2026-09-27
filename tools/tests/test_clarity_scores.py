"""Tests for tools/clarity_scores.py's paragraph extraction.

Only the offline half: nothing here imports typesafe_sdk, which is not a
project dependency, or calls Jev.
"""

from tools.clarity_scores import paragraphs
from tools.markdown import Document

TEXT = """\
# Title

## First Section

An opening paragraph
wrapped over two lines.

```python
# demo.py
x = 1
```

The paragraph after the listing
explains it, and see [Other](01_Intro.md#anchor).

- a list item
- another

After the list.

## Second Section

A fresh start.
"""


def extract() -> list:
    return list(paragraphs(Document.from_text(TEXT)))


def test_joins_lines_and_numbers_from_one() -> None:
    first = extract()[0]
    assert first.text == "An opening paragraph wrapped over two lines."
    assert first.line == 5
    assert first.section == "First Section"


def test_listing_directly_above_is_context() -> None:
    after = extract()[1]
    assert after.listing == "# demo.py\nx = 1"
    assert after.previous.startswith("An opening")


def test_list_items_are_skipped_and_clear_the_listing() -> None:
    after_list = extract()[2]
    assert after_list.text == "After the list."
    assert after_list.listing == ""


def test_heading_resets_context() -> None:
    last = extract()[3]
    assert last.section == "Second Section"
    assert last.previous == ""


def test_state_strips_link_targets_and_empty_fields() -> None:
    state = extract()[1].state()
    assert state["paragraph"].endswith("see Other.")
    first = extract()[0].state()
    assert set(first) == {"section", "paragraph"}
