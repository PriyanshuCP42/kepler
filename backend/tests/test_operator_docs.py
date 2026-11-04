"""An operator's documentation page, read into plain blocks."""

import json

from alpha_harness.brain import operator_docs

PAGE = {
    "id": "ts_regression",
    "content": [
        {
            "type": "TEXT",
            "value": "<p><b>ts_regression(y, x, d, lag = 0, rettype = 0)</b></p>"
            "<p>Given two variables over d days:</p><ul><li>0: residual</li><li>2: beta</li></ul>",
        },
        {
            "type": "IMAGE",
            "value": {"title": "OLS Rettype List", "url": "https://example.test/rettype.png"},
        },
        {
            "type": "TABLE",
            "value": json.dumps({"data": [["Day", "Volume"], ["2024-06-01", "<b>100</b>"]]}),
        },
        {
            "type": "SIMULATION_EXAMPLE",
            "value": json.dumps(
                {
                    "settings": {
                        "region": "USA",
                        "delay": 1,
                        "universe": "TOP3000",
                        "neutralization": "INDUSTRY",
                        "decay": 3,
                        "truncation": 0.01,
                    },
                    "regular": "ts_regression(returns, close, 20)",
                }
            ),
        },
        {"type": "TABLE", "value": "not json"},
        {"type": "TEXT", "value": "<p>   </p>"},
    ],
}


def test_text_keeps_paragraphs_and_list_items_apart():
    text = operator_docs.plain("<p><b>sig</b></p><p>one</p><ul><li>a</li><li>b</li></ul>")
    assert text.splitlines() == ["sig", "one", "- a", "- b"]


def test_a_page_reads_into_typed_blocks():
    found = operator_docs.blocks(PAGE)
    assert [b["type"] for b in found] == ["text", "image", "table", "example"]
    assert "- 2: beta" in found[0]["text"]
    assert found[1] == {
        "type": "image",
        "text": "OLS Rettype List",
        "url": "https://example.test/rettype.png",
    }
    assert found[2]["text"].splitlines()[1] == "| 2024-06-01 | 100 |"
    assert found[3]["text"] == (
        "USA/D1/TOP3000, INDUSTRY, decay 3, truncation 0.01: ts_regression(returns, close, 20)"
    )


def test_an_unreadable_block_is_left_out_not_raised():
    assert operator_docs.blocks({"content": [{"type": "TABLE", "value": "{"}, "junk"]}) == []
