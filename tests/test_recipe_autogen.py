# SPDX-License-Identifier: AGPL-3.0-or-later

import re

from sr_adapter.recipe_autogen import RecipeExample, RecipeSuggester, render_yaml


def test_recipe_suggester_matches_examples(tmp_path):
    examples = [
        RecipeExample(text="Invoice #12345", target_type="title"),
        RecipeExample(text="Invoice #98765", target_type="title"),
    ]
    suggester = RecipeSuggester(examples)
    suggestion = suggester.suggest(negatives=["Receipt 12345"])
    pattern = re.compile(suggestion.pattern)
    assert suggestion.missed == 0
    assert suggestion.false_positives == 0
    for example in examples:
        assert pattern.search(example.text)

    yaml_text = render_yaml("invoice-title", suggestion)
    output = tmp_path / "recipe.yaml"
    output.write_text(yaml_text, encoding="utf-8")
    assert "invoice-title" in yaml_text
    assert "patterns" in yaml_text
