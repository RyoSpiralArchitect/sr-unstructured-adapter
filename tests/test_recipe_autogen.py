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


def test_suggester_preserves_unicode_and_varied_example_shapes():
    examples = [
        RecipeExample(text="請求書 123", target_type="title"),
        RecipeExample(text="請求書 456 (再発行)", target_type="title"),
    ]
    suggestion = RecipeSuggester(examples).suggest()
    assert suggestion.missed == 0
    assert suggestion.score == 1.0


def test_rendered_recipe_preserves_negative_classification():
    import yaml
    from sr_adapter.recipe import _build_recipe_config, apply_recipe_block
    from sr_adapter.schema import Block

    suggestion = RecipeSuggester([
        RecipeExample(text="Invoice #12345", target_type="title"),
    ]).suggest(negatives=["This is a paragraph."])
    config = _build_recipe_config("invoice", yaml.safe_load(render_yaml("invoice", suggestion)))
    positive = apply_recipe_block(Block(text="Invoice #12345"), config)
    negative = apply_recipe_block(Block(text="This is a paragraph."), config)
    assert positive.type == "title"
    assert negative.type == "paragraph"


def test_suggester_rejects_mixed_target_types():
    import pytest
    with pytest.raises(ValueError, match="share a target type"):
        RecipeSuggester([
            RecipeExample(text="Title", target_type="title"),
            RecipeExample(text="Body", target_type="paragraph"),
        ])
