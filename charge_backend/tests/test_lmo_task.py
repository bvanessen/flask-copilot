###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import asyncio

import pytest

from charge_backend.lmo.lmo_task import (
    LMOTask,
    MoleculeOutputSchema,
    apply_diversity_penalty,
    build_customization_text,
    build_refine_prompt,
    build_system_prompt,
    build_user_prompt,
    max_tanimoto_similarity,
)


# --- MoleculeOutputSchema ---


def test_schema_accepts_valid_smiles():
    result = MoleculeOutputSchema(
        reasoning_summary="test",
        smiles_list=["CCO", "c1ccccc1"],
        property_name="density",
        property_list=[1.0, 2.0],
    )
    assert result.as_list() == [("CCO", 1.0), ("c1ccccc1", 2.0)]
    assert result.as_dict()["smiles_list"] == ["CCO", "c1ccccc1"]


def test_schema_rejects_invalid_smiles():
    with pytest.raises(ValueError):
        MoleculeOutputSchema(
            reasoning_summary="test",
            smiles_list=["not_a_smiles((("],
            property_name="density",
            property_list=[1.0],
        )


# --- Prompt builders ---


def test_build_system_prompt_mentions_property_and_direction():
    prompt = build_system_prompt("density", "greater")
    assert "density" in prompt
    assert "greater" in prompt


def test_build_user_prompt_fills_template():
    prompt = build_user_prompt(
        lead_smiles="CCO",
        property_name="density",
        property_description="crystalline density (g/cm^3)",
        optimize_direction="greater",
        calculate_property_tool="calculate_property_hf",
        depth=3,
        molecular_similarity=0.7,
        num_top_candidates=3,
        number_of_molecules=10,
    )
    assert "lead_smiles = CCO" in prompt
    assert "calculate_property_hf" in prompt
    assert "exactly 3 rounds" in prompt
    assert "Require density strictly higher than the lead." in prompt
    assert "{" not in prompt.replace("{property_name}", "")  # no unfilled fields


def test_build_refine_prompt_includes_history():
    prompt = build_refine_prompt(
        lead_smiles="CCO",
        property_name="density",
        property_description="crystalline density (g/cm^3)",
        optimize_direction="less",
        calculate_property_tool="calculate_property_hf",
        molecular_similarity=0.7,
        num_top_candidates=3,
        number_of_molecules=10,
        previous_smiles=["CCN", "CCC"],
        previous_values=[1.1, 1.2],
    )
    assert "previous_smiles = [CCN, CCC]" in prompt
    assert "previous_values = [1.1, 1.2]" in prompt
    assert "Require density strictly lower than the lead." in prompt


def test_customization_text_disabled():
    assert build_customization_text(False, 0.2, 0.9, 0.9, ["drug-likeness"]) == ""


def test_customization_text_branches():
    text = build_customization_text(True, 0.2, 0.9, 0.9, ["drug-likeness"])
    assert "diverse chemical modifications" in text  # low similarity
    assert "diversity penalty: 0.90" in text
    assert "Focus on exploration" in text
    assert "Lipinski" in text

    text = build_customization_text(True, 0.9, 0.0, 0.1, None)
    assert "conservative modifications" in text  # high similarity
    assert "Focus on exploitation" in text

    # Mid-range values with no constraints produce no guidance
    assert build_customization_text(True, 0.7, 0.0, 0.5, []) == ""


# --- Similarity / diversity penalty ---


def test_max_tanimoto_similarity_identical():
    assert max_tanimoto_similarity("CCO", ["CCO"]) == 1.0


def test_max_tanimoto_similarity_takes_max():
    sim = max_tanimoto_similarity("CCO", ["c1ccccc1", "CCO"])
    assert sim == 1.0


def test_max_tanimoto_similarity_empty_and_invalid():
    assert max_tanimoto_similarity("CCO", []) == 0.0
    assert max_tanimoto_similarity("not_a_smiles(((", ["CCO"]) == 0.0
    # invalid entries in `others` are skipped
    assert max_tanimoto_similarity("CCO", ["not_a_smiles((("]) == 0.0


def test_apply_diversity_penalty_below_threshold_unchanged():
    assert (
        apply_diversity_penalty(
            2.0, "greater", 0.5, similarity=0.3, similarity_threshold=0.7
        )
        == 2.0
    )


def test_apply_diversity_penalty_zero_penalty_unchanged():
    assert (
        apply_diversity_penalty(
            2.0, "greater", 0.0, similarity=0.9, similarity_threshold=0.7
        )
        == 2.0
    )


def test_apply_diversity_penalty_greater_direction_reduces():
    # penalized = 2.0 * (1 - 0.5 * 0.8) = 1.2
    assert apply_diversity_penalty(
        2.0, "greater", 0.5, similarity=0.8, similarity_threshold=0.7
    ) == pytest.approx(1.2)


def test_apply_diversity_penalty_less_direction_increases():
    # penalized = 2.0 * (1 + 0.5 * 0.8) = 2.8
    assert apply_diversity_penalty(
        2.0, "less", 0.5, similarity=0.8, similarity_threshold=0.7
    ) == pytest.approx(2.8)


# --- LMOTask.check_proposal with stubbed property evaluation ---


def _make_task(**kwargs) -> LMOTask:
    defaults = dict(
        lead_molecule="CCO",
        user_prompt="test prompt",
        property_tool_name="test_tool",
        property_name="density",
        optimize_direction="greater",
        reference_property_value=1.0,
    )
    defaults.update(kwargs)
    return LMOTask(**defaults)


def _stub_eval(task: LMOTask, value: float) -> None:
    async def fake_evaluate(smiles: str) -> float:
        return value

    task._evaluate_property = fake_evaluate


def test_check_proposal_accepts_improvement():
    task = _make_task()
    _stub_eval(task, 2.0)
    # CCO -> CCN keeps synthesizability comparable; use the lead itself to be safe
    value = asyncio.run(task.check_proposal("CCO"))
    assert value == 2.0


def test_check_proposal_rejects_invalid_smiles():
    task = _make_task()
    _stub_eval(task, 2.0)
    with pytest.raises(ValueError, match="Invalid SMILES"):
        asyncio.run(task.check_proposal("not_a_smiles((("))


def test_check_proposal_rejects_worse_property_greater():
    task = _make_task()
    _stub_eval(task, 0.5)  # below reference of 1.0
    with pytest.raises(ValueError, match="too low"):
        asyncio.run(task.check_proposal("CCO"))


def test_check_proposal_rejects_worse_property_less():
    task = _make_task(optimize_direction="less")
    _stub_eval(task, 2.0)  # above reference of 1.0
    with pytest.raises(ValueError, match="too high"):
        asyncio.run(task.check_proposal("CCO"))


def test_check_final_proposal_parses_json_list():
    task = _make_task()
    _stub_eval(task, 2.0)
    assert asyncio.run(task.check_final_proposal('["CCO"]')) is True


def test_check_final_proposal_parses_python_literal_list():
    task = _make_task()
    _stub_eval(task, 2.0)
    assert asyncio.run(task.check_final_proposal("['CCO']")) is True


def test_check_final_proposal_rejects_garbage():
    task = _make_task()
    _stub_eval(task, 2.0)
    with pytest.raises(ValueError):
        asyncio.run(task.check_final_proposal("not a list at all"))


def test_get_initial_property_value_uses_provided_reference():
    task = _make_task(reference_property_value=3.5)

    async def fail_evaluate(smiles: str) -> float:  # must not be called
        raise AssertionError("MCP fetch should be skipped")

    task._evaluate_property = fail_evaluate
    assert asyncio.run(task.get_initial_property_value()) == 3.5
