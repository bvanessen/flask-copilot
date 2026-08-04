import ast
import json
from pathlib import Path
from typing import List, Optional, Tuple

import charge
from charge.tasks.task import Task
from charge.utils.mcp_workbench_utils import call_mcp_tool_directly
from flask_tools.chemistry import smiles_utils
from loguru import logger
from pydantic import BaseModel, field_validator
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

with open(PROMPTS_DIR / "lmo_user_prompt.txt", "r") as f:
    PROPERTY_USER_PROMPT = f.read()

with open(PROMPTS_DIR / "lmo_refine_prompt.txt", "r") as f:
    FURTHER_REFINE_PROMPT = f.read()

SYSTEM_PROMPT_TEMPLATE = (
    "You are a world-class chemist specializing in molecular design and property optimization. "
    "Your task is to propose novel small molecules that optimize the specified property '"
    "{property_name}' (direction: '{optimize_direction}') while maintaining synthetic accessibility. "
    "You will be provided with a lead molecule as a starting point. Generate new molecules in SMILES format "
    "and use available tools to evaluate them.\n\n"
)


class MoleculeOutputSchema(BaseModel):
    """
    Structure output representing a valid list of SMILES strings.
    """

    reasoning_summary: str
    smiles_list: List[str]
    property_name: str
    property_list: List[float]

    @field_validator("smiles_list")
    @classmethod
    def validate_smiles_list(cls, smiles_list):
        if not isinstance(smiles_list, list):
            raise ValueError("smiles_list must be a list.")
        for smiles in smiles_list:
            if not isinstance(smiles, str):
                raise ValueError("Each SMILES must be a string.")
            if not smiles_utils.verify_smiles(smiles):
                raise ValueError(f"Invalid SMILES string: {smiles}")
        return smiles_list

    def as_list(self) -> List[Tuple[str, float]]:
        return list(zip(self.smiles_list, self.property_list))

    def as_dict(self) -> dict:
        return {
            "reasoning_summary": self.reasoning_summary,
            "smiles_list": self.smiles_list,
            "property_name": self.property_name,
            "property_list": self.property_list,
        }


def build_system_prompt(property_name: str, optimize_direction: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        property_name=property_name,
        optimize_direction=optimize_direction,
    )


def build_customization_text(
    enable_constraints: bool,
    molecular_similarity: float,
    diversity_penalty: float,
    exploration_rate: float,
    additional_constraints: Optional[list[str]],
) -> str:
    """Build the optimization-strategy guidance appended to user prompts."""
    if not enable_constraints:
        return ""

    customization_guidance = []

    if molecular_similarity < 0.5:
        customization_guidance.append(
            f"You should explore diverse chemical modifications, "
            f"as the molecular similarity threshold is low ({molecular_similarity:.2f})."
        )
    elif molecular_similarity > 0.8:
        customization_guidance.append(
            f"You should make conservative modifications, "
            f"keeping molecules very similar to the parent (similarity threshold: {molecular_similarity:.2f})."
        )

    if diversity_penalty > 0.5:
        customization_guidance.append(
            f"Prioritize generating chemically diverse molecules to explore different regions "
            f"of chemical space (diversity penalty: {diversity_penalty:.2f})."
        )

    if exploration_rate > 0.7:
        customization_guidance.append(
            f"Focus on exploration - try novel structural modifications "
            f"(exploration rate: {exploration_rate:.2f})."
        )
    elif exploration_rate < 0.3:
        customization_guidance.append(
            f"Focus on exploitation - make incremental improvements to known good structures "
            f"(exploration rate: {exploration_rate:.2f})."
        )

    constraint_guidance_map = {
        "drug-likeness": "Ensure all molecules satisfy drug-likeness criteria (Lipinski's Rule of Five)",
        "synthesizability": "Prioritize molecules with high synthetic accessibility scores (SA score < 3)",
        "lead-likeness": (
            "Apply lead-likeness criteria suitable for early drug discovery "
            "(MW 200-350, LogP 1-3)"
        ),
        "pan-assay-interference": "Filter out Pan-Assay Interference Compounds (PAINS) and other promiscuous binders",
        "toxicity-rules": "Apply structural alerts to avoid potential toxicity issues",
        "reactive-groups": "Avoid molecules with highly reactive functional groups (epoxides, acyl halides, etc.)",
    }
    for constraint in additional_constraints or []:
        if constraint in constraint_guidance_map:
            customization_guidance.append(constraint_guidance_map[constraint])

    if not customization_guidance:
        return ""
    return (
        "\n\nOptimization Strategy:\n"
        + "\n".join(f"- {g}" for g in customization_guidance)
        + "\n"
    )


def _prompt_common_fields(
    property_name: str,
    property_description: str,
    optimize_direction: str,
    calculate_property_tool: str,
    molecular_similarity: float,
    num_top_candidates: int,
    number_of_molecules: int,
    lead_smiles: str,
) -> dict:
    direction = "higher" if optimize_direction == "greater" else "lower"
    return dict(
        property_name=property_name,
        property_description=property_description,
        objective="maximize" if optimize_direction == "greater" else "minimize",
        chemical_domain_context=(
            "small-molecule property optimization; consider synthesizability and lead-likeness"
        ),
        similarity_mode=f"threshold={molecular_similarity:.2f}",
        similarity_anchor="lead",
        calculate_property_tool=calculate_property_tool,
        top_k=num_top_candidates,
        improvement_line=f"Require {property_name} strictly {direction} than the lead.",
        lead_smiles=lead_smiles,
        candidates_per_round=number_of_molecules,
    )


def build_user_prompt(
    lead_smiles: str,
    property_name: str,
    property_description: str,
    optimize_direction: str,
    calculate_property_tool: str,
    depth: int,
    molecular_similarity: float,
    num_top_candidates: int,
    number_of_molecules: int,
    customization_text: str = "",
) -> str:
    """Format the initial optimization prompt from prompts/lmo_user_prompt.txt."""
    return (
        PROPERTY_USER_PROMPT.format(
            rounds=depth,
            **_prompt_common_fields(
                property_name,
                property_description,
                optimize_direction,
                calculate_property_tool,
                molecular_similarity,
                num_top_candidates,
                number_of_molecules,
                lead_smiles,
            ),
        )
        + customization_text
    )


def build_refine_prompt(
    lead_smiles: str,
    property_name: str,
    property_description: str,
    optimize_direction: str,
    calculate_property_tool: str,
    molecular_similarity: float,
    num_top_candidates: int,
    number_of_molecules: int,
    previous_smiles: List[str],
    previous_values: List[float],
    customization_text: str = "",
) -> str:
    """Format the refinement-round prompt from prompts/lmo_refine_prompt.txt."""
    return (
        FURTHER_REFINE_PROMPT.format(
            previous_smiles=", ".join(previous_smiles),
            previous_values=", ".join(map(str, previous_values)),
            **_prompt_common_fields(
                property_name,
                property_description,
                optimize_direction,
                calculate_property_tool,
                molecular_similarity,
                num_top_candidates,
                number_of_molecules,
                lead_smiles,
            ),
        )
        + customization_text
    )


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def max_tanimoto_similarity(smiles: str, others: list[str]) -> float:
    """
    Max Tanimoto similarity (Morgan fingerprints, radius 2) between `smiles`
    and each molecule in `others`. Unparseable SMILES contribute 0.0.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not others:
        return 0.0
    fp = _MORGAN_GEN.GetSparseCountFingerprint(mol)
    max_sim = 0.0
    for other in others:
        other_mol = Chem.MolFromSmiles(other)
        if other_mol is None:
            continue
        sim = DataStructs.TanimotoSimilarity(
            fp, _MORGAN_GEN.GetSparseCountFingerprint(other_mol)
        )
        max_sim = max(max_sim, sim)
    return max_sim


def apply_diversity_penalty(
    property_value: float,
    optimize_direction: str,
    diversity_penalty: float,
    similarity: float,
    similarity_threshold: float,
) -> float:
    """
    Penalize a property value for ranking when the candidate is too similar
    to previously generated molecules. Only the ranking score is penalized;
    the true value should still be displayed and persisted.
    """
    if diversity_penalty <= 0 or similarity < similarity_threshold:
        return property_value
    if optimize_direction == "greater":
        return property_value * (1 - diversity_penalty * similarity)
    return property_value * (1 + diversity_penalty * similarity)


class LMOTask(Task):
    def __init__(
        self,
        lead_molecule: str,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        verification_prompt: Optional[str] = None,
        refinement_prompt: Optional[str] = None,
        property_tool_name: Optional[str] = None,
        property_name: str = "density",
        optimize_direction: str = "greater",
        reference_property_value: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize LMOTask with customizable property optimization.

        Args:
            lead_molecule: SMILES string of the lead molecule
            user_prompt: Custom user prompt (optional)
            system_prompt: Custom system prompt (optional)
            verification_prompt: Custom verification prompt (optional)
            refinement_prompt: Custom refinement prompt (optional)
            property_tool_name: Name of MCP tool that takes SMILES string and returns float property value.
                                Defaults to get_density if None.
            property_name: Name of the property being optimized (for logging)
            optimize_direction: "greater" to maximize property, "less" to minimize it
            reference_property_value: Known property value of the lead molecule. If
                                      provided, get_initial_property_value() skips the
                                      MCP fetch.
            **kwargs: Additional arguments passed to Task
        """

        if property_tool_name is None:
            property_tool_name = "get_density"

        if system_prompt is None:
            system_prompt = build_system_prompt(property_name, optimize_direction)

        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verification_prompt=verification_prompt,
            refinement_prompt=refinement_prompt,
            **kwargs,
        )

        self.lead_molecule = lead_molecule
        self.max_synth_score = smiles_utils.get_synthesizability(lead_molecule)
        self.property_tool_name = property_tool_name
        self.property_name = property_name
        self.optimize_direction = optimize_direction
        self.reference_property_value = reference_property_value
        self.set_structured_output_schema(MoleculeOutputSchema)

    async def _evaluate_property(self, smiles: str) -> float:
        """
        Compute the target property for `smiles` via the property MCP tool.

        Raises:
            ValueError: If the tool is unavailable or returns a non-JSON message.
        """
        property_result_msg = await call_mcp_tool_directly(
            tool_name=self.property_tool_name,
            arguments={
                "smiles": smiles,
                "property": self.property_name,
            },
            urls=self.server_urls or [],
            paths=self.server_files or [],
            bearer_token=self.bearer_token,
        )
        try:
            _, property_value = json.loads(property_result_msg.content)
        except json.decoder.JSONDecodeError:
            msg = (
                f"{self.property_tool_name} returned a bare string, "
                f"not a json message: {property_result_msg}"
            )
            logger.error(msg)
            raise ValueError(msg)
        return property_value

    async def get_initial_property_value(self) -> float:
        """
        Ensure reference_property_value is set, fetching it from the lead
        molecule if it was not supplied at construction time.
        """
        if self.reference_property_value is None:
            try:
                self.reference_property_value = await self._evaluate_property(
                    self.lead_molecule
                )
            except ValueError:
                self.reference_property_value = 0.0
        return self.reference_property_value

    async def check_proposal(self, smiles: str) -> float:
        """
        Check if the proposed SMILES string is valid.
        If it is valid, checks if its synthesizability score is less than or equal to the lead molecule
        and if its property value meets the optimization criteria.

        Args:
            smiles (str): The proposed SMILES string.
        Returns:
            float: The backend-computed property value of the molecule.
        Raises:
            ValueError: If the SMILES string is invalid or does not meet the criteria.
        """
        # NOTE: This must stay deterministic (no LLM calls); it is used for
        # backend-side verification of every candidate.
        if not smiles_utils.verify_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")

        synth_score = smiles_utils.get_synthesizability(smiles)
        if synth_score > self.max_synth_score:
            raise ValueError(
                f"Synthesizability score too high: {synth_score} > {self.max_synth_score}"
            )

        property_value = await self._evaluate_property(smiles)

        if self.optimize_direction == "greater":
            if property_value < self.reference_property_value:
                raise ValueError(
                    f"{self.property_name} too low: {property_value} < {self.reference_property_value}"
                )
        else:  # "less"
            if property_value > self.reference_property_value:
                raise ValueError(
                    f"{self.property_name} too high: {property_value} > {self.reference_property_value}"
                )
        return property_value

    # NOTE: @charge.verifier is only consumed by the legacy
    # charge.clients.client.Client (HVR) path; AgentFrameworkBackend does not
    # introspect it. Kept for forward compatibility with that path.
    @charge.verifier
    async def check_final_proposal(self, smiles_list_as_string: str) -> bool:
        """
        Check if the proposed SMILES strings are valid and meet the criteria.
        The criteria are:
        1. The SMILES must be valid.
        2. The synthesizability score must be less than or equal to the lead molecule.
        3. The property value must meet the optimization criteria (greater or less than reference).

        Args:
            smiles_list_as_string (str): The proposed list of SMILES strings.
        Returns:
            bool: True if the proposal is valid and meets the criteria, False otherwise.

        Raises:
            ValueError: If the output is not a valid list of SMILES strings or if any
                        SMILES string is invalid or does not meet the criteria.
        """
        try:
            try:
                smiles_list = json.loads(smiles_list_as_string)
            except json.JSONDecodeError:
                smiles_list = ast.literal_eval(smiles_list_as_string)
            if not isinstance(smiles_list, list):
                return False
        except (ValueError, SyntaxError):
            raise ValueError("Output is not a valid list of SMILES strings.")

        for smiles in smiles_list:
            await self.check_proposal(smiles)
        return True
