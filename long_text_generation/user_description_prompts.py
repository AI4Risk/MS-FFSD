"""Prompts for user description generation."""

from __future__ import annotations

import json
from typing import Mapping


def _value(row: Mapping[str, object], key: str, default: str = "Not provided") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


def build_llm_input(row: Mapping[str, object]) -> str:
    """Return only the fields authorized for multi-transaction LLM generation."""
    basic_profile = _value(row, "basic_profile")
    try:
        profile = json.loads(basic_profile)
    except json.JSONDecodeError as exc:
        raise ValueError("basic_profile must be valid JSON for LLM prompt construction.") from exc
    if not isinstance(profile, Mapping):
        raise ValueError("basic_profile must contain a JSON object.")

    attributes = profile.get("user_attributes", {})
    transaction_profile = profile.get("transaction_profile", {})
    spending_preferences = profile.get("spending_preferences", {})
    if not isinstance(attributes, Mapping) or not isinstance(transaction_profile, Mapping) or not isinstance(spending_preferences, Mapping):
        raise ValueError("basic_profile contains malformed user profile sections.")

    return json.dumps(
        {
            "Source_id": _value(profile, "Source_id"),
            "Gender": _value(attributes, "Gender"),
            "transaction_profile": transaction_profile,
            "spending_preferences": spending_preferences,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_source_description_prompt(row: Mapping[str, object]) -> str:
    basic_profile = build_llm_input(row)
    return f"""You are an assistant for semantic enrichment of financial transaction data. Based on the gender, transaction profile, and spending preferences of an anonymous user Source, generate a natural-language English description of the user's overall consumption behavior.

Construct a coherent and evidence-grounded description that combines explicit consumption interests with higher-level behavioral semantics. Do not mechanically translate the input profile into natural language.

[Input Information]

{basic_profile}

[Profile Label Interpretation]

The monthly_pattern labels have the following meanings and must be interpreted exactly as defined below:

* early-month pattern: transaction activity is relatively concentrated near the beginning of each month.
* late-month pattern: transaction activity is relatively concentrated near the end of each month.
* early-month pattern; late-month pattern: transaction activity is relatively concentrated near both the beginning and the end of each month.
* monthly-cycle pattern: transaction activity shows a recurring concentration around common bill-payment periods, without a dominant beginning-of-month or end-of-month pattern.
* weak monthly pattern: transaction activity does not show a pronounced concentration at a particular stage of the monthly cycle.

These definitions are provided only to clarify the meaning of the profile labels. Do not infer salary cycles, budgeting behavior, liquidity conditions, payment obligations, or other real-life causes from monthly_pattern.

[Information Usage]

1. spending_preferences

preferred_categories defines the main consumption categories associated with the user.

The supplied categories may be explicitly mentioned and naturally combined in Source_description. When multiple categories are provided, organize them into a coherent description of the user's consumption scope and category composition rather than listing them independently.

Keep the description within the semantic scope of the supplied categories. Do not extend category meanings into inferred consumption purposes, personal needs, lifestyle meanings, values, motivations, or real-life roles.

2. transaction_profile

transaction_profile is reasoning evidence only.

Use combinations of its behavioral signals to derive higher-level structural characteristics of the user's consumption behavior. The resulting semantics should describe relationships among multiple behavioral dimensions rather than reproduce any individual behavioral signal.

Do not directly state, restate, paraphrase, summarize, or generalize the original transaction-profile information. Do not convert behavioral patterns into inferred planning, decision-making, personal routines, preferences, lifestyle characteristics, financial management, or real-life explanations.

Output only the higher-level structural semantics obtained from jointly considering multiple behavioral signals.

3. Gender

Gender is weak supplementary context only.

Do not explicitly mention or paraphrase gender in Source_description. Gender must not independently determine any generated characteristic or be used to infer unsupported personal or behavioral attributes.

[Generation Requirements]

1. Generate one English Source_description describing the user's overall consumption profile over the full observation period.

2. The description should combine:

   * the user's supplied consumption categories and their overall category composition;
   * higher-level structural behavioral semantics derived from multiple transaction-profile signals.

3. Preferred consumption categories may be stated directly, but transaction-profile information must remain implicit.

4. Do not reproduce or indirectly reveal individual transaction-profile signals through equivalent natural-language expressions.

5. State derived semantic characteristics directly as properties of the user's consumption profile. Do not describe evidence, reasoning, implications, interpretations, or relationships between observations and conclusions.

6. Keep all derived content at the level of consumption structure and behavioral organization. Do not introduce inferred purposes, motivations, intentions, personality, lifestyle, personal circumstances, financial conditions, or real-life explanations.

7. Do not introduce information that cannot be reasonably supported by the supplied profile.

8. Do not include demographic information, field names, profile labels, statistics, exact numerical values, risk-related information, or meta-analytical language.

9. Use natural, fluent, neutral, and objective English. Avoid analytical or inferential phrasing. The text should read as a direct description of the user's consumption profile.

10. Avoid generic or repetitive statements. Preserve meaningful distinctions among users without inventing unsupported differences.

11. Source_description must be a single English paragraph of 45–65 words. Use English only.

[Output Format]

Output exactly the following line and nothing else:

Source_description: English user description
"""
