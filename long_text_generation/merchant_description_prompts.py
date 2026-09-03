"""Prompts for merchant description generation."""

from __future__ import annotations

from typing import Mapping


def _value(
    row: Mapping[str, object],
    key: str,
    default: str = "Not provided",
) -> str:
    value = row.get(key, default)
    if value is None:
        return default

    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


def build_target_description_prompt(row: Mapping[str, object]) -> str:
    basic_profile = _value(row, "basic_profile")
    existing_merchant_concepts = _value(
        row,
        "existing_merchant_concepts",
        "No previously generated merchant concepts in this MCC_level2.",
    )

    return f"""You are an assistant for semantic enrichment of financial transaction data. Based on the merchant category, transaction profile, and associated user profile of an anonymous merchant Target, generate an English merchant name and a natural-language merchant description.

Construct a specific, coherent, and restrained merchant instance rather than restating the input fields or translating profile labels one by one.

[Input Information]

{basic_profile}

[Existing Merchant Concepts in This MCC_level2]

{existing_merchant_concepts}

[Information Hierarchy]

1. merchant_category

Consumption_category, MCC_level1, MCC_level2, and fine_grained_candidates form a nested hierarchy: Consumption_category contains MCC_level1, MCC_level1 contains MCC_level2, and fine_grained_candidates lists permitted finer-grained directions under MCC_level2. The generated merchant belongs specifically to its supplied MCC_level2.

MCC_level2 determines the merchant's business type and reasonable product or service scope, while MCC_level1 and Consumption_category define the broader semantic boundary. When fine_grained_candidates contains more specific directions, select one coherent direction for each merchant. Do not combine unrelated candidates or invent a subtype outside the supplied category hierarchy.

2. transaction_profile

Use transaction_profile only to help select a plausible product or service emphasis and broad consumption context within the fixed category. Not every field needs to be used. Do not turn profile signals into concrete operating facts such as pricing policies, schedules, memberships, customer flow, locations, promotions, inventory practices, delivery arrangements, or service procedures, even when such details would be plausible for the merchant category.

3. associated_user_profile

Use associated_user_profile only as weak supplementary context for product or service emphasis and general consumer needs. Treat dominant_gender, dominant_age_group, and dominant_education as weak signals, and occupation_top3 as the lowest-priority signal. Do not express these attributes as explicit customer demographics, occupations, social identities, or surrounding environments.

4. existing_merchant_concepts

Existing merchant concepts contain the names and core business concepts of merchants previously generated within the same MCC_level2. Use this memory only to reduce naming and semantic repetition.

Do not duplicate or closely imitate an existing merchant name.

Before finalizing the current merchant concept, compare it with the existing merchant concepts. Treat concepts as overlapping when they express essentially the same core products or services, product or service combination, broad business format, or central business idea, even if different words are used.

If the current concept substantially overlaps with an existing concept and another plausible direction is available within MCC_level2, revise the current concept toward a meaningfully different product or service emphasis, combination, or broad business format.

Also consider repetition across the existing memory as a whole. If a broader business direction has already appeared repeatedly, avoid using that direction again when other category-consistent concepts remain plausible, even if the new concept would not substantially overlap with any single existing merchant.

When multiple concepts are equally plausible, prefer the one with the least semantic overlap with existing concepts. Common category-inherent products or services may still recur when necessary. Do not introduce unsupported facts or move outside MCC_level2 merely to create differences.

When information conflicts, apply the following priority:

merchant_category > transaction_profile > associated_user_profile > existing_merchant_concepts.

[Generation Requirements]

1. Generate one English merchant name, Merchant, and one English merchant description, Merchant_description.

2. Merchant should sound like a natural name for a fictional small or medium-sized business and clearly fit the specific business type. Avoid overly literary, promotional, or obviously generated names, as well as repeated generic endings such as “House,” “Hub,” “Corner,” “Place,” “Club,” or “Lounge.” Do not use Chinese pinyin, real brands, institutions, platforms, chain businesses, or specific real-world locations.

3. Merchant must not duplicate or closely imitate any merchant name listed in existing_merchant_concepts. Avoid names that differ only by minor spelling changes, word-order changes, or trivial substitutions.

4. Merchant_description should describe the merchant itself. It may include the business format, representative products or services, basic service methods, consumption occasions, business positioning, and consumer needs when they form a coherent merchant instance.

5. Describe the merchant directly; do not explain customer activity, demand patterns, or how the generated characteristics correspond to the input profile.

6. Do not attempt to include every profile field. Do not invent unsupported details such as exact locations, narrowly defined customer groups, staff or teams, facilities, exact operating hours, customer traffic, promotions, membership systems, organizational scale, or management policies.

7. Use natural, fluent, and idiomatic English. Keep the tone neutral and objective. The description must not read like advertising copy, a customer review, a story, or a data analysis report.

8. Do not include field names, profile labels, statistics, or analytical expressions such as Target_id, transaction count, associated user count, Top1, Top2, Top3, average amount, transaction frequency, “according to the data,” or “the profile indicates.”

9. Do not include risk- or label-related terms such as fraud, risk, anomaly, suspicious, money laundering, illicit activity, label, or fraud rate.

10. Merchant_description must be a single English paragraph of 60–80 words. Use English only; do not include Chinese characters or mixed-language expressions. Do not add vague or unsupported content merely to reach the required length.

11. Generate Memory_keywords as 3–5 concise English concept labels separated by semicolons. The keywords should capture the merchant's distinctive core business concept, including its primary product or service emphasis, representative product or service combination, and broad business format or functional purpose when applicable.

12. Memory_keywords should preserve the most distinctive business information needed to recognize semantic overlap in later merchants from the same MCC_level2. Prefer concrete product or service concepts and meaningful combinations over generic category-level terms. The keywords should not merely restate MCC_level2 or use broad labels that could apply to most merchants in the category. Do not include the merchant name, complete sentences, transaction patterns, amount levels, customer demographics, education, occupations, locations, or other profile-derived attributes.

[Output Format]

Output exactly the following three lines and nothing else:

Merchant: English merchant name
Merchant_description: English merchant description
Memory_keywords: keyword one; keyword two; keyword three
"""