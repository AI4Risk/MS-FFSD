"""Prompts for duplicate virtual merchant-name resolution."""

from __future__ import annotations


def build_merchant_rename_prompt(
    old_merchant: str,
    merchant_description: str,
) -> str:
    return f"""You are a merchant naming assistant. The merchant record below must be renamed because its current Merchant name is duplicated. Based on the original merchant name and merchant description, generate a new English merchant name to replace only the Merchant field.

[Input Information]

Original merchant name:
{old_merchant}

Merchant description:
{merchant_description}

[Requirements]

1. The new merchant name must be consistent with the business type, products or services, and overall semantics described in the merchant description.

2. The new merchant name must be meaningfully different from the original merchant name. Do not merely add numbers, letters, serial identifiers, branch labels, location labels, or similar suffixes or prefixes.

3. Generate a natural English name suitable for a fictional small or medium-sized merchant. Use naming patterns appropriate to the merchant's specific business type.

4. Do not use Chinese characters, Chinese pinyin, real brands, real platforms, real institutions, real chain businesses, or specific real-world place names.

5. Avoid overly literary, promotional, exaggerated, or obviously generated names. Do not rely excessively on generic endings such as “House,” “Hub,” “Corner,” “Place,” “Club,” or “Lounge.”

6. Generate only a replacement merchant name. Do not rewrite, summarize, or otherwise modify the merchant description.

[Output Format]

Output exactly the following JSON object and nothing else:

{{
  "new_merchant": "New English merchant name"
}}
"""