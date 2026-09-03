#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manual merchant priors for dataset behavior_level2 labels.

The initialization may assign behavior_level2 almost randomly, so a semantic
optimizer should not learn the category template from the current label groups.
This module keeps those templates as explicit business priors.

The online/offline anchor is refreshed to the 2024 China retail context:
physical-goods online retail represented 26.5% of total retail sales. Scores
therefore encode relative online/remote tendency, not literal online sales
share, and avoid the pandemic-period 2021 uplift used in earlier drafts.
"""

import argparse
import os

import pandas as pd


DEFAULT_INPUT = "consistency_optimization/priors/prior_behavior_level2_table.csv"
DEFAULT_OUTPUT = "work/consistency_optimization/manual_merchant_behavior_level2_priors.csv"
ONLINE_PRIOR_ANCHOR_YEAR = 2024
ONLINE_PRIOR_SOURCE = "NBS 2024 communique: physical-goods online retail share 26.5%; post-pandemic offline/service recovery"

AGE_LABELS = ("Young Adults", "Middle-aged Adults", "Older Adults")
EDUCATION_LABELS = (
    "Primary School",
    "Junior Secondary School",
    "Senior Secondary School",
    "Junior College and Above",
)

INDUSTRY_PRIOR_MAP = {
    "white_collar": ["Financial Services", "Information Transmission, Software and Information Technology Services", "Scientific Research and Technical Services", "Public Administration, Social Security and Social Organizations"],
    "service_worker": ["Accommodation and Food Services", "Resident Services, Repair and Other Services", "Wholesale and Retail Trade"],
    "family_consumer": ["Education", "Health and Social Work", "Public Administration, Social Security and Social Organizations", "Wholesale and Retail Trade"],
    "commuter": ["Manufacturing", "Construction", "Transport, Storage and Postal Services"],
    "student": ["Education"],
    "teacher": ["Education"],
    "parent": ["Education", "Health and Social Work", "Public Administration, Social Security and Social Organizations"],
    "driver": ["Transport, Storage and Postal Services", "Construction", "Manufacturing"],
    "self_employed": ["Wholesale and Retail Trade", "Leasing and Business Services", "Resident Services, Repair and Other Services"],
    "manager": ["Financial Services", "Leasing and Business Services", "Public Administration, Social Security and Social Organizations"],
    "professional": ["Scientific Research and Technical Services", "Financial Services", "Information Transmission, Software and Information Technology Services"],
    "business_owner": ["Leasing and Business Services", "Wholesale and Retail Trade", "Real Estate"],
    "digital_worker": ["Information Transmission, Software and Information Technology Services", "Scientific Research and Technical Services"],
    "patient": ["Health and Social Work", "Public Administration, Social Security and Social Organizations"],
    "family_caregiver": ["Health and Social Work", "Education", "Public Administration, Social Security and Social Organizations"],
    "retiree": ["Public Administration, Social Security and Social Organizations", "Health and Social Work"],
    "tourist": ["Accommodation and Food Services", "Transport, Storage and Postal Services", "Culture, Sports and Entertainment"],
    "business_traveler": ["Transport, Storage and Postal Services", "Leasing and Business Services", "Financial Services"],
    "gamer": ["Information Transmission, Software and Information Technology Services", "Culture, Sports and Entertainment"],
    "content_consumer": ["Culture, Sports and Entertainment", "Information Transmission, Software and Information Technology Services"],
    "home_owner": ["Real Estate", "Construction", "Resident Services, Repair and Other Services"],
    "fitness_enthusiast": ["Culture, Sports and Entertainment", "Health and Social Work"],
}


def bucket(score):
    if score >= 0.75:
        return "strong_online"
    if score >= 0.55:
        return "mixed_online"
    if score >= 0.35:
        return "mixed_offline"
    return "strong_offline"


def hours(*ranges):
    vals = []
    for item in ranges:
        if isinstance(item, tuple):
            vals.extend(range(item[0], item[1] + 1))
        else:
            vals.append(item)
    return ",".join(str(x) for x in sorted(set(vals)))


BASE = {
    "Food and Beverage Services": dict(online=0.18, amount="low_to_medium", vol="medium", h=hours(7, 8, (11, 13), (17, 21)),
                 female=0.50, age=(0.45, 0.38, 0.17), edu=(0.22, 0.38, 0.20, 0.20),
                 occ="white_collar|service_worker|family_consumer"),
    "Local Lifestyle and Self-Service Sharing": dict(online=0.20, amount="medium", vol="medium", h=hours((8, 22)),
                 female=0.55, age=(0.42, 0.42, 0.16), edu=(0.20, 0.38, 0.22, 0.20),
                 occ="family_consumer|service_worker|white_collar"),
    "Local Commuting and On-Demand Travel": dict(online=0.22, amount="low", vol="low_to_medium", h=hours((6, 10), (17, 21)),
                 female=0.48, age=(0.52, 0.35, 0.13), edu=(0.16, 0.36, 0.23, 0.25),
                 occ="commuter|white_collar|student"),
    "Vehicle Energy and Automotive Services": dict(online=0.16, amount="medium", vol="low_to_medium", h=hours((7, 22)),
                 female=0.36, age=(0.33, 0.52, 0.15), edu=(0.16, 0.38, 0.24, 0.22),
                 occ="driver|self_employed|white_collar"),
    "Apparel and General Shopping": dict(online=0.60, amount="medium", vol="medium", h=hours((10, 22)),
                 female=0.62, age=(0.58, 0.32, 0.10), edu=(0.12, 0.30, 0.23, 0.35),
                 occ="white_collar|student|self_employed"),
    "High-Value Goods and Professional Services": dict(online=0.50, amount="high", vol="high", h=hours((9, 20)),
                 female=0.48, age=(0.20, 0.58, 0.22), edu=(0.08, 0.22, 0.25, 0.45),
                 occ="manager|professional|business_owner"),
    "Household and Personal Goods Retail": dict(online=0.55, amount="medium", vol="medium", h=hours((9, 22)),
                 female=0.58, age=(0.46, 0.42, 0.12), edu=(0.14, 0.34, 0.24, 0.28),
                 occ="family_consumer|white_collar|self_employed"),
    "Education Payments and Training Services": dict(online=0.42, amount="high", vol="medium", h=hours((8, 21)),
                 female=0.54, age=(0.20, 0.62, 0.18), edu=(0.08, 0.24, 0.24, 0.44),
                 occ="parent|student|teacher"),
    "Hotels, Scenic Attractions and Tourism Services": dict(online=0.58, amount="high", vol="high", h=hours((6, 23)),
                 female=0.50, age=(0.44, 0.42, 0.14), edu=(0.10, 0.28, 0.24, 0.38),
                 occ="tourist|white_collar|business_traveler"),
    "Housing Payments and Home Improvement Services": dict(online=0.32, amount="high", vol="medium", h=hours((8, 21)),
                 female=0.50, age=(0.18, 0.57, 0.25), edu=(0.14, 0.34, 0.24, 0.28),
                 occ="home_owner|family_consumer|self_employed"),
    "Food, Tobacco and Liquor Retail": dict(online=0.26, amount="low_to_medium", vol="low", h=hours((7, 22)),
                 female=0.50, age=(0.36, 0.42, 0.22), edu=(0.22, 0.40, 0.20, 0.18),
                 occ="family_consumer|service_worker|retiree"),
    "Telecommunications, Logistics and Digital Infrastructure Services": dict(online=0.68, amount="low_to_medium", vol="medium", h=hours((8, 23)),
                 female=0.44, age=(0.56, 0.34, 0.10), edu=(0.10, 0.26, 0.24, 0.40),
                 occ="white_collar|digital_worker|self_employed"),
    "Online Entertainment and Hobby Spending": dict(online=0.88, amount="low_to_medium", vol="high", h=hours((10, 23)),
                 female=0.46, age=(0.72, 0.23, 0.05), edu=(0.08, 0.26, 0.25, 0.41),
                 occ="student|white_collar|digital_worker"),
    "Offline Culture, Sports and Entertainment": dict(online=0.22, amount="low_to_medium", vol="medium", h=hours((9, 23)),
                 female=0.48, age=(0.58, 0.32, 0.10), edu=(0.12, 0.32, 0.24, 0.32),
                 occ="student|white_collar|family_consumer"),
    "Pharmaceutical, Medical Device and Health Retail": dict(online=0.42, amount="low_to_medium", vol="medium", h=hours((8, 21)),
                 female=0.55, age=(0.22, 0.43, 0.35), edu=(0.18, 0.38, 0.22, 0.22),
                 occ="family_caregiver|retiree|white_collar"),
    "Long-Distance Travel Transport": dict(online=0.62, amount="high", vol="high", h=hours((6, 23)),
                 female=0.48, age=(0.40, 0.44, 0.16), edu=(0.10, 0.28, 0.24, 0.38),
                 occ="business_traveler|tourist|driver"),
    "Specialized Consumer Medical and Care Services": dict(online=0.22, amount="high", vol="high", h=hours((8, 19)),
                 female=0.58, age=(0.26, 0.48, 0.26), edu=(0.12, 0.32, 0.24, 0.32),
                 occ="patient|family_caregiver|white_collar"),
    "General Clinical Care and Health Management": dict(online=0.28, amount="medium", vol="medium", h=hours((8, 18)),
                 female=0.52, age=(0.18, 0.42, 0.40), edu=(0.18, 0.38, 0.22, 0.22),
                 occ="patient|retiree|family_caregiver"),
}

GENERIC_BASE = dict(
    online=0.50,
    amount="medium",
    vol="medium",
    h=hours((8, 22)),
    female=0.50,
    age=(0.40, 0.42, 0.18),
    edu=(0.16, 0.34, 0.23, 0.27),
    occ="white_collar|family_consumer|self_employed",
)


def prior_for(category, mcc):
    p = dict(BASE.get(category, GENERIC_BASE))
    basis = ["behavior_level1 base", "2024 online/offline retail anchor"]

    def setp(**kwargs):
        p.update(kwargs)

    # online/digital services
    if any(k in mcc for k in ["Online", "Online", "Internet", "Software", "Information Search", "Online Tools"]):
        setp(online=max(p["online"], 0.78), h=hours((9, 23)), female=min(max(p["female"], 0.45), 0.55),
             age=(0.66, 0.28, 0.06), edu=(0.06, 0.22, 0.24, 0.48),
             occ="digital_worker|white_collar|student")
        basis.append("digital/online keyword")

    if any(k in mcc for k in ["Gaming", "Board Games", "Gaming", "Gaming Cafes"]):
        setp(online=0.84 if "Gaming" in mcc else 0.28, female=0.35, age=(0.76, 0.20, 0.04),
             edu=(0.08, 0.28, 0.26, 0.38), amount="low_to_medium", vol="high",
             h=hours((12, 23)), occ="student|gamer|white_collar")
        basis.append("game/youth male-skew")

    if any(k in mcc for k in ["Live Streaming", "Social Networking", "Video, Audio", "Reading"]):
        setp(online=0.88, female=0.50, age=(0.72, 0.23, 0.05), amount="low_to_medium", vol="high",
             h=hours((10, 23)), occ="student|white_collar|content_consumer")
        basis.append("online content")

    # female/family categories
    if any(k in mcc for k in ["Beauty", "Hair", "Beauty", "Nail", "Medical Aesthetics"]):
        setp(female=0.76 if "Medical Aesthetics" not in mcc else 0.80, age=(0.58, 0.35, 0.07),
             edu=(0.10, 0.28, 0.24, 0.38), amount="medium" if "Medical Aesthetics" not in mcc else "high",
             vol="medium" if "Medical Aesthetics" not in mcc else "high",
             occ="white_collar|student|self_employed")
        basis.append("beauty/female skew")

    if any(k in mcc for k in ["Mother and Baby", "Children's Toys", "Preschool", "Youth Palace"]):
        setp(female=0.68, age=(0.18, 0.70, 0.12), edu=(0.10, 0.28, 0.24, 0.38),
             amount="medium", vol="medium", occ="parent|teacher|family_consumer")
        basis.append("parent/child spending")

    # vehicle and male-skew categories
    if any(k in mcc for k in ["Fuel", "Vehicle", "Car", "Expressway", "Designated Driving", "Parking", "Charging and Battery Swap"]):
        setp(female=0.35, age=(0.32, 0.52, 0.16), edu=(0.14, 0.36, 0.24, 0.26),
             occ="driver|self_employed|business_traveler")
        if "Fuel" in mcc or "Parking" in mcc or "Expressway" in mcc:
            setp(amount="low_to_medium", vol="low_to_medium", online=0.15)
        basis.append("vehicle/male-skew")

    # medical and health
    if any(k in mcc for k in ["Pharmaceutical", "Medical Device", "Health", "Optical", "Nutrition"]):
        setp(female=max(p["female"], 0.55), age=(0.24, 0.42, 0.34),
             edu=(0.16, 0.36, 0.23, 0.25), occ="family_caregiver|retiree|white_collar")
        basis.append("health retail")

    if any(k in mcc for k in ["Dental", "Ophthalmic", "Care Institution", "Clinic", "Hospital", "Health Examination", "Medical Laboratories", "Diagnostic"]):
        setp(female=max(p["female"], 0.54), age=(0.22, 0.46, 0.32), amount="high",
             vol="high" if any(k in mcc for k in ["Dental", "Ophthalmic", "Medical Laboratories", "Diagnostic"]) else "medium",
             edu=(0.12, 0.32, 0.24, 0.32), occ="patient|family_caregiver|retiree")
        basis.append("medical service")

    if "Online Medical" in mcc:
        setp(online=0.78, female=0.56, age=(0.36, 0.46, 0.18), edu=(0.08, 0.25, 0.24, 0.43),
             h=hours((8, 23)), occ="patient|white_collar|family_caregiver")
        basis.append("online medical")

    # travel/hotel/transport
    if any(k in mcc for k in ["Hotel", "Inn", "Homestay", "Travel Agency", "Direct Tourism-Related Services", "Air Travel", "Rail", "Long-Distance Ground Travel", "Water Transport"]):
        setp(online=max(p["online"], 0.62), amount="high", vol="high", female=0.50,
             age=(0.42, 0.44, 0.14), edu=(0.08, 0.26, 0.24, 0.42),
             h=hours((6, 23)), occ="tourist|business_traveler|white_collar")
        basis.append("travel booking")

    if any(k in mcc for k in ["Scenic Attractions", "Amusement Park", "Carnival"]):
        setp(online=min(p["online"], 0.28), amount="medium", vol="high", female=0.50,
             age=(0.46, 0.40, 0.14), edu=(0.12, 0.30, 0.24, 0.34),
             h=hours((8, 21)), occ="tourist|family_consumer|student")
        basis.append("offline attraction")

    # education
    if any(k in mcc for k in ["Higher Education"]):
        setp(age=(0.46, 0.42, 0.12), edu=(0.04, 0.18, 0.24, 0.54), amount="high",
             occ="student|parent|teacher")
        basis.append("higher education")
    elif any(k in mcc for k in ["Education", "Training", "Primary and Secondary Education"]):
        setp(age=(0.18, 0.66, 0.16), edu=(0.08, 0.24, 0.25, 0.43), amount="high",
             occ="parent|student|teacher")
        basis.append("education/training")

    # home/real estate/professional
    if any(k in mcc for k in ["Real Estate", "Housing", "Property Management", "Renovation", "Building Materials", "Hardware", "Home Furnishings", "Home Textiles", "Repair"]):
        setp(age=(0.16, 0.58, 0.26), amount="high" if "Repair" not in mcc else "medium",
             vol="medium", female=0.52 if any(k in mcc for k in ["Home Furnishings", "Home Textiles"]) else 0.48,
             edu=(0.12, 0.34, 0.24, 0.30), occ="home_owner|family_consumer|self_employed")
        basis.append("home/property")

    if any(k in mcc for k in ["Legal", "Accounting", "Financial Consulting", "Advertising", "Exhibition", "Recruitment"]):
        setp(age=(0.18, 0.62, 0.20), edu=(0.04, 0.18, 0.24, 0.54),
             amount="high", vol="high", female=0.48, occ="professional|manager|business_owner")
        basis.append("professional service")

    if "Funeral" in mcc:
        setp(online=0.15, female=0.48, age=(0.10, 0.48, 0.42), edu=(0.20, 0.40, 0.22, 0.18),
             amount="high", vol="high", h=hours((8, 18)), occ="family_consumer|retiree|service_worker")
        basis.append("funeral/offline")

    # retail specifics
    if any(k in mcc for k in ["Convenience", "Fresh Produce", "Food", "Supermarket and Hypermarket", "Tobacco, Liquor", "Tea"]):
        setp(online=0.18 if "Convenience" in mcc else 0.26, amount="low_to_medium", vol="low_to_medium",
             age=(0.34, 0.43, 0.23), edu=(0.22, 0.40, 0.20, 0.18),
             occ="family_consumer|service_worker|retiree")
        basis.append("daily retail")

    if any(k in mcc for k in ["3C", "Digital Electronics", "Home Appliances", "Office Supplies"]):
        setp(online=0.68, female=0.42, age=(0.56, 0.34, 0.10), edu=(0.08, 0.24, 0.24, 0.44),
             amount="high" if "3C" in mcc or "Home Appliances" in mcc else "medium",
             vol="medium", occ="white_collar|student|digital_worker")
        basis.append("electronics/office")

    if any(k in mcc for k in ["Pet"]):
        setp(female=0.58, age=(0.48, 0.40, 0.12), edu=(0.10, 0.28, 0.25, 0.37),
             amount="medium", vol="medium", occ="white_collar|family_consumer|student")
        basis.append("pet economy")

    if any(k in mcc for k in ["Flowers", "Green Plants", "Wedding", "Photography"]):
        setp(female=0.60, age=(0.48, 0.42, 0.10), edu=(0.10, 0.28, 0.25, 0.37),
             amount="medium", vol="high", occ="white_collar|family_consumer|self_employed")
        basis.append("occasion/lifestyle")

    if any(k in mcc for k in ["Apparel", "Footwear and Bags", "Outlet Shopping", "Department Store", "Shopping Mall", "Commercial Street Retail"]):
        setp(female=0.62 if "Apparel" in mcc or "Footwear and Bags" in mcc else 0.56, age=(0.56, 0.34, 0.10),
             edu=(0.12, 0.30, 0.23, 0.35), amount="medium", vol="medium",
             occ="white_collar|student|family_consumer")
        basis.append("fashion/shopping")

    if any(k in mcc for k in ["KTV", "Bar"]):
        setp(female=0.45, age=(0.68, 0.26, 0.06), amount="medium", vol="high",
             h=hours((18, 23)), occ="student|white_collar|service_worker")
        basis.append("night entertainment")

    if any(k in mcc for k in ["Fitness", "Yoga", "Dance", "Outdoor Sports"]):
        setp(female=0.52, age=(0.62, 0.32, 0.06), amount="medium", vol="medium",
             edu=(0.10, 0.28, 0.24, 0.38), occ="white_collar|student|fitness_enthusiast")
        basis.append("sports/fitness")

    if any(k in mcc for k in ["Cultural and Sports Venues", "Cinema", "Performance", "Events", "Books, Media", "Musical Instruments"]):
        setp(female=0.50, age=(0.56, 0.34, 0.10), amount="low_to_medium", vol="medium",
             edu=(0.10, 0.28, 0.24, 0.38), occ="student|white_collar|family_consumer")
        basis.append("culture/entertainment")

    if any(k in mcc for k in ["Public Transport", "Shared Two-Wheel"]):
        setp(online=0.18, amount="low", vol="low", age=(0.58, 0.32, 0.10),
             occ="commuter|student|white_collar")
        basis.append("daily commute")

    if any(k in mcc for k in ["Ride-Hailing"]):
        setp(online=0.40, amount="low_to_medium", vol="medium", age=(0.52, 0.38, 0.10),
             occ="commuter|white_collar|tourist")
        basis.append("ride hailing")

    return p, ";".join(basis)


def parse_hours(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return list(range(8, 23))
    return [int(x) for x in str(value).split(",") if str(x).strip() != ""]


def occupation_industry_distribution(occupation_tokens):
    counts = {}
    tokens = [x for x in str(occupation_tokens).split("|") if x]
    for token in tokens:
        industries = INDUSTRY_PRIOR_MAP.get(token, [])
        if not industries:
            continue
        weight = 1.0 / float(len(industries))
        for industry in industries:
            counts[industry] = counts.get(industry, 0.0) + weight
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def occupation_industry_top3(occupation_tokens):
    dist = occupation_industry_distribution(occupation_tokens)
    if not dist:
        return ""
    top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
    return "|".join(key for key, _ in top)


def format_distribution(dist):
    if not dist:
        return ""
    return "|".join("%s:%.4f" % (key, value) for key, value in sorted(dist.items(), key=lambda x: x[1], reverse=True))


def parse_distribution(value):
    if pd.isna(value):
        return {}
    out = {}
    for part in str(value).split("|"):
        if ":" not in part:
            continue
        key, raw = part.rsplit(":", 1)
        key = key.strip()
        try:
            score = float(raw)
        except Exception:
            continue
        if key:
            out[key] = max(score, 0.0)
    total = sum(out.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in out.items()}


def manual_prior_to_business_prior(row):
    female = float(row["female_ratio_prior"])
    occupation_dist = parse_distribution(row.get("occupation_industry_distribution_prior", ""))
    if not occupation_dist:
        occupation_dist = occupation_industry_distribution(row["occupation_top3_prior"])
    return {
        "online_score": float(row["online_score"]),
        "online_bucket": row["online_bucket"],
        "amount_preference": row["amount_preference"],
        "amount_volatility": row["amount_volatility"],
        "active_hours": parse_hours(row["active_hours"]),
        "gender_distribution": {
            "Female": female,
            "Male": max(0.0, 1.0 - female),
        },
        "age_distribution": {
            AGE_LABELS[0]: float(row["age_young_adults_prior"]),
            AGE_LABELS[1]: float(row["age_middle_aged_adults_prior"]),
            AGE_LABELS[2]: float(row["age_older_adults_prior"]),
        },
        "education_distribution": {
            EDUCATION_LABELS[0]: float(row["education_primary_school_prior"]),
            EDUCATION_LABELS[1]: float(row["education_junior_secondary_prior"]),
            EDUCATION_LABELS[2]: float(row["education_senior_secondary_prior"]),
            EDUCATION_LABELS[3]: float(row["education_junior_college_above_prior"]),
        },
        "occupation_distribution": occupation_dist,
        "rationale": row.get("prior_basis", "manual business prior"),
    }


def build_manual_prior_table(category_mcc_pairs, category_col="behavior_level1", mcc_col="behavior_level2"):
    rows = []
    pairs = (
        category_mcc_pairs[[category_col, mcc_col]]
        .drop_duplicates()
        .sort_values([category_col, mcc_col])
        .to_dict("records")
    )
    for row in pairs:
        category = row[category_col]
        mcc = row[mcc_col]
        p, basis = prior_for(category, mcc)
        ay, am, ao = p["age"]
        ep, ej, es, ec = p["edu"]
        occupation_dist = occupation_industry_distribution(p["occ"])
        rows.append(
            {
                category_col: category,
                mcc_col: mcc,
                "online_prior_anchor_year": ONLINE_PRIOR_ANCHOR_YEAR,
                "online_prior_source": ONLINE_PRIOR_SOURCE,
                "online_score": round(float(p["online"]), 3),
                "online_bucket": bucket(float(p["online"])),
                "amount_preference": p["amount"],
                "amount_volatility": p["vol"],
                "active_hours": p["h"],
                "female_ratio_prior": round(float(p["female"]), 3),
                "age_young_adults_prior": round(float(ay), 3),
                "age_middle_aged_adults_prior": round(float(am), 3),
                "age_older_adults_prior": round(float(ao), 3),
                "education_primary_school_prior": round(float(ep), 3),
                "education_junior_secondary_prior": round(float(ej), 3),
                "education_senior_secondary_prior": round(float(es), 3),
                "education_junior_college_above_prior": round(float(ec), 3),
                "occupation_top3_prior": p["occ"],
                "occupation_industry_top3_prior": occupation_industry_top3(p["occ"]),
                "occupation_industry_distribution_prior": format_distribution(occupation_dist),
                "prior_basis": basis,
            }
        )
    return pd.DataFrame(rows)


def build_manual_prior_dict(table, category_col="behavior_level1", mcc_col="behavior_level2"):
    out = {}
    for row in table.to_dict("records"):
        prior = manual_prior_to_business_prior(row)
        out[row[mcc_col]] = prior
        out["%s||%s" % (row[category_col], row[mcc_col])] = prior
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = build_manual_prior_table(df)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(args.output)
    print(out.shape)


if __name__ == "__main__":
    main()
