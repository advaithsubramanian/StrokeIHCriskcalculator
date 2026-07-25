"""
South Asian Stroke and Intracerebral Hemorrhage (ICH) Risk Calculator
=======================================================================

Reference Python implementation. This module reproduces exactly the logic
used by the accompanying browser calculator (south_asian_stroke_calculator.html)
and is provided so the underlying math can be inspected, tested, and reused
independently of a browser.

Target population: South Asian-origin adults living in Western, high-income
healthcare systems (the diaspora population sampled by the MASALA, UK
Biobank, and SABRE cohorts this calculator's coefficients are drawn from).
This tool is NOT validated for people currently resident in South Asia; see
Khan et al. (2017) for evidence that diaspora and resident stroke incidence
differ substantially.

Architecture: two linked models.

  MODEL A - General cardiovascular risk context: the standard ACC/AHA Pooled
      Cohort Equations (PCE, White-cohort coefficients; no South Asian PCE
      coefficient set exists), recalibrated with a South Asian-specific
      multiplier. Gives a familiar clinical comparison point and quantifies
      the PCE's known under-prediction directly.

  MODEL B - Stroke/ICH subtype-specific model: a multiplicative relative-risk
      score built from INTERSTROKE's South Asia-region, subtype-specific
      (ischemic vs. ICH) odds ratios, anchored to a South Asian diaspora
      baseline incidence rate (Khan et al. 2017). This is the calculator's
      primary, more novel output.

PCE coefficients source (White men/women only, verified against the source
report's own worked example): Goff DC Jr, Lloyd-Jones DM, Bennett G, et al.
"2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk." Circulation.
2014;129(25 Suppl 2):S49-S73. The worked-example check below (_validate_pce)
reproduces the report's own published answer (White man, age 55, total
cholesterol 213, HDL 50, untreated systolic blood pressure 120, nonsmoker, no
diabetes -> 5.3%).

KNOWN GAPS AND APPROXIMATIONS IN THIS VERSION:
  - No South Asia-specific family history coefficient was located anywhere in
    the literature reviewed (confirmed absent across three independent
    sources: Gunarathne et al. 2009, Khan et al. 2013, and Vyas et al. 2021),
    so family history is not included as a model input.
  - Khan et al. (2017) reports only a single age/sex-standardized population
    incidence rate for ischemic stroke and ICH, not an age-stratified curve.
    Age-scaling here combines that baseline rate with a generic
    doubling-per-decade trend and a subtype-specific shape modifier derived
    from Vyas et al. (2021), which reports age-banded hazard ratios
    (immigrant vs. long-term Canadian resident) separately for ischemic
    stroke and ICH. This is real, subtype-specific, age-stratified data from
    a diaspora population, but it is pooled across all immigrant source
    regions, not South Asian-specific, so it should be read as an informed
    approximation rather than a directly measured South Asian age-incidence
    curve. See age_scale_factor() below.
  - INTERSTROKE's odds ratios are case-control estimates, treated here as
    approximating hazard ratios under the rare-disease assumption, standard
    practice for stroke research given its low absolute incidence.
"""

import math

# ---------------------------------------------------------------------------
# MODEL A: Pooled Cohort Equations (White coefficients) + South Asian layer
# ---------------------------------------------------------------------------

# Table 4, Goff et al. 2013 ACC/AHA Full Work Group Report (White cohort only;
# the PCE has no South Asian-specific coefficient set, and current ACC/AHA
# guidance is to use the White coefficients as the closest available base
# population).
PCE_COEF = {
    "male": {
        "ln_age": 12.344,
        "ln_age_sq": 0.0,
        "ln_totchol": 11.853,
        "ln_age_totchol": -2.664,
        "ln_hdl": -7.990,
        "ln_age_hdl": 1.769,
        "ln_treated_sbp": 1.797,
        "ln_age_treated_sbp": 0.0,
        "ln_untreated_sbp": 1.764,
        "ln_age_untreated_sbp": 0.0,
        "smoker": 7.837,
        "ln_age_smoker": -1.795,
        "diabetes": 0.658,
        "group_mean": 61.18,
        "baseline_survival": 0.9144,
    },
    "female": {
        "ln_age": -29.799,
        "ln_age_sq": 4.884,
        "ln_totchol": 13.540,
        "ln_age_totchol": -3.114,
        "ln_hdl": -13.578,
        "ln_age_hdl": 3.149,
        "ln_treated_sbp": 2.019,
        "ln_age_treated_sbp": 0.0,
        "ln_untreated_sbp": 1.957,
        "ln_age_untreated_sbp": 0.0,
        "smoker": 7.574,
        "ln_age_smoker": -1.665,
        "diabetes": 0.661,
        "group_mean": -29.18,
        "baseline_survival": 0.9665,
    },
}

# South Asian recalibration multiplier: a risk-dependent curve rather than a
# flat multiplier, fit through two calibration points from two independent
# South Asian cohorts.
#   Anchor 1 - Pursnani et al. (2022), Kaiser Permanente Northern California,
#       low-PCE-risk stratum: PCE predicted 1.8%, observed 4.9%
#       -> multiplier needed = 2.72
#   Anchor 2 - Patel et al. (2021), UK Biobank, cohort average:
#       PCE predicted 4.8%, observed 6.8% -> multiplier needed = 1.42
# A flat multiplier was tested first and found to undershoot the low-risk
# Pursnani stratum by roughly 47%, since a single multiplier cannot match
# both a 2.72-fold and a 1.42-fold correction at different points on the risk
# spectrum; this two-point curve replaces it.
#
# A separate, third cohort (Rodriguez et al. 2019, a Northern California EHR
# cohort disaggregating Asian Indian participants specifically, n=13,815)
# found the opposite direction: the PCE overestimated risk in Asian Indians
# by approximately 30% (predicted 1.2% vs. observed 0.9%). Re-reading
# Pursnani et al.'s own data, their never-on-statin subgroup, the closest
# comparator to Rodriguez et al.'s not-yet-on-statin cohort, showed a much
# weaker, non-significant gap (predicted 1.4% vs. observed 2.1%, p=0.12) than
# the statin-inclusive headline figure used as the anchor below, raising the
# possibility that part of that headline figure reflects confounding by
# treatment indication rather than a pure calibration failure. This is a
# genuine, unresolved disagreement in the literature about which direction
# the correction should point, discussed at length in the accompanying
# manuscript (Discussion, "Direction of PCE Miscalibration in South Asians").
# This multiplier is retained as one reading of the evidence consistent with
# two of the three cohorts, not a settled correction factor.
_CALIB_ANCHORS = [
    (1.8, 2.72),   # (PCE-predicted %, multiplier needed) - Pursnani et al. 2022
    (4.8, 1.42),   # Patel et al. 2021
]


def south_asian_multiplier(pce_predicted_pct):
    """Two-point calibration curve (see note above). Interpolates in
    log-log space between the two known anchors; outside that range, holds
    flat at the nearest anchor's value rather than extrapolating a fitted
    curve indefinitely, a deliberately conservative choice given that only
    two calibration points are available."""
    lo_pct, lo_mult = _CALIB_ANCHORS[0]
    hi_pct, hi_mult = _CALIB_ANCHORS[1]
    if pce_predicted_pct <= lo_pct:
        return lo_mult
    if pce_predicted_pct >= hi_pct:
        return hi_mult
    # log-log linear interpolation (equivalent to a power-law fit through
    # both anchors)
    x = math.log(pce_predicted_pct)
    x0, x1 = math.log(lo_pct), math.log(hi_pct)
    y0, y1 = math.log(lo_mult), math.log(hi_mult)
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return math.exp(y)

# Country-of-origin ratios, derived from Patel et al. (2021) UK Biobank
# ethnic-subgroup hazard ratios versus a White European reference
# (Bangladeshi 3.66, Pakistani 2.45, other South Asian 2.41, Indian 2.08),
# each divided by that cohort's overall South Asian composite hazard ratio
# (2.03) so the ratio is centered on 1.00 for an unspecified/pooled origin.
# Applied proportionally to the recalibration multiplier above; an explicit,
# documented approximation, not a directly measured quantity.
COUNTRY_RATIO = {
    "pooled": 1.00,
    "bangladeshi": 1.80,
    "pakistani": 1.21,
    "other_south_asian": 1.19,
    "indian": 1.02,
}


def pce_10y_risk_pct(sex, age, totchol, hdl, sbp, bp_treated, smoker, diabetes):
    """Standard ACC/AHA Pooled Cohort Equations, White coefficients."""
    c = PCE_COEF[sex]
    ln_age = math.log(age)
    ln_totchol = math.log(totchol)
    ln_hdl = math.log(hdl)
    ln_sbp = math.log(sbp)
    smoker = 1 if smoker else 0
    diabetes = 1 if diabetes else 0

    s = (
        ln_age * c["ln_age"]
        + (ln_age ** 2) * c["ln_age_sq"]
        + ln_totchol * c["ln_totchol"]
        + ln_age * ln_totchol * c["ln_age_totchol"]
        + ln_hdl * c["ln_hdl"]
        + ln_age * ln_hdl * c["ln_age_hdl"]
        + smoker * c["smoker"]
        + ln_age * smoker * c["ln_age_smoker"]
        + diabetes * c["diabetes"]
    )
    if bp_treated:
        s += ln_sbp * c["ln_treated_sbp"] + ln_age * ln_sbp * c["ln_age_treated_sbp"]
    else:
        s += ln_sbp * c["ln_untreated_sbp"] + ln_age * ln_sbp * c["ln_age_untreated_sbp"]

    risk = 1 - (c["baseline_survival"] ** math.exp(s - c["group_mean"]))
    return round(risk * 100, 2)


def model_a_south_asian_ascvd_risk(sex, age, totchol, hdl, sbp, bp_treated,
                                    smoker, diabetes, country="pooled"):
    """MODEL A: PCE baseline recalibrated for South Asian ethnicity, using
    the risk-dependent two-point calibration curve above rather than a flat
    multiplier."""
    base = pce_10y_risk_pct(sex, age, totchol, hdl, sbp, bp_treated, smoker, diabetes)
    base_multiplier = south_asian_multiplier(base)
    multiplier = base_multiplier * COUNTRY_RATIO[country]
    recalibrated = round(min(base * multiplier, 99.9), 2)
    return {"pce_uncorrected_pct": base, "south_asian_recalibrated_pct": recalibrated,
            "base_multiplier_before_country_adj": round(base_multiplier, 2),
            "multiplier_applied": round(multiplier, 2),
            "delta_percentage_points": round(recalibrated - base, 2),
            "risk_category": ascvd_category(recalibrated),
            "confidence": model_a_confidence(country)}


# Standard ACC/AHA PCE risk tiers (<5%, 5-<7.5%, 7.5-<20%, >=20%), applied to
# the recalibrated figure rather than the raw PCE output, since the
# recalibrated figure is this tool's primary estimate.
def ascvd_category(recalibrated_pct):
    if recalibrated_pct < 5:
        return "Low"
    if recalibrated_pct < 7.5:
        return "Borderline"
    if recalibrated_pct < 20:
        return "Intermediate"
    return "High"


def model_a_confidence(country):
    """A single, persistently-reported confidence label (rather than a
    warning shown only conditionally) so the tool's uncertainty is always
    visible, not just flagged on specific inputs."""
    if country != "pooled":
        return {
            "label": "Lower",
            "note": ("Country-of-origin refinement applied: compounds a "
                     "single-cohort-derived factor on top of the general "
                     "recalibration."),
        }
    return {
        "label": "Moderate",
        "note": "Based on two independent South Asian cohort predicted-vs-observed comparisons.",
    }


# ---------------------------------------------------------------------------
# MODEL B: INTERSTROKE-derived stroke/ICH-specific multiplicative risk score
# ---------------------------------------------------------------------------

# O'Donnell et al. 2016 (INTERSTROKE), South Asia-region, subtype-specific
# odds ratios. NS = not statistically significant in this dataset, included
# anyway because the direction is consistent across every South Asian source
# reviewed even where a given split isn't significant; treat outputs built
# from these factors with appropriate caution.
INTERSTROKE_SA_OR = {
    "hypertension":  {"ischemic": 3.61, "ich": 4.37},   # both significant
    "smoking":       {"ischemic": 1.32, "ich": 1.35},   # not significant, either subtype
    "diabetes":      {"ischemic": 1.16, "ich": 3.23},   # ischemic NS; ICH significant but based on small numbers
    "high_whr":      {"ischemic": 1.47, "ich": 2.80},   # not significant, either subtype; ICH CI very wide
}

# Khan et al. 2017, South Asian diaspora, age/sex-standardized incidence per
# 100,000 person-years (Ontario and British Columbia, most recent study
# year, 2010).
BASELINE_INCIDENCE_PER_100K = {
    "ischemic": 39.7,
    "ich": 5.0,
}
BASELINE_REFERENCE_AGE = 55  # approximate mean age across the MASALA, UK Biobank, and SABRE cohorts

# Generic age-scaling trend: stroke incidence roughly doubles per decade of
# age beyond the reference age. A standard, ethnicity-blind epidemiological
# heuristic, used as the base trend below.
AGE_DOUBLING_DECADE = 10.0

# Subtype-specific age shape from Vyas et al. (2021), Figure 2: adjusted
# hazard ratio, immigrants vs. long-term Ontario residents, by age band. This
# is real age-stratified, stroke-subtype-specific data from a diaspora
# population, but it is pooled across all immigrant source regions, not
# South Asian-specific. A hazard ratio near 1.0 across ages does not mean
# risk is flat with age, it means immigrants and residents rise together at
# that age, so this curve is used as a normalized shape modifier layered on
# top of the generic doubling trend above, not as a replacement for it.
_VYAS_AGE_BAND_MIDPOINTS = [24, 35, 45, 55, 65, 75, 85]  # bands: 18-29, 30-39, 40-49, 50-59, 60-69, 70-79, >=80
_VYAS_IMMIGRANT_HR_BY_AGE = {
    "ischemic": [0.84, 0.68, 0.69, 0.78, 0.82, 0.74, 0.59],
    "ich":      [1.01, 1.06, 0.98, 1.04, 1.03, 0.86, 0.69],
}
_VYAS_REF_BAND_INDEX = 3  # 50-59 band, midpoint 55 == BASELINE_REFERENCE_AGE


def _immigrant_age_shape(age, subtype):
    """Piecewise-linear interpolation of the Vyas et al. (2021) age-banded
    hazard-ratio curve, normalized to the 50-59 band (which contains the
    model's reference age). Held flat outside the observed range (age
    24-85)."""
    xs = _VYAS_AGE_BAND_MIDPOINTS
    ys = _VYAS_IMMIGRANT_HR_BY_AGE[subtype]
    ref_hr = ys[_VYAS_REF_BAND_INDEX]
    if age <= xs[0]:
        raw = ys[0]
    elif age >= xs[-1]:
        raw = ys[-1]
    else:
        raw = ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= age <= xs[i + 1]:
                frac = (age - xs[i]) / (xs[i + 1] - xs[i])
                raw = ys[i] + frac * (ys[i + 1] - ys[i])
                break
    return raw / ref_hr


def age_scale_factor(age, subtype):
    """Combines the generic doubling-per-decade trend with the Vyas et al.
    (2021) subtype-specific immigrant age-shape modifier (see note above)."""
    decades_from_ref = (age - BASELINE_REFERENCE_AGE) / AGE_DOUBLING_DECADE
    generic = 2 ** decades_from_ref
    shape = _immigrant_age_shape(age, subtype)
    return generic * shape


# Multiplying independent risk-factor odds ratios together assumes each
# factor's effect is uncorrelated with the others, which is not realistic:
# hypertension, diabetes, and high waist-to-hip ratio are correlated
# conditions with overlapping causal pathways. No South Asian-specific
# multivariable (jointly estimated) stroke risk model was located in the
# literature reviewed; INTERSTROKE publishes population attributable risk
# for combined factors, not per-person joint coefficients, so it cannot
# directly replace the multiplicative approach used here. Rather than invent
# an unvalidated dampening constant, this flags reduced confidence when three
# or more major risk factors stack, so the output isn't presented with false
# precision.
CONFIDENCE_FLAG_THRESHOLD = 3


def model_b_stroke_ich_risk(age, sex, hypertension, smoker, diabetes, high_whr,
                             country="pooled", years=10):
    """MODEL B: multiplicative INTERSTROKE relative-risk score anchored to
    Khan et al. (2017) South Asian diaspora baseline incidence."""
    factors = {
        "hypertension": hypertension,
        "smoking": smoker,
        "diabetes": diabetes,
        "high_whr": high_whr,
    }
    n_factors_present = sum(1 for v in factors.values() if v)
    results = {}
    for subtype in ("ischemic", "ich"):
        rr = 1.0
        for factor, present in factors.items():
            if present:
                rr *= INTERSTROKE_SA_OR[factor][subtype]
        annual_per_100k = (
            BASELINE_INCIDENCE_PER_100K[subtype]
            * rr
            * age_scale_factor(age, subtype)
            * COUNTRY_RATIO[country]
        )
        # Converts an annual rate into an approximate N-year cumulative risk
        # (1 - exp(-rate*years)); adequate at these low absolute rates.
        annual_rate = annual_per_100k / 100000.0
        cumulative_risk_pct = round((1 - math.exp(-annual_rate * years)) * 100, 3)
        results[subtype] = {
            "relative_risk_vs_no_factors": round(rr, 2),
            f"{years}y_risk_pct": cumulative_risk_pct,
            "risk_category": stroke_category(cumulative_risk_pct),
            "confidence": model_b_confidence(n_factors_present, country),
        }
    return results


# Descriptive bands only (<1%, 1-<3%, >=3%), chosen to make the output
# legible. Unlike Model A's PCE bands, no validated clinical risk tier exists
# for South Asian stroke-subtype risk; these labels are interpretive, not
# diagnostic, and should not be read as equivalent in rigor to ascvd_category().
def stroke_category(risk_pct):
    if risk_pct < 1:
        return "Low"
    if risk_pct < 3:
        return "Moderate"
    return "Elevated"


def model_b_confidence(n_factors_present, country):
    """Mirrors model_a_confidence(): a persistently-reported label rather
    than a warning that only appears conditionally."""
    notes = []
    label = "Moderate"
    if n_factors_present >= CONFIDENCE_FLAG_THRESHOLD:
        label = "Low"
        notes.append("Three or more risk factors present; multiplicative combination "
                     "likely overstates true risk.")
    elif country != "pooled":
        label = "Lower"
    if country != "pooled":
        notes.append("Country-of-origin refinement applied; adds a single-cohort-derived "
                     "compounding factor.")
    return {"label": label, "note": " ".join(notes)}


# ---------------------------------------------------------------------------
# Validation of Model A against the ACC/AHA report's own published example
# ---------------------------------------------------------------------------

def _validate_pce():
    # White man, 55y, TC 213, HDL 50, untreated SBP 120, nonsmoker, no
    # diabetes -> ACC/AHA report states 5.3%. Full-precision recomputation
    # here gives 5.38% (0.08 percentage points off) because the report's own
    # worked example rounds intermediate ln() values to two decimals before
    # multiplying; using full precision throughout (the correct approach for
    # a real implementation, and what other independent PCE reproductions,
    # e.g. the CVrisk R package, also do) gives a very slightly different but
    # more accurate answer. Tolerance is set to 0.15 points to accommodate
    # this known, well-documented rounding quirk rather than chasing the
    # report's own rounding.
    r = pce_10y_risk_pct("male", 55, 213, 50, 120, bp_treated=False, smoker=False, diabetes=False)
    assert abs(r - 5.3) < 0.15, f"PCE validation failed: got {r}, expected ~5.3"
    # White woman, same profile -> report states 2.1%.
    r2 = pce_10y_risk_pct("female", 55, 213, 50, 120, bp_treated=False, smoker=False, diabetes=False)
    assert abs(r2 - 2.1) < 0.15, f"PCE validation failed: got {r2}, expected ~2.1"
    print(f"PCE validation OK: male={r}% (report: 5.3%), female={r2}% (report: 2.1%)")


if __name__ == "__main__":
    _validate_pce()
    print()

    print("=== Example 1: 55-year-old South Asian man, hypertensive, diabetic, non-smoker, average WHR ===")
    a1 = model_a_south_asian_ascvd_risk(
        sex="male", age=55, totchol=213, hdl=45, sbp=145, bp_treated=True,
        smoker=False, diabetes=True, country="pooled")
    b1 = model_b_stroke_ich_risk(
        age=55, sex="male", hypertension=True, smoker=False, diabetes=True,
        high_whr=False, country="pooled")
    print("Model A (general ASCVD context):", a1)
    print("Model B (stroke/ICH-specific):   ", b1)
    print()

    print("=== Example 2: same profile, Bangladeshi-origin refinement applied ===")
    a2 = model_a_south_asian_ascvd_risk(
        sex="male", age=55, totchol=213, hdl=45, sbp=145, bp_treated=True,
        smoker=False, diabetes=True, country="bangladeshi")
    b2 = model_b_stroke_ich_risk(
        age=55, sex="male", hypertension=True, smoker=False, diabetes=True,
        high_whr=False, country="bangladeshi")
    print("Model A (general ASCVD context):", a2)
    print("Model B (stroke/ICH-specific):   ", b2)
    print()

    print("=== Example 3: 65-year-old South Asian woman, no major risk factors (baseline comparison) ===")
    a3 = model_a_south_asian_ascvd_risk(
        sex="female", age=65, totchol=190, hdl=55, sbp=118, bp_treated=False,
        smoker=False, diabetes=False, country="pooled")
    b3 = model_b_stroke_ich_risk(
        age=65, sex="female", hypertension=False, smoker=False, diabetes=False,
        high_whr=False, country="pooled")
    print("Model A (general ASCVD context):", a3)
    print("Model B (stroke/ICH-specific):   ", b3)
    print()
