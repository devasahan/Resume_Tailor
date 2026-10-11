import pytest

from app.pipeline import validate
from app.schemas import (
    ExperienceOut, JDAnalysis, SkillGroup, TailoredSections, TailorOutput,
)


def make_output(n=3, **overrides):
    sections = dict(
        headline="Senior Data Engineer | Python | SQL",
        summary="Senior Data Engineer with experience in Python and SQL.",
        skills=[SkillGroup(category="Core", skills=["Python", "SQL", "python"])],
        experience=[
            ExperienceOut(company_index=i, job_title=f"Engineer {i}", bullets=[f"• Built things {'abc'[i]}", "- Improved things"])
            for i in range(n)
        ],
    )
    sections.update(overrides)
    analysis = JDAnalysis(
        target_role="Senior Data Engineer", seniority="Senior", domain="Data",
        must_have_skills=["Python", "SQL"], nice_to_have_skills=["Spark"], keywords=["Python", "SQL", "Airflow"],
        key_responsibilities=["Build pipelines"],
    )
    return TailorOutput(jd_analysis=analysis, sections=TailoredSections(**sections))


def test_build_resume_keeps_fixed_fields_and_cleans_text(profile):
    resume = validate.build_resume(profile, make_output())
    assert resume.profile == profile
    assert resume.experience[0].bullets[0] == "Built things a"  # bullet glyph stripped
    assert resume.skills[0].skills == ["Python", "SQL"]  # duplicate removed, case-insensitive


def test_experience_must_match_companies(profile):
    out = make_output(n=2)
    with pytest.raises(ValueError, match="exactly one entry"):
        validate.build_resume(profile, out)
    out = make_output()
    out.sections.experience[1].company_index = 0
    with pytest.raises(ValueError):
        validate.build_resume(profile, out)


def test_fixed_field_tampering_is_detected(profile):
    resume = validate.build_resume(profile, make_output())
    resume.profile.companies[0].end = "Dec 2030"
    with pytest.raises(ValueError, match="altered"):
        validate.assert_fixed_fields_intact(profile, resume)


def test_clean_text_removes_symbols_and_emoji_but_keeps_headline_separator():
    assert validate.clean_text("Led 🚀 team — “fast”\t work |  Python") == 'Led team - "fast" work | Python'


def test_claim_flags_catch_unsupplied_numbers_and_inferred_skills(profile, jd):
    out = make_output()
    out.sections.experience[0].bullets = ["Cut costs by 30% across teams", "Built pipelines in Python"]
    out.sections.skills = [SkillGroup(category="Core", skills=["Python", "Kubernetes"])]
    resume = validate.build_resume(profile, out)
    flags = validate.claim_flags(resume, jd, hints="")
    verify = [f for f in flags if f.level == "verify"]
    assert any("30%" in f.reason for f in verify)
    assert any(f.text == "Kubernetes" and f.level == "info" for f in flags)
    # a number the user supplied is not flagged
    assert not [f for f in validate.claim_flags(resume, jd, hints="Cut costs by 30% at Northwind") if f.level == "verify"]


def test_keyword_coverage_and_lint(profile):
    resume = validate.build_resume(profile, make_output())
    cov = validate.keyword_coverage(resume, make_output().jd_analysis)
    assert "Python" in cov.found and "Airflow" in cov.missing
    assert 0 < cov.score < 1
    warnings = validate.ats_lint(resume)
    assert not any("first-person" in w for w in warnings)


def test_cover_letter_cleaning():
    greeting, paras, closing = validate.clean_cover_letter("Dear X,", ["One. ", "", "Two."], "Sincerely,")
    assert paras == ["One.", "Two."]
    with pytest.raises(ValueError):
        validate.clean_cover_letter("Dear X,", ["Only one"], "Sincerely,")
