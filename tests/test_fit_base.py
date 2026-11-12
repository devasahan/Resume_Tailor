"""The restyle changes only the look. The bullet plan and the page count are still worked out on the layout from before
it (app/pipeline/old_layout.py: Carlito, measured without drawing), so every plan is the one made then, no resume takes
more pages than it did, and the new floor holds what the old one held. The numbers below were recorded with the app as
it was before the restyle (commit aa88a1a: fpdf2 and the Carlito files). Every person and employer here is invented."""
import asyncio
import random
from datetime import date

import pytest

from app.pipeline import fit, old_layout
from app.pipeline.fit import (
    CALIBRATED,
    MAX_PAGES,
    MIN_LAST_PAGE_FILL,
    OLD_LADDER,
    bullet_budget,
    styles,
)
from app.pipeline.old_layout import BOLD, ITALIC, REGULAR
from app.pipeline.tailor import refit
from app.render.pdf import resume_layout
from app.schemas import (
    Certification,
    Company,
    Contact,
    Education,
    FitStyle,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
)

TODAY = date(2026, 11, 12)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
EMPLOYERS = ["Northwind Labs", "Bluefin Commerce", "Contoso Systems", "Fabrikam Inc.", "Lakeside Community College",
             "Tailspin Logistics", "Litware Analytics", "Adventure Works", "Wingtip Payments", "Proseware",
             "Woodgrove Financial", "Alpine Ski House", "Coho Vineyard Group", "Lamna Healthcare", "Trey Research"]
PLACES = ["Austin, TX", "", "Seattle, WA", "Denver, CO", "", "Chicago, IL", "Boston, MA"]
MODES = ["Remote", "Hybrid", "", "Onsite"]
CERTIFICATIONS = [  # expiring long after any run of these tests: an expired one would leave the plan resume
    Certification(name="Cloud Solutions Architect Professional", issuer="Northwind Cloud Academy", year="Mar 2021",
                  expires="Mar 2098"),
    Certification(name="Certified Kubernetes Administrator", issuer="Fabrikam Institute", year="Mar 2022"),
    Certification(name="Professional Scrum Master I", issuer="Litware Guild", year="Mar 2023", expires="Mar 2099"),
    Certification(name="Data Engineering Associate", issuer="Contoso Academy", year="Mar 2024")]


def month(number: int) -> str:
    return f"{MONTHS[number % 12]} {number // 12}"


def career(employers: int, certifications: int = 0, short_stays: bool = False) -> Profile:
    """Taylor Quinn's invented career, the most recent job first: stays of 8 to 30 months, or 3 to 11 for a run of
    short contracts, a month or two apart."""
    companies, end = [], 2026 * 12 + 10  # Nov 2026, as a month number
    for i in range(employers):
        stay = 3 + (5 * i + 2) % 9 if short_stays else 8 + (7 * i + 3) % 23
        start = end - stay
        companies.append(Company(name=EMPLOYERS[i], start=month(start), end="Present" if i == 0 else month(end),
                                 location=PLACES[i % len(PLACES)], mode=MODES[i % len(MODES)]))
        end = start - 1 - i % 2
    return Profile(name="Taylor Quinn", contact=Contact(email="taylor.quinn@example.com", phone="(555) 010-4477",
                                                        linkedin="https://www.linkedin.com/in/taylorquinn-example/",
                                                        location="Denver, CO"),
                   companies=companies,
                   education=[Education(school="State University", degree="B.S.", field="Computer Science",
                                        start="2008", end="2012")],
                   certifications=[c.model_copy() for c in CERTIFICATIONS[:certifications]])


# ------------------------------------------------------------------ measuring
SAMPLES = ["Senior Data Engineer", "Zürich · Ελληνικά · Кириллица", "数据 😀 ﬁ", "a{nb}b", "{nb}", ""]
WIDTHS = [  # fpdf2's get_string_width of each sample in Carlito Regular, Bold and Italic at 9.75 pt
    [83.1675, 84.94200000000001, 83.04075], [120.46125, 124.0395, 119.3595], [24.3945, 24.63825, 24.3945],
    [24.618750000000002, 24.882, 24.85275], [14.82975, 14.82975, 14.82975], [0, 0, 0]]


def test_text_is_measured_as_fpdf2_measured_carlito():
    """The same numbers: characters Carlito lacks get its .notdef width, and fpdf2's page count alias "{nb}" is measured
    as "000" between the text around it. Python 3.12 adds up floats more exactly than 3.11 (in fpdf2 too), so the last
    bit of a text holding "{nb}" depends on the Python version: those were recorded on 3.13."""
    for text, widths in zip(SAMPLES, WIDTHS):
        measured = [old_layout.text_width(text, style, 9.75) for style in (REGULAR, BOLD, ITALIC)]
        assert measured == (widths if "{nb}" not in text else pytest.approx(widths, rel=1e-12)), text


# ------------------------------------------------------------------ the old layout
SENTENCES = [
    ("Rebuilt the checkout payment service for 41,000 daily orders in Go and PostgreSQL, adding idempotency keys and "
     "retry queues so failed card captures dropped from 1.9% to 0.4% of orders."),
    ("Moved the nightly settlement batch from cron scripts to Airflow with typed Python tasks, cutting the run from 6 "
     "hours to 47 minutes."),
    ("Led the café ordering rollout in Zürich and Kraków with naïve Bayes ranking, ﬁnance sign-off and the ﬂow "
     "reviewed by the Ελληνικά and Кириллица teams."),
    "Shipped the 数据 export (中文字符 included) and the 😀 feedback widget 🚀 behind a feature flag for 3 markets.",
    ("Documented https://www.example.com/a/very/long/path/that/never/breaks/because/it/has/no/spaces/at/all/index.html "
     "for partners."),
    "Kept the {nb} page alias literal in the report builder and a{nb}b too.",
    "Short one.",
    ("Cut p95 latency of the search API from 840 to 310 milliseconds by moving hot queries to Elasticsearch and "
     "caching facets in Redis, which removed three synchronous calls to the catalog service and let the merchandising "
     "team run seasonal promotions without paging the on-call engineer every weekend during the busiest weeks of the "
     "year."),
]
NAMES = ["Taylor Quinn", "Jordan Rivera",
         "Riley Brooks of the Northwind Valley and the Lakeside Community Council of Elders",
         "Τέιλορ Κουίν", "Джордан Ривера-Брукс", "泰勒·奎因", "Rïley Bröoks", "", "RILEY BROOKS",
         "Jordan Rivera-Montgomery-Fitzwilliam-Delacroix-Vandermeer-Ashworth"]
HEADLINES = ["Senior Software Engineer | Python | AWS", "Backend Engineer", " ".join(SENTENCES[0].split()[:40]), "",
             "Ingénieure Données | Python | Σ", "数据工程师", "Staff Engineer", "Data Engineer {nb}", "x" * 150,
             "Senior Full-Stack Engineer | TypeScript"]
CATEGORIES = ["Languages", "Data Platforms", "Cloud and Infrastructure and Many Other Things", "Языки", "Tools"]
TITLES = ["Senior Software Engineer", "", "Principal Engineer, Payments Platform and Developer Experience",
          "Разработчик"]


def odd_resumes() -> list[Resume]:
    """Resumes that walk the old layout's corners: a name too long for its size, a very long headline, no contact
    line, long unbreakable tokens, accents, Greek, Cyrillic, CJK, emoji and ligatures, fpdf2's {nb} alias, jobs
    without bullets or dates, education details, certifications with and without issuer and dates, and careers long
    enough for three pages."""
    resumes = []
    for number in range(10):
        jobs = [1, 3, 15, 6, 9, 2, 12, 4, 7, 14][number]
        companies = []
        for i in range(jobs):
            name = EMPLOYERS[(number + i) % 15] + (" International Holdings Group" if i % 4 == 3 else "")
            start = "" if (number + i) % 7 == 6 else month(2025 * 12 - 13 * i)
            end = "" if (number + i) % 5 == 4 else ("Present" if i == 0 else month(2025 * 12 - 13 * i + 11))
            companies.append(Company(name=name, start=start, end=end, location=PLACES[(number + i) % len(PLACES)],
                                     mode=MODES[(number * 3 + i) % len(MODES)]))
        education = [Education(school="State University", degree="B.S." if number % 3 else "",
                               field="Computer Science" if number % 2 else "", start="2008" if number % 4 else "",
                               end="2012", details=SENTENCES[number % 8] if number % 3 == 1 else "")
                     ][:1 if number % 5 else 0]
        if number % 2:
            education.append(Education(school="Lakeside Community College", degree="A.S.",
                                       details=SENTENCES[(number + 3) % 8]))
        certifications = [Certification(name=c.name if k % 3 else SENTENCES[(number + k) % 8][:70],
                                        issuer=c.issuer if (number + k) % 2 else "",
                                        year=c.year if (number + k) % 3 else "", expires="Mar 2031" if k == 1 else "")
                          for k, c in enumerate(CERTIFICATIONS[:number % 5])]
        contact = Contact(email="" if number == 4 else "riley.brooks@example.com",
                          phone="(555) 010-2290" if number % 2 else "",
                          linkedin=["", "linkedin.com/in/" + "x" * 90,
                                    "https://www.linkedin.com/in/rileybrooks/"][number % 3],
                          location="" if number in (4, 7) else "Kraków, Poland",
                          website="https://github.com/rileybrooks" if number % 4 == 1 else "")
        if number == 7:
            contact = Contact()
        profile = Profile(name=NAMES[number], contact=contact, companies=companies, education=education,
                          certifications=certifications)
        summary = " ".join((SENTENCES[number % 8] + " " + SENTENCES[(number + 1) % 8]).split()[:20 + 9 * number])
        skills = [SkillGroup(category=CATEGORIES[(number + g) % 5],
                             skills=[w.strip(",.") for w in SENTENCES[(number + g) % 8].split()[:3 + 3 * g]])
                  for g in range(number % 6)]
        experience = [ResumeExperience(company_index=i, job_title=TITLES[i % 4],
                                       bullets=[SENTENCES[(number + i + b) % 8] for b in range((number + 2 * i) % 8)])
                      for i in range(jobs)]
        resumes.append(Resume(profile=profile, headline=HEADLINES[number], summary=summary, skills=skills,
                              experience=experience, certifications=[c.model_copy() for c in certifications]))
    return resumes


OLD_LAYOUTS = [  # odd_resumes()[n] on Letter, then on A4: 'pages:last-page fill' at the five old steps
    "1:0.2536368843069874 1:0.24699900793650806 1:0.22267817059483727 1:0.21692361111111116 1:0.2133958333333333",
    "1:0.2367264028651993 1:0.23068803028498072 1:0.22476366544729837 1:0.20286664328670334 1:0.19956747067762926",
    "1:0.9521262886597939 1:0.849716553287982 1:0.8242073512906849 1:0.7648125000000001 1:0.6819097222222221",
    "1:0.8886461131379546 1:0.8625263074296154 1:0.7702933039636027 1:0.7474184623777426 1:0.7160763225915393",
    "3:0.6617339633447878 3:0.4828125000000001 3:0.43065375982042625 3:0.32276388888888885 3:0.057749999999999996",
    "3:0.5574710339574228 3:0.432375014890998 3:0.22351151844130646 3:0.18341905986569512 3:0.10273545571445271",
    "2:0.6228307560137455 2:0.4615291950113379 2:0.4100308641975307 2:0.3557708333333333 2:0.24243055555555554",
    "2:0.5369776423579092 2:0.44828190975393456 2:0.3433505093812688 2:0.2812187455350764 2:0.24239177025289324",
    "2:0.893241695303551 2:0.7346690759637188 2:0.6497895622895619 2:0.6119826388888886 2:0.42310416666666656",
    "2:0.7967098317497229 2:0.7126732319421832 2:0.5540521050492333 2:0.503445297380145 2:0.47534063307745256",
    "1:0.9160151775486828 1:0.8697704081632657 1:0.8431677890011225 1:0.8170972222222226 1:0.7491875000000002",
    "1:0.8727632335057267 1:0.8295642563104744 1:0.7880134786086092 1:0.7641481250568267 1:0.7476522620114565",
    "3:0.10325028636884308 2:0.9425418083900231 2:0.8874438832772169 2:0.8247708333333336 2:0.6066458333333332",
    "3:0.04283766988734316 2:0.8902500364002176 2:0.7595091059277033 2:0.7133324241125356 2:0.6300185740820117",
    "2:0.23690578465063009 2:0.12966269841269845 2:0.12568041526374862 2:0.05907291666666666 1:0.9337083333333343",
    "2:0.12481123628539738 2:0.12110021310672546 2:0.016651588456646865 1:0.9569191702710786 1:0.935886944888232",
    "2:0.3326102520045819 2:0.22526218820861688 2:0.1814955106621773 2:0.12925694444444444 2:0.07450694444444445",
    "2:0.21665397105400314 2:0.2103866364875777 2:0.12451979178958686 2:0.07132837158555118 2:0.06967878528101416",
    "3:0.9653207331042386 3:0.6800559807256233 3:0.5630892255892253 3:0.5279131944444443 3:0.27455555555555555",
    "3:0.6909019230512234 3:0.6696084660286701 3:0.4403755129869277 3:0.31403512190053123 3:0.3070243801062489",
]


def test_the_old_layout_gives_the_pages_and_fill_of_before():
    for number, resume in enumerate(odd_resumes()):
        for paper, recorded in zip(("Letter", "A4"), OLD_LAYOUTS[2 * number:2 * number + 2]):
            for (font_pt, margin_in), expected in zip(OLD_LADDER, recorded.split()):
                style = FitStyle(font_pt=font_pt, margin_in=margin_in, paper=paper)
                pages, fill = old_layout.resume_layout(resume, style)
                expected_pages, expected_fill = expected.split(":")
                assert pages == int(expected_pages) and fill == pytest.approx(float(expected_fill), abs=1e-9), (
                    number, paper, font_pt)


# ------------------------------------------------------------------ the bullet plan
PLANS = {  # (certifications, paper, short stays): the plans for 2, 3, ... 15 employers, one digit a job
    (0, "Letter", False): "55 555 5553 55533 555333 5553333 55533333 555333333 5553333333 55533333222 555332222222 "
                          "5543222222222 44422222222222 333222222222222",
    (0, "Letter", True): "33 333 3334 33343 333432 3334323 33343232 333432323 3334323233 33343232333 333432323333 "
                         "3334323233222 33343232222222 333222222222222",
    (0, "A4", False): "55 555 5553 55533 555333 5553333 55533333 555333333 5553333333 55533333333 555333333222 "
                      "5553333222222 55433222222222 544222222222222",
    (0, "A4", True): "33 333 3334 33343 333432 3334323 33343232 333432323 3334323233 33343232333 333432323333 "
                     "3334323233333 33343232333222 333432322222222",
    (4, "Letter", False): "55 555 5553 55533 555333 5553333 55533333 555333333 5553333222 55533222222 544322222222 "
                          "4442222222222 33322222222222 222222222222222",
    (4, "Letter", True): "33 333 3334 33343 333432 3334323 33343232 333432323 3334323233 33343232333 333432323322 "
                         "3334322222222 33322222222222 222222222222222",
    (4, "A4", False): "55 555 5553 55533 555333 5553333 55533333 555333333 5553333333 55533333322 555333222222 "
                      "5543322222222 54422222222222 433222222222222",
    (4, "A4", True): "33 333 3334 33343 333432 3334323 33343232 333432323 3334323233 33343232333 333432323333 "
                     "3334323233322 33343232222222 333322222222222",
}


def test_the_bullet_plans_are_the_ones_made_before_the_restyle():
    """The model is asked for exactly as many bullets per job as before, for careers of 2 to 15 employers, with and
    without certifications, on both papers."""
    for (certifications, paper, short_stays), plans in PLANS.items():
        for employers, plan in enumerate(plans.split(), start=2):
            profile = career(employers, certifications, short_stays)
            assert bullet_budget(profile, paper, today=TODAY) == [int(n) for n in plan], (employers, certifications,
                                                                                          paper, short_stays)


# ------------------------------------------------------------------ the page count
ONE_PAGE_BULLETS = [  # by job, the most recent first: long bullets, as drafts at the top of the range write them
    [("Led the migration of 23 customer-facing services from self-managed virtual machines to Kubernetes on AWS "
      "with reusable Terraform modules and Helm charts, which let every product team ship through one tested "
      "release pipeline."),
     ("Split the slow monthly invoice job into small parallel tasks on a Celery queue with progress tracking, "
      "bringing invoice generation for 18,700 accounts from a full night down to 52 minutes."),
     ("Replaced a fragile shell-script deployment with GitHub Actions workflows, automated database migrations "
      "and canary releases on Amazon ECS, so the team released several times a day with far fewer failed or "
      "reverted deploys."),
     ("Designed the multi-tenant permissions model for the clinic scheduling product in PostgreSQL with row-level"
      " security and audit trails, which let large hospital groups manage staff access without opening support "
      "tickets."),
     ("Built a fraud scoring API in Python and FastAPI that combined device fingerprints, order history and "
      "refund signals, so the risk team reviewed fewer false alarms and blocked repeat chargeback accounts within"
      " minutes.")],
    [("Partnered with product and design on the subscription billing redesign, writing the Stripe integration, "
      "proration rules, invoice previews and dunning emails that recovered failed renewals without manual "
      "follow-up from support."),
     ("Rebuilt the checkout payment service for 41,000 daily orders in Go and PostgreSQL, adding idempotency "
      "keys, retry queues and settlement reconciliation so failed card captures dropped from 1.9% to 0.4% of all "
      "orders processed."),
     ("Cut p95 latency of the product search API from 840 to 310 milliseconds by moving hot queries to "
      "Elasticsearch, caching facet counts in Redis and removing three synchronous calls to the catalog and "
      "pricing services.")],
    [("Set up OpenTelemetry tracing, structured logs and Grafana dashboards across the API gateway and eleven "
      "core services, which helped on-call engineers find the cause of production incidents in minutes instead of"
      " hours."),
     ("Built the claims intake API in C# and ASP.NET Core with Azure Service Bus and schema validation, letting "
      "partner clinics submit insurance claims electronically and cutting average processing time from 9 days to "
      "4 days."),
     ("Replaced manual certificate renewal with cert-manager and automated DNS validation on every cluster, "
      "ending the expired certificate outages that had taken the customer portal down twice a year."),
     ("Introduced structured logging and log sampling across 31 services with a shared Python library, cutting "
      "the log bill by $41,300 a year while keeping every error and slow request searchable."),
     ("Maintained the shared Terraform modules for networking, managed databases and message queues across 3 AWS "
      "accounts, and documented them so product teams created new environments safely without waiting on "
      "platform.")],
    [("Created the public partner API with versioned REST endpoints, per-client rate limits and generated OpenAPI"
      " documentation, which helped 64 partners integrate on their own and cut onboarding calls for the solutions"
      " team."),
     ("Migrated the legacy monolith's user accounts and session handling to a separate authentication service "
      "with OAuth 2.0 and short-lived tokens, closing two audit findings and enabling single sign-on for "
      "enterprise customers."),
     ("Profiled and fixed the slowest PostgreSQL queries behind the customer reporting screens by adding covering"
      " indexes and rewriting two report builders, bringing the heaviest report from 38 seconds to under 3 "
      "seconds."),
     ("Automated test data setup for the end-to-end browser suite with factories, seeded containers and Docker "
      "Compose profiles, so the full suite ran in 14 minutes on every pull request and flaky failures mostly "
      "stopped.")],
]


def one_page_resume() -> Resume:
    """Four short stays on A4, written to their plan (5, 3, 5, 4 bullets). Before the restyle 10.5 pt left a second
    page under a quarter full, so the fitting rule went on to 9.5 pt on one page."""
    jobs = [("Wingtip Payments", "Dec 2025", "Present", "Hybrid", "Atlanta, GA", "Lead Software Engineer"),
            ("Proseware", "Nov 2024", "Sep 2025", "Hybrid", "Seattle, WA", "Senior Full-Stack Engineer"),
            ("Woodgrove Financial", "Aug 2023", "Sep 2024", "Onsite", "Austin, TX", "Software Engineer II"),
            ("Alpine Ski House", "Dec 2021", "Jun 2023", "Hybrid", "", "Senior Backend Engineer")]
    profile = Profile(name="Jordan Rivera", contact=Contact(email="jordan.rivera@example.com", phone="(555) 014-2290",
                                                            linkedin="linkedin.com/in/jordanrivera-example",
                                                            location="Denver, CO"),
                      companies=[Company(name=name, start=start, end=end, mode=mode, location=place)
                                 for name, start, end, mode, place, _title in jobs],
                      education=[Education(school="Lakeshore State University", degree="B.S.",
                                           field="Computer Science", start="2008", end="2012")])
    summary = ("Software engineer who builds reliable payment and scheduling systems for growing SaaS companies, from "
               "the data model to the release pipeline. Writes services in Java, Kotlin and Python, runs them on "
               "Kubernetes and keeps them observable with tracing and good alerts. Has helped finance, support and "
               "operations teams trust their numbers and spend less time on manual work. Enjoys mentoring, careful "
               "code review and turning vague requests into small, shippable steps.")
    skills = [("Languages", "Go, Python, Java, Kotlin, TypeScript, SQL, Bash"),
              ("Backend and Data", "PostgreSQL, Redis, Kafka, Elasticsearch, Snowflake, dbt, Airflow, gRPC, REST APIs"),
              ("Cloud and DevOps", "AWS, Kubernetes, Docker, Terraform, Helm, GitHub Actions, Datadog, Grafana"),
              ("Practices", ("System Design, Event-Driven Architecture, Code Review, Testing, Incident Response, "
                             "Mentoring"))]
    return Resume(profile=profile, headline="Backend Engineer", summary=summary,
                  skills=[SkillGroup(category=name, skills=items.split(", ")) for name, items in skills],
                  experience=[ResumeExperience(company_index=i, job_title=job[5], bullets=bullets)
                              for i, (job, bullets) in enumerate(zip(jobs, ONE_PAGE_BULLETS))])


def test_a_resume_of_one_page_before_the_restyle_keeps_one_page():
    """The new layout's first calibrated step fills this resume's second page just past a quarter, so picking the page
    count on the new layout gave two pages at the design's sizes. The count from before keeps it on one page."""
    resume = one_page_resume()
    assert fit.old_pages(resume, "A4") == 1
    pages, fill = resume_layout(resume, styles("A4")[CALIBRATED])
    assert pages == 2 and fill >= MIN_LAST_PAGE_FILL  # the fitting rule on the new layout would keep two pages
    _style, pages, _pdf, notes = fit.try_styles(resume, "A4")
    assert pages == 1 and any("one page" in note for note in notes)


def test_no_resume_takes_more_pages_than_before():
    """Seeded plan resumes of short careers, around the one-page mark: the pages try_styles gives are never more than
    the fitting rule gave on the old layout."""
    rng = random.Random(2026)
    for _ in range(16):
        employers = rng.randint(2, 5)
        profile = career(employers, rng.choice([0, 4]), rng.random() < 0.5)
        resume = fit._plan_resume(profile, [rng.randint(2, 6) for _ in range(employers)])
        paper = rng.choice(["Letter", "A4"])
        before, found = fit.old_pages(resume, paper), fit.try_styles(resume, paper)
        assert before is not None and found is not None and found[1] <= before, (employers, paper, before)


# ------------------------------------------------------------------ the floor
def test_the_floor_holds_a_long_career_filled_to_the_old_limit():
    """Fifteen jobs and four certifications, filled right up to what the old 9.5 pt floor held on two pages (one more
    bullet takes a third): the new floor holds it too, so ticking the certifications on the Tailor page (refit) or
    trimming removes no bullet. At 8.2 pt the last certification ran onto a third page."""
    profile, old_floor = career(15, 4), FitStyle(font_pt=9.5, margin_in=0.5)
    resume = fit._plan_resume(profile, [3, 3, 3] + [2] * 12)
    assert old_layout.resume_layout(resume, old_floor)[0] == 2
    assert old_layout.resume_layout(fit._plan_resume(profile, [3, 3, 3, 3] + [2] * 11), old_floor)[0] == 3
    assert resume_layout(resume, FitStyle(font_pt=8.2, margin_in=0.5))[0] == 3
    assert resume_layout(resume, styles("Letter")[-1])[0] == MAX_PAGES
    fitted, _style, pages, fits, _notes, removed = asyncio.run(refit(resume, "Letter"))
    assert fits and pages == MAX_PAGES and removed == {} and fitted == resume
    trimmed, _style, pages, _pdf, notes = fit.trim_to_fit(resume)
    assert pages == MAX_PAGES and trimmed == resume and not any(note.startswith("Removed") for note in notes)
