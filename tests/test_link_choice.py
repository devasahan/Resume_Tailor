"""Links, the admin's only for now: LinkedIn, GitHub and the website, typed on the Profile page's Links box, and tick
boxes on the Tailor page that choose which of them a resume and its cover letter print. GitHub has a field of its own,
so a portfolio is kept and printed beside it; a resume uploaded again keeps the typed links. Every other account works
as before: no box, no tick boxes, and every link its upload read is printed. Every person, address and company here is
invented."""
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pypdf import PdfReader

from app import main
from app.llm.schema import strict_json_schema
from app.parsing.heuristic import parse_resume_text
from app.parsing.resume_parser import extract_text
from app.pipeline import old_layout, validate
from app.pipeline import tailor as tailor_module
from app.prompts import parse_system
from app.render.docx import cover_docx, resume_docx
from app.render.pdf import contact_line, render_cover_pdf, render_resume_pdf
from app.schemas import Company, Contact, CoverLetter, FitStyle, Profile, TailorResult

from .test_api import save_profile
from .test_api import tailor as tailor_request
from .test_contact_links import contact_paragraph, page_links
from .test_validate import make_output

KINDS = ("linkedin", "github", "website")
LINKEDIN, GITHUB, WEBSITE = "linkedin.com/in/taylorquinn", "github.com/taylorquinn", "taylorquinn.dev"
MAIL = "mailto:taylor@example.com"
OPENS = {"linkedin": "https://linkedin.com/in/taylorquinn", "github": "https://github.com/taylorquinn",
         "website": "https://taylorquinn.dev"}
PLACE_PHONE_MAIL = "Austin, TX · (555) 010-4477 · taylor@example.com"
BODY = "EXPERIENCE\nNorthwind Labs | Remote\nMar 2022 - Present\n\nEDUCATION\nState University\n"
# a resume with every link (read with the rules: GitHub and the portfolio both kept) and one with none
RESUME = f"Taylor Quinn\nAustin, TX | taylor@example.com | (555) 010-4477 | {LINKEDIN} | {GITHUB} | {WEBSITE}\n\n{BODY}"
NO_LINKS = f"Taylor Quinn\nAustin, TX | taylor@example.com | (555) 010-4477\n\n{BODY}"
LETTER = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["I build data pipelines at Northwind Labs."],
                     closing="Sincerely,")


def links_of(contact) -> tuple[str, str, str]:
    contact = contact if isinstance(contact, dict) else contact.model_dump()
    return tuple(contact[kind] for kind in KINDS)


def upload(client, text: str = RESUME, name: str = "taylor.docx") -> dict:
    return save_profile(client, name=name, text=text)


def stored_links(client, resume_id: int) -> tuple[str, str, str]:
    entry = next(r for r in client.get("/api/resumes").json()["resumes"] if r["id"] == resume_id)
    return links_of(entry["profile"]["contact"])


def contact_text(pdf: bytes, end: str = "SUMMARY") -> str:
    """A file's contact line as printed under the name and the title line, its lines joined as one (it breaks between
    two items when it is too long for one line)."""
    lines = PdfReader(io.BytesIO(pdf)).pages[0].extract_text().splitlines()
    return " · ".join(lines[2:lines.index(end)])


def drawing(pdf: bytes) -> list[bytes]:
    """What each page draws (its content stream): the text, colours, sizes and places, without the link annotations."""
    return [page.get_contents().get_data() for page in PdfReader(io.BytesIO(pdf)).pages]


# ------------------------------------------------------------------ the data: GitHub apart from the website
def test_github_has_a_field_of_its_own_and_old_rows_load():
    assert Contact().github == "" and Contact.model_validate({"email": "taylor@example.com"}).github == ""
    contact = Contact(linkedin=LINKEDIN, github=GITHUB, website=WEBSITE)
    assert links_of(contact) == (LINKEDIN, GITHUB, WEBSITE)


@pytest.mark.parametrize("website", [GITHUB, "https://www.github.com/taylorquinn/", "gitlab.com/tquinn",
                                     "HTTP://GitHub.com/TaylorQuinn", "github.com/taylorquinn?tab=repositories"])
def test_a_github_or_gitlab_address_kept_as_the_website_moves_to_github(website):
    """Profiles and runs saved before kept GitHub in the website: they load with it as GitHub, as typed."""
    contact = Contact.model_validate({"email": "taylor@example.com", "website": website})
    assert (contact.github, contact.website) == (website, "")
    old = Profile.model_validate_json(json.dumps({"name": "Taylor Quinn", "contact": {"website": f" {website} "}}))
    assert (old.contact.github, old.contact.website) == (f" {website} ", "")


@pytest.mark.parametrize("contact,expected", [
    ({"website": WEBSITE}, ("", WEBSITE)),  # a portfolio stays the website
    ({"website": "taylorquinn.github.io/cv"}, ("", "taylorquinn.github.io/cv")),  # GitHub Pages is a website
    ({"website": "gist.github.com/taylorquinn"}, ("", "gist.github.com/taylorquinn")),
    ({"website": "github.com"}, ("", "github.com")),  # no profile in it
    ({"github": GITHUB, "website": "gitlab.com/tquinn"}, (GITHUB, "gitlab.com/tquinn")),  # GitHub kept: nothing moves
])
def test_only_a_github_or_gitlab_address_moves_and_only_into_an_empty_github(contact, expected):
    loaded = Contact.model_validate(contact)
    assert (loaded.github, loaded.website) == expected


def test_the_ai_reader_is_asked_for_github_apart_from_the_website():
    """The reader's strict schema asks for every field: the prompt names GitHub's, and says the website is another
    site of the person's, never LinkedIn's or GitHub's."""
    contact = strict_json_schema(Profile)["properties"]["contact"]
    assert "github" in contact["required"] and "website" in contact["required"]
    prompt = parse_system()
    line = next(line for line in prompt.splitlines() if line.startswith("- contact:"))
    assert "github" in line and "GitHub or GitLab" in line and "never LinkedIn's, GitHub's or GitLab's" in line


# ------------------------------------------------------------------ the rules reader
@pytest.mark.parametrize("header,github,website", [
    ([f"Austin, TX | taylor@example.com | {GITHUB} | {WEBSITE}"], GITHUB, WEBSITE),
    ([f"taylor@example.com | Portfolio: https://{WEBSITE} | GitHub: {GITHUB}"], GITHUB, f"https://{WEBSITE}"),
    ([f"https://{WEBSITE} | taylor@example.com", "gitlab.com/tquinn"], "gitlab.com/tquinn", f"https://{WEBSITE}"),
    (["taylor@example.com", f"My GitHub page: https://www.{GITHUB}/ · www.{WEBSITE}"], f"https://www.{GITHUB}/",
     f"www.{WEBSITE}"),
    (["taylor@example.com | github.com/taylorquinn?tab=repositories"], "github.com/taylorquinn?tab=repositories", ""),
    ([f"taylor@example.com | {LINKEDIN} | taylorquinn.github.io/cv"], "", "taylorquinn.github.io/cv"),
])
def test_the_rules_reader_keeps_github_and_a_portfolio_apart(header, github, website):
    profile = parse_resume_text("\n".join(["Taylor Quinn", *header, *BODY.splitlines()]))
    assert (profile.contact.github, profile.contact.website) == (github, website)
    assert profile.name == "Taylor Quinn" and profile.contact.email == "taylor@example.com"


# ------------------------------------------------------------------ printing
EVERY = Contact(email="taylor@example.com", phone="(555) 010-4477", linkedin=f"https://www.{LINKEDIN}",
                github=f"https://{GITHUB}", website=f"https://{WEBSITE}/", location="Austin, TX")


def taylor_resume(profile, contact: Contact = EVERY):
    person = profile.model_copy(deep=True, update={"name": "Taylor Quinn", "contact": contact})
    return validate.build_resume(person, make_output())


def test_the_contact_line_prints_github_after_linkedin_and_before_the_website(profile):
    """Place, phone, email, LinkedIn, GitHub, website, whatever the order written; all three linked like the email,
    in the PDF (resume and letter) and the Word copy, in that order."""
    line = f"{PLACE_PHONE_MAIL} · {LINKEDIN} · {GITHUB} · {WEBSITE}"
    assert contact_line(EVERY) == line
    assert old_layout.contact_line(EVERY) == line.replace(" · ", " | ")  # the layout before the restyle too
    resume = taylor_resume(profile)
    targets = [MAIL, f"https://www.{LINKEDIN}", f"https://{GITHUB}", f"https://{WEBSITE}/"]
    for pdf, end in ((render_resume_pdf(resume, FitStyle())[0], "SUMMARY"),
                     (render_cover_pdf(resume, LETTER)[0], LETTER.greeting)):
        assert [target for target, _rect in page_links(pdf)] == targets
        assert contact_text(pdf, end) == line
    for data in (resume_docx(resume, FitStyle()), cover_docx(resume, LETTER)):
        _doc, paragraph = contact_paragraph(data)
        assert paragraph.text.replace("\n", " · ") == line  # its lines broken where the PDF breaks them
        assert [(link.text, link.address) for link in paragraph.hyperlinks] == [
            ("taylor@example.com", MAIL), (LINKEDIN, targets[1]), (GITHUB, targets[2]), (WEBSITE, targets[3])]


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_the_apps_own_files_read_back_with_all_three_links(profile, kind):
    resume = taylor_resume(profile)
    data = render_resume_pdf(resume, FitStyle())[0] if kind == "pdf" else resume_docx(resume, FitStyle())
    contact = parse_resume_text(extract_text(data, f"resume.{kind}")).contact
    assert links_of(contact) == (LINKEDIN, GITHUB, WEBSITE)


def test_a_run_saved_before_prints_exactly_the_text_it_did(profile):
    """A run saved before GitHub had its field (its address in the website) loads with it as GitHub, and prints the
    same contact line, in the same place on its page, in the new look and in the layout from before the restyle (its
    page plan and page count come from that one), with the same links."""
    resume = taylor_resume(profile, EVERY.model_copy(update={"github": "", "website": f"https://{GITHUB}"}))
    saved = TailorResult(requested_model="m", resume=resume, pages=1, fits=True).model_dump()
    saved["resume"]["profile"]["contact"].pop("github")  # as stored before
    assert saved["resume"]["profile"]["contact"]["website"] == f"https://{GITHUB}"
    loaded = TailorResult.model_validate_json(json.dumps(saved)).resume
    assert (loaded.profile.contact.github, loaded.profile.contact.website) == (f"https://{GITHUB}", "")
    before = f"{PLACE_PHONE_MAIL} · {LINKEDIN} · {GITHUB}"
    assert contact_line(loaded.profile.contact) == before
    assert old_layout.contact_line(loaded.profile.contact) == before.replace(" · ", " | ")
    for style in (FitStyle(), FitStyle(layout=1)):
        assert old_layout.resume_layout(loaded, style) == old_layout.resume_layout(resume, style)
        (pdf, pages), (before_pdf, before_pages) = render_resume_pdf(loaded, style), render_resume_pdf(resume, style)
        assert pages == before_pages and drawing(pdf) == drawing(before_pdf) and contact_text(pdf) == before
        assert [target for target, _rect in page_links(pdf)] == [MAIL, f"https://www.{LINKEDIN}", f"https://{GITHUB}"]


# ------------------------------------------------------------------ the Profile page's Links box (the admin's only)
def links_body(entry: dict, **typed) -> dict:
    """The Save links body: the three boxes as the page shows them (what was saved), these ones typed, and the three
    as the page loaded them."""
    contact = entry["profile"]["contact"]
    loaded = {kind: contact.get(kind, "") for kind in KINDS}
    return {**loaded, **typed, "loaded": loaded}


def test_the_links_box_saves_the_three_links_trimmed_as_typed(client):
    entry = upload(client)
    assert links_of(entry["profile"]["contact"]) == (LINKEDIN, GITHUB, WEBSITE)  # what the upload read
    url = f"/api/resumes/{entry['id']}/links"
    body = links_body(entry, linkedin=f"  https://www.{LINKEDIN}/ ", github=" gitlab.com/tquinn", website="")
    response = client.put(url, json=body)
    assert response.status_code == 200, response.text
    saved = response.json()["resume"]
    assert links_of(saved["profile"]["contact"]) == (f"https://www.{LINKEDIN}/", "gitlab.com/tquinn", "")
    assert stored_links(client, entry["id"]) == (f"https://www.{LINKEDIN}/", "gitlab.com/tquinn", "")
    assert saved["profile"]["contact"]["email"] == "taylor@example.com"  # the rest of the contact stays
    # cleared and typed again
    again = client.put(url, json=links_body(saved, linkedin="", github="", website=f"https://{WEBSITE}/cv?lang=en"))
    assert again.status_code == 200, again.text
    assert stored_links(client, entry["id"]) == ("", "", f"https://{WEBSITE}/cv?lang=en")


def test_a_stale_tab_another_accounts_resume_and_another_account_are_refused(client, user_client):
    entry = upload(client)
    url = f"/api/resumes/{entry['id']}/links"
    first, second = links_body(entry, github="gitlab.com/tquinn"), links_body(entry, website="")
    assert client.put(url, json=first).status_code == 200
    stale = client.put(url, json=second)  # opened before the first save
    assert stale.status_code == 409 and stale.json()["detail"] == (
        "The links of this resume were changed in another tab. Reload this page and save again.")
    assert stored_links(client, entry["id"]) == (LINKEDIN, "gitlab.com/tquinn", WEBSITE)
    mine = upload(user_client)
    assert client.put(f"/api/resumes/{mine['id']}/links", json=links_body(mine, github="")).status_code == 404
    assert client.put("/api/resumes/99999/links", json=first).status_code == 404
    refused = user_client.put(f"/api/resumes/{mine['id']}/links", json=links_body(mine, github=""))
    assert refused.status_code == 403 and refused.json()["detail"] == "Only the admin can do that."
    assert stored_links(user_client, mine["id"]) == (LINKEDIN, GITHUB, WEBSITE)
    other_site = client.put(url, json=first, headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert other_site.status_code == 403


LINKEDIN_RULE = "The LinkedIn box needs your LinkedIn address, like linkedin.com/in/your-name."
GITHUB_RULE = "The GitHub box needs your GitHub address, like github.com/your-name."
WEBSITE_RULE = "The Website box needs the address of your own site, like your-name.dev."
NOT_LINKEDIN = ("The Website box is for another site of yours, like a portfolio: put your LinkedIn address in the "
                "LinkedIn box.")
NOT_GITHUB = ("The Website box is for another site of yours, like a portfolio: put your GitHub address in the GitHub "
              "box.")


@pytest.mark.parametrize("kind,value,message", [
    ("linkedin", GITHUB, LINKEDIN_RULE), ("linkedin", "linkedin.com", LINKEDIN_RULE),
    ("linkedin", "https://linkedin.com.evil.example/in/taylorquinn", LINKEDIN_RULE),
    ("linkedin", "javascript:alert(1)//linkedin.com/in/x", LINKEDIN_RULE),
    ("github", WEBSITE, GITHUB_RULE), ("github", "https://github.com/", GITHUB_RULE),
    ("github", "taylorquinn.github.io", GITHUB_RULE), ("github", "github.com/taylor quinn", GITHUB_RULE),
    ("github", 'github.com/taylorquinn"onmouseover="alert(1)', GITHUB_RULE),
    ("website", f"https://www.{LINKEDIN}", NOT_LINKEDIN), ("website", "uk.linkedin.com/in/taylorquinn", NOT_LINKEDIN),
    ("website", f"www.{GITHUB}", NOT_GITHUB), ("website", "GitLab.com/tquinn", NOT_GITHUB),
    ("website", "data:text/html,<b>taylor</b>", WEBSITE_RULE), ("website", "N/A", WEBSITE_RULE),
    ("website", "localhost:8000/cv", WEBSITE_RULE), ("website", "file:///home/taylor/cv.pdf", WEBSITE_RULE),
    ("website", "taylorquinn\u200b.dev", WEBSITE_RULE), ("website", "//evil.example/x", WEBSITE_RULE),
    ("website", f"{WEBSITE}/" + "x" * 300, "The Website box has 316 characters: a link can have up to 300."),
])
def test_only_a_safe_link_on_its_own_site_is_saved(client, kind, value, message):
    """Each is empty or a web address a browser can open from the files, on its box's site: refused with a plain
    message, and nothing is saved."""
    entry = upload(client)
    response = client.put(f"/api/resumes/{entry['id']}/links", json=links_body(entry, **{kind: value}))
    assert response.status_code == 422 and response.json()["detail"] == message
    assert stored_links(client, entry["id"]) == (LINKEDIN, GITHUB, WEBSITE)


@pytest.mark.parametrize("kind,value", [
    ("linkedin", f"https://www.{LINKEDIN}/"), ("linkedin", "uk.linkedin.com/in/taylorquinn"),
    ("linkedin", "HTTPS://WWW.LinkedIn.com/in/TaylorQuinn"), ("linkedin", "linkedin.com/in/josé-quinn"),
    ("github", "https://gitlab.com/tquinn/"), ("github", "www.github.com/taylorquinn"),
    ("website", "taylorquinn.github.io/cv"), ("website", "http://taylorquinn.dev:8080/cv?lang=en"),
    ("website", "gist.github.com/taylorquinn"), ("website", "x" * 289 + ".dev/cv"),
])
def test_links_written_the_usual_ways_are_saved(client, kind, value):
    entry = upload(client)
    response = client.put(f"/api/resumes/{entry['id']}/links", json=links_body(entry, **{kind: value}))
    assert response.status_code == 200, response.text
    assert response.json()["resume"]["profile"]["contact"][kind] == value


def test_the_page_checks_the_length_the_server_does(client):
    assert main.MAX_LINK_CHARS == 300
    page = client.get("/profile").text
    assert f"const MAX_LINK_CHARS = {main.MAX_LINK_CHARS};" in page
    assert "body: { ...typed, loaded }" in page and "maxlength: MAX_LINK_CHARS" in page


# ------------------------------------------------------------------ a resume uploaded again
def contact_profile(name: str = "Taylor Quinn", **links: str) -> Profile:
    return Profile(name=name, contact=Contact(email="taylor@example.com", **links),
                   companies=[Company(name="Northwind Labs", start="Mar 2022", end="Present")])


def test_a_new_upload_keeps_the_typed_links_it_leaves_out_and_what_the_file_says_wins():
    earlier = [contact_profile(linkedin=f"https://www.{LINKEDIN}", github="gitlab.com/tquinn", website=WEBSITE)]
    new = contact_profile(name="TAYLOR  QUINN", github=GITHUB)  # the same person; the file has another GitHub
    assert main.carry_links(new, earlier) == ["LinkedIn", "Website"]
    assert links_of(new.contact) == (f"https://www.{LINKEDIN}", GITHUB, WEBSITE)


def test_a_cleared_link_stays_cleared_and_another_persons_resume_gets_none():
    newest = contact_profile(linkedin=LINKEDIN)  # the website and GitHub were cleared in the Links box
    oldest = contact_profile(linkedin=LINKEDIN, github=GITHUB, website=WEBSITE)
    new = contact_profile()
    assert main.carry_links(new, [newest, oldest]) == ["LinkedIn"]
    assert links_of(new.contact) == (LINKEDIN, "", "")
    other = contact_profile(name="Morgan Lee")
    assert main.carry_links(other, [newest, oldest]) == [] and links_of(other.contact) == ("", "", "")


def test_an_admin_upload_keeps_the_typed_links_and_says_so(client):
    entry = upload(client)
    url = f"/api/resumes/{entry['id']}/links"
    body = links_body(entry, github="gitlab.com/tquinn", website=f"https://{WEBSITE}/cv")
    assert client.put(url, json=body).status_code == 200
    again = upload(client, text=NO_LINKS.replace("Austin, TX |", f"Austin, TX | {LINKEDIN}/ |"), name="again.docx")
    assert again["id"] != entry["id"]
    assert links_of(again["profile"]["contact"]) == (f"{LINKEDIN}/", "gitlab.com/tquinn", f"https://{WEBSITE}/cv")
    assert "Links kept from your earlier upload: GitHub, Website." in again["warnings"]


def test_another_accounts_upload_has_the_files_links_only(user_client):
    """No Links box, so nothing typed to keep: a new upload has what its file says, as before."""
    upload(user_client)
    again = upload(user_client, text=NO_LINKS, name="again.docx")
    assert links_of(again["profile"]["contact"]) == ("", "", "")
    assert not any(warning.startswith("Links kept") for warning in again["warnings"])


# ------------------------------------------------------------------ the Tailor page's choice
def printed(client, run_id: int) -> dict[str, list[str]]:
    """The link targets of each of the run's files (resume and letter, PDF and Word)."""
    pdf = {name: [target for target, _rect in page_links(client.get(f"/api/runs/{run_id}/{name}.pdf").content)]
           for name in ("resume", "cover")}
    word = {name: [link.address for link in contact_paragraph(
        client.get(f"/api/runs/{run_id}/{name}.docx").content)[1].hyperlinks] for name in ("resume", "cover")}
    return {"resume.pdf": pdf["resume"], "cover.pdf": pdf["cover"], "resume.docx": word["resume"],
            "cover.docx": word["cover"]}


@pytest.mark.parametrize("links,kept", [
    (["github"], ["github"]), ([], []), (["website", "linkedin"], ["linkedin", "website"]),
    (["linkedin", "github", "website"], ["linkedin", "github", "website"]), (None, ["linkedin", "github", "website"]),
], ids=["GitHub only", "none", "two", "all", "no choice sent"])
def test_only_the_ticked_links_are_printed(client, monkeypatch, links, kept):
    """The admin's ticks: the run is made from a copy of the resume without the unticked links, so the page plan, the
    resume, its letter and their files (PDF and Word, clickable) all leave them off; the saved resume keeps them. No
    choice sent (the API, scripts): every saved link, as before."""
    planned = []
    real_plan = tailor_module.plan_pages
    monkeypatch.setattr(tailor_module, "plan_pages",
                        lambda profile, *args, **kwargs: planned.append(profile.contact) or real_plan(
                            profile, *args, **kwargs))
    entry = upload(client)
    extra = {} if links is None else {"links": links}
    response = tailor_request(client, profile_id=entry["id"], cover_letter=True, company_name="Fabrikam", **extra)
    assert response.status_code == 200, response.text
    run = response.json()
    expected = tuple(dict(zip(KINDS, (LINKEDIN, GITHUB, WEBSITE)))[kind] if kind in kept else "" for kind in KINDS)
    assert links_of(run["resume"]["profile"]["contact"]) == expected and links_of(planned[-1]) == expected
    assert run["resume"]["profile"]["contact"]["email"] == "taylor@example.com"
    targets = [MAIL] + [OPENS[kind] for kind in KINDS if kind in kept]
    assert printed(client, run["run_id"]) == dict.fromkeys(("resume.pdf", "cover.pdf", "resume.docx", "cover.docx"),
                                                            targets)
    shown = [shown for kind, shown in zip(KINDS, (LINKEDIN, GITHUB, WEBSITE)) if kind in kept]
    assert contact_text(client.get(f"/api/runs/{run['run_id']}/resume.pdf").content) == " · ".join(
        [PLACE_PHONE_MAIL, *shown])
    assert stored_links(client, entry["id"]) == (LINKEDIN, GITHUB, WEBSITE)  # the saved resume is never changed
    # a letter written again later is the run's: the same links
    rewritten = client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})
    assert rewritten.status_code == 200, rewritten.text
    assert [target for target, _rect in page_links(client.get(f"/api/runs/{run['run_id']}/cover.pdf").content)] == (
        targets)


def test_another_accounts_choice_is_ignored_and_every_link_printed(client, user_client):
    upload(user_client)
    response = tailor_request(user_client, links=["github"], cover_letter=True, company_name="Fabrikam")
    assert response.status_code == 200, response.text
    run = response.json()
    assert set(run) == {"run_id", "status", "error", "pages", "fits", "coverage", "cover_letter", "created_at",
                        "spelling", "certifications"}  # nothing new in what a user is sent
    every = [MAIL, *OPENS.values()]
    for name in ("resume", "cover"):
        pdf = user_client.get(f"/api/runs/{run['run_id']}/{name}.pdf").content
        assert [target for target, _rect in page_links(pdf)] == every, name
    assert links_of(client.get(f"/api/runs/{run['run_id']}").json()["resume"]["profile"]["contact"]) == (
        LINKEDIN, GITHUB, WEBSITE)


def test_a_choice_names_only_the_three_links(client):
    upload(client)
    assert tailor_request(client, links=["email"]).status_code == 422
    assert tailor_request(client, links="github").status_code == 422


# ------------------------------------------------------------------ the pages, run with Node.js
PROFILE_PAGE = Path(__file__).parents[1] / "app/web/templates/profile.html"
TAILOR_PAGE = Path(__file__).parents[1] / "app/web/templates/tailor.html"
# Stand-ins for the browser: elements made by el() (as app.js makes them) with their children, attributes, data-*,
# checked, hidden, value and event handlers; $ and $$ find '#id', '.class', a tag, or one inside another ('#rows
# input'); anything else is an element of its own, made when first asked for.
DOM = r"""
const fs = require('fs');
class Classes {
  constructor(names) { this.names = new Set(names); }
  add(name) { this.names.add(name); }
  remove(name) { this.names.delete(name); }
  toggle(name, on = !this.names.has(name)) { if (on) this.names.add(name); else this.names.delete(name); return on; }
  contains(name) { return this.names.has(name); }
}
function node(tag = 'div') {
  return {
    tagName: tag.toUpperCase(), nodeType: 1, classList: new Classes([]), children: [], attributes: {}, dataset: {},
    style: {}, textContent: '', value: '', src: '', checked: false, disabled: false, readOnly: false, hidden: false,
    open: false, handlers: {}, parts: {},
    set className(value) { this.classList = new Classes(String(value).split(' ').filter(Boolean)); },
    replaceChildren(...nodes) {
      this.children = nodes.flat().filter(n => n !== null && n !== undefined && n !== false);
      const chosen = this.children.find(n => n.tagName === 'OPTION' && 'selected' in n.attributes);
      if (chosen) this.value = chosen.attributes.value;
    },
    append(...nodes) { this.children.push(...nodes.flat()); },
    prepend(...nodes) { this.children.unshift(...nodes.flat()); },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name]; },
    addEventListener(type, fn) { (this.handlers[type] = this.handlers[type] || []).push(fn); },
    removeEventListener(type, fn) { this.handlers[type] = (this.handlers[type] || []).filter(f => f !== fn); },
    focus() {}, scrollIntoView() {}, click() {},
    get firstChild() { return this.children[0]; },
  };
}
async function fire(target, type) {
  for (const handler of target.handlers[type] || []) await handler({ type, target, preventDefault() {} });
}
function el(tag, attrs = {}, ...children) {
  const made = node(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value === null || value === undefined) continue;
    if (key === 'class') made.className = value;
    else if (key === 'text') made.textContent = value;
    else if (key.startsWith('on') && typeof value === 'function') made.addEventListener(key.slice(2), value);
    else {
      made.setAttribute(key, value === true ? '' : value);
      if (key.startsWith('data-')) {
        made.dataset[key.slice(5).replace(/-(\w)/g, (_m, c) => c.toUpperCase())] = String(value);
      }
      if (key === 'checked' || key === 'hidden' || key === 'disabled') made[key] = true;
      if (key === 'value') made.value = String(value);
    }
  }
  made.replaceChildren(...children.flat().map(c => (c === null || c === undefined || c === false || c.nodeType) ? c
    : String(c)));
  return made;
}
const kids = n => (n && typeof n === 'object' ? n.children : []).filter(c => c && typeof c === 'object');
const below = n => kids(n).flatMap(c => [c, ...below(c)]);
const text = n => (typeof n === 'string' ? n : n.textContent + n.children.map(text).join(''));
function is(n, simple) {
  if (simple.startsWith('#')) return n.attributes.id === simple.slice(1);
  if (simple.startsWith('.')) return n.classList.contains(simple.slice(1));
  return /^[a-z]+$/.test(simple) && n.tagName === simple.toUpperCase();
}
function documentOf() {
  const elements = {};
  const top = selector => elements[selector] || (elements[selector] = Object.assign(node(), {
    attributes: /^#[\w-]+$/.test(selector) ? { id: selector.slice(1) } : {} }));
  const $$ = (selector, root) => {
    let parts = selector.trim().split(/\s+/), found;
    if (!root && /^#[\w-]+$/.test(parts[0])) { found = [top(parts[0])]; parts = parts.slice(1); }
    else found = root ? [root] : Object.values(elements);
    for (const part of parts) found = [...new Set(found.flatMap(below).filter(n => is(n, part)))];
    return found;
  };
  const $ = (selector, root) => {
    if (!root && /^#[\w-]+$/.test(selector)) return top(selector);
    return $$(selector, root)[0] || (root ? (root.parts[selector] = root.parts[selector] || node()) : top(selector));
  };
  return { $, $$ };
}
const notice = (kind, words) => el('div', { class: `notice ${kind}`, text: words });
const settle = () => new Promise(resolve => setTimeout(resolve, 50));
"""
PROFILE_HARNESS = DOM + r"""
// Runs the Profile page's whole script (Settings > Certifications section off), then what each case asks, and prints
// what the cards show and what the page sent.
async function show(c) {
  const page = fs.readFileSync(process.argv[2], 'utf8');
  const code = page.slice(page.indexOf('<script>') + '<script>'.length, page.lastIndexOf('</script>'))
    .replace('{{ certifications_on | tojson }}', 'false');
  const { $, $$ } = documentOf();
  const calls = [];
  const api = async (path, options = {}) => {
    calls.push([options.method || 'GET', path, options.body === undefined ? null : options.body]);
    if (path === '/api/resumes' && !options.method) return { resumes: c.resumes };
    if (path === '/api/settings') return { resolved: { parser: 'reader-a' } };
    if (path === '/api/models') return { models: [{ id: 'reader-a', label: 'Reader A', demo: true }] };
    const answer = (c.answers || {})[path];
    if (answer && answer.error) throw new Error(answer.error);
    if (answer) return answer;
    throw new Error(`No stand-in for ${path}`);
  };
  const exposed = '\nreturn { load };';
  const profile = new Function('$', '$$', 'el', 'api', 'notice', 'ME', 'window', 'document', 'confirm', 'alert',
    'FormData', code + exposed)($, $$, el, api, notice, { id: '1', admin: c.admin }, { addEventListener() {} },
    { addEventListener() {} }, () => false, () => {}, class {});
  await settle();
  const card = id => $$('.card').find(n => $$('.link-list', n).concat($$('.project-list', n))
    .some(box => box.attributes['data-resume'] === String(id)));
  const seen = id => {
    const box = $$('.link-list').find(n => n.attributes['data-resume'] === String(id));
    const line = kids(card(id)).find(n => n.classList.contains('small'));
    return {
      contact: line ? { text: line.textContent, hidden: line.hidden } : null,
      box: box ? { summary: text($('summary', box)), open: box.open,
        inputs: $$('input', box).map(input => [input.value, input.attributes.placeholder, input.attributes.maxlength]),
        labels: $$('label', box).map(label => text(label)), status: text(kids($('.row', box)).slice(-1)[0]) } : null,
    };
  };
  const out = { start: seen(c.resumes[0].id), calls: calls.map(([method, path]) => `${method} ${path}`) };
  for (const step of c.steps || []) {
    const box = $$('.link-list').find(n => n.attributes['data-resume'] === String(step.resume));
    if (step.type) Object.entries(step.type).forEach(([i, value]) => { $$('input', box)[Number(i)].value = value; });
    if (step.open) box.open = true;
    calls.length = 0;
    if (step.save) await fire($$('button', box).find(b => b.textContent === 'Save links'), 'click');
    if (step.load) await profile.load();
    out[step.name] = { ...seen(step.resume), calls: JSON.parse(JSON.stringify(calls)) };
  }
  return out;
}
(async () => {
  const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const out = {};
  for (const [name, c] of Object.entries(cases)) out[name] = await show(c);
  console.log(JSON.stringify(out));
})().catch(err => { console.error(err); process.exit(1); });
"""


def node_run(tmp_path, harness: str, page: Path, cases: dict) -> dict:
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    done = subprocess.run(["node", str(tmp_path / "harness.js"), str(page), str(tmp_path / "cases.json")],
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def entry_of(resume_id: int, contact: dict, name: str = "Taylor Quinn") -> dict:
    """A resume as GET /api/resumes lists it."""
    return {"id": resume_id, "created_at": "2026-11-20T10:00:00+00:00", "filename": "taylor.docx", "method": "",
            "read_with": "", "warnings": [], "expired": [],
            "profile": {"name": name, "contact": contact, "certifications": [], "education": [],
                        "companies": [{"name": "Northwind Labs", "start": "Mar 2022", "end": "Present", "mode": "",
                                       "location": "", "projects": ""}]}}


SAVED = {"email": "taylor@example.com", "phone": "(555) 010-4477", "linkedin": LINKEDIN, "github": GITHUB,
         "website": WEBSITE, "location": "Austin, TX"}
CARD_LINE = f"taylor@example.com | (555) 010-4477 | {LINKEDIN} | {GITHUB} | {WEBSITE} | Austin, TX"


@pytest.mark.skipif(shutil.which("node") is None, reason="runs the Profile page's own script with Node.js")
def test_the_profile_page_has_the_links_box_for_the_admin_only(tmp_path):
    typed = {"0": f"  https://www.{LINKEDIN} ", "1": "gitlab.com/tquinn", "2": ""}
    answer = {"resume": entry_of(7, {**SAVED, "linkedin": f"https://www.{LINKEDIN}", "github": "gitlab.com/tquinn",
                                     "website": ""})}
    cases = {
        "user": {"admin": False, "resumes": [entry_of(7, SAVED)]},
        "admin": {"admin": True, "resumes": [entry_of(7, SAVED)], "answers": {"/api/resumes/7/links": answer},
                  "steps": [{"name": "saved", "resume": 7, "type": typed, "save": True}]},
        "too long": {"admin": True, "resumes": [entry_of(7, SAVED)], "steps": [
            {"name": "refused", "resume": 7, "type": {"1": "github.com/" + "x" * 300}, "save": True}]},
        "stale": {"admin": True, "resumes": [entry_of(7, SAVED)], "answers": {
            "/api/resumes/7/links": {"error": "The links of this resume were changed in another tab."}},
                  "steps": [{"name": "refused", "resume": 7, "type": {"2": ""}, "save": True}]},
        "redrawn": {"admin": True, "resumes": [entry_of(7, SAVED)],
                    "steps": [{"name": "again", "resume": 7, "type": {"1": "gitlab.com/tquinn"}, "open": True,
                               "load": True}]},
        "no links yet": {"admin": True, "resumes": [entry_of(7, {**SAVED, "linkedin": "", "github": "", "website": "",
                                                                 "email": "", "phone": "", "location": ""})]},
    }
    seen = node_run(tmp_path, PROFILE_HARNESS, PROFILE_PAGE, cases)

    # another account: the card as before, with every link the upload read; no box
    assert seen["user"]["start"] == {"contact": {"text": CARD_LINE, "hidden": False}, "box": None}
    assert seen["user"]["calls"] == ["GET /api/resumes"]
    # the admin: the box holds what was saved, each box at most 300 characters
    start = seen["admin"]["start"]
    assert start["contact"] == {"text": CARD_LINE, "hidden": False}
    assert start["box"]["summary"] == "Links (3 of 3)" and start["box"]["labels"] == ["LinkedIn", "GitHub", "Website"]
    assert start["box"]["inputs"] == [[LINKEDIN, "linkedin.com/in/your-name", "300"],
                                      [GITHUB, "github.com/your-name", "300"], [WEBSITE, "your-name.dev", "300"]]
    # Save sends each box trimmed and all three as loaded; the box and the card's contact line then show what was saved
    saved = seen["admin"]["saved"]
    assert saved["calls"] == [["PUT", "/api/resumes/7/links", {
        "linkedin": f"https://www.{LINKEDIN}", "github": "gitlab.com/tquinn", "website": "",
        "loaded": {"linkedin": LINKEDIN, "github": GITHUB, "website": WEBSITE}}]]
    assert [value for value, _example, _most in saved["box"]["inputs"]] == [f"https://www.{LINKEDIN}",
                                                                            "gitlab.com/tquinn", ""]
    assert saved["box"]["summary"] == "Links (2 of 3)"
    assert saved["box"]["status"] == "Saved. New resumes and cover letters print the ones you tick on the Tailor page."
    assert saved["contact"]["text"] == (
        f"taylor@example.com | (555) 010-4477 | https://www.{LINKEDIN} | gitlab.com/tquinn | Austin, TX")
    # too long: said in plain words before anything is sent; a refusal from the server is shown as it is
    refused = seen["too long"]["refused"]
    assert refused["calls"] == [] and refused["box"]["status"] == (
        "The GitHub box has 311 characters, and a link can have up to 300. Shorten it, then press Save links again.")
    assert seen["stale"]["refused"]["box"]["status"] == "The links of this resume were changed in another tab."
    # typed but not saved: kept when the cards are drawn again (another upload, a save in another box)
    again = seen["redrawn"]["again"]
    assert again["calls"] == [["GET", "/api/resumes", None]] and again["box"]["open"]
    assert [value for value, _example, _most in again["box"]["inputs"]] == [LINKEDIN, "gitlab.com/tquinn", WEBSITE]
    # nothing read yet: empty boxes to type in, and no empty contact line
    empty = seen["no links yet"]["start"]
    assert empty["contact"] == {"text": "", "hidden": True} and empty["box"]["summary"] == "Links (0 of 3)"


TAILOR_HARNESS = DOM + r"""
// Runs the Tailor page's whole script, then what each case asks (ticks, another resume, the buttons), and prints the
// tick boxes, what the page remembered and what it sent.
async function show(c) {
  const page = fs.readFileSync(process.argv[2], 'utf8');
  const code = page.slice(page.indexOf('<script>') + '<script>'.length, page.lastIndexOf('</script>'));
  const { $, $$ } = documentOf();
  ['#result', '#toggles', '#timing', '#certs', '#steps', '#links'].forEach(id => $(id).classList.add('hidden'));
  const store = { ...(c.store || {}) }, calls = [];
  let made = 6, busyBoxes = null;
  const run = (id, letter) => ({ run_id: id, status: 'ok', pages: 1, fits: true, coverage: { score: 0.9 },
    cover_letter: letter ? { paragraphs: ['I build data pipelines.'] } : null, spelling: 'us', certifications: [] });
  const api = async (path, options = {}) => {
    calls.push([path, options.body === undefined ? null : options.body]);
    if (path === '/api/resumes') return { resumes: c.resumes };
    if (path === '/api/settings') return { resolved: { resume: 'model-a', cover_letter: 'model-a' } };
    if (path === '/api/models') return { models: [{ id: 'model-a', label: 'Model A', demo: false }] };
    if (path === '/api/tailor') {
      busyBoxes = $$('#link-rows input').map(box => box.disabled);
      return run(++made, Boolean(options.body.cover_letter));
    }
    const letter = /^\/api\/runs\/(\d+)\/cover-letter$/.exec(path);
    if (letter) return run(Number(letter[1]), true);
    throw new Error(`No stand-in for ${path}`);
  };
  const history = { state: null, replaceState(state) { this.state = JSON.parse(JSON.stringify(state)); } };
  const exposed = '\nreturn { create, get current() { return current; } };';
  const tailor = new Function('$', '$$', 'el', 'api', 'finished', 'notice', 'chip', 'money', 'ME', 'recallValue',
    'rememberValue', 'location', 'history', 'window', 'document', code + exposed)(
    $, $$, el, api, async data => data, notice, words => el('span', { text: words }), String,
    { id: '1', admin: c.admin }, name => (name in store ? store[name] : null),
    (name, value) => { store[name] = String(value); }, { search: '' }, history,
    { matchMedia: () => ({ matches: false }), addEventListener() {} }, { addEventListener() {} });
  await settle();
  const seen = () => ({
    shown: !$('#links').classList.contains('hidden'),
    boxes: $$('#link-rows label').map(label => [...$$('span', label).map(text), $('input', label).checked,
      $('input', label).disabled]),
    unticked: store.untickedLinks === undefined ? null : JSON.parse(store.untickedLinks),
    messages: $('#messages').children.map(text),
  });
  const out = { start: { ...seen(), calls: calls.map(([path]) => path) } };
  for (const step of c.steps || []) {
    calls.length = 0;
    busyBoxes = null;
    if (step.tick) {
      const box = $$('#link-rows input').find(input => input.attributes.id === `link-${step.tick[0]}`);
      box.checked = step.tick[1];
      await fire(box, 'change');
    }
    if (step.choose) { $('#resume').value = String(step.choose); await fire($('#resume'), 'change'); }
    if (step.create) {
      $('#jd').value = 'Senior Data Engineer at Fabrikam. Build pipelines in Python and SQL on AWS for analysts.';
      $('#company').value = 'Fabrikam';
      $('#job-link').value = 'https://jobs.example.com/data-engineer';
      await tailor.create(...step.create);
    }
    const current = tailor.current;
    out[step.name] = { ...seen(), calls: JSON.parse(JSON.stringify(calls)), busyBoxes,
      current: current && { run: current.run.run_id, links: current.links === undefined ? 'unset' : current.links },
      state: history.state };
  }
  return out;
}
(async () => {
  const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const out = {};
  for (const [name, c] of Object.entries(cases)) out[name] = await show(c);
  console.log(JSON.stringify(out));
})().catch(err => { console.error(err); process.exit(1); });
"""
LINK_JD = "Senior Data Engineer at Fabrikam. Build pipelines in Python and SQL on AWS for analysts."
LINK_JOB = {"jd": LINK_JD, "profile_id": 5, "spelling": "us", "company_name": "Fabrikam",
            "job_link": "https://jobs.example.com/data-engineer", "cover_letter": True}


@pytest.mark.skipif(shutil.which("node") is None, reason="runs the Tailor page's own script with Node.js")
def test_the_tailor_page_ticks_the_chosen_resumes_links_for_the_admin_only(tmp_path):
    every = entry_of(5, {**SAVED, "linkedin": f"https://www.{LINKEDIN}/"})
    newer = entry_of(6, {**SAVED, "linkedin": "", "github": "github.com/taylorquinn-labs", "website": ""})
    none = entry_of(8, {**SAVED, "linkedin": "", "github": "", "website": ""})
    remembered = json.dumps([f"github {GITHUB}", "website portfolio.example"])  # unticked before
    cases = {
        "admin": {"admin": True, "resumes": [every, newer], "steps": [
            {"name": "website off", "tick": ["website", False]},
            {"name": "made", "create": [True]},
            {"name": "letter again", "create": [True, True]},
            {"name": "website on", "tick": ["website", True]},
            {"name": "other links", "create": [True, True]},
            {"name": "other resume", "choose": 6},
        ]},
        "remembered": {"admin": True, "resumes": [every, newer], "store": {"untickedLinks": remembered}, "steps": [
            {"name": "other resume", "choose": 6},
        ]},
        "no saved links": {"admin": True, "resumes": [none], "steps": [{"name": "made", "create": [True]}]},
        "user": {"admin": False, "resumes": [every], "steps": [
            {"name": "made", "create": [True]},
            {"name": "letter again", "create": [True, True]},
        ]},
    }
    seen = node_run(tmp_path, TAILOR_HARNESS, TAILOR_PAGE, cases)
    shown = [["LinkedIn", LINKEDIN, True, False], ["GitHub", GITHUB, True, False], ["Website", WEBSITE, True, False]]

    # the first time every saved link is ticked, as the files print it; opening the page asks nothing new
    admin = seen["admin"]
    assert admin["start"] == {"shown": True, "boxes": shown, "unticked": None, "messages": [],
                              "calls": ["/api/resumes", "/api/settings", "/api/models"]}
    # an untick is remembered (per account) by its address
    assert admin["website off"]["unticked"] == [f"website {WEBSITE}"]
    assert admin["website off"]["boxes"][2] == ["Website", WEBSITE, False, False]
    # the request names the ticked links; the boxes wait while it runs; the run and the page's address keep them
    made = admin["made"]
    assert made["calls"] == [["/api/tailor", {**LINK_JOB, "links": ["linkedin", "github"]}]]
    assert made["busyBoxes"] == [True, True, True] and made["boxes"][0][3] is False and made["messages"] == []
    assert made["current"] == {"run": 7, "links": ["linkedin", "github"]}
    assert made["state"]["links"] == ["linkedin", "github"]
    # Add or Rewrite cover letter keeps the resume made with the same links, and makes it again for other links
    assert admin["letter again"]["calls"] == [
        ["/api/runs/7/cover-letter", {"job_link": "https://jobs.example.com/data-engineer"}]]
    assert admin["website on"]["unticked"] == []
    assert admin["other links"]["calls"] == [
        ["/api/tailor", {**LINK_JOB, "links": ["linkedin", "github", "website"]}]]
    assert admin["other links"]["current"] == {"run": 8, "links": ["linkedin", "github", "website"]}
    # another resume: its own links, drawn again
    assert admin["other resume"]["boxes"] == [["GitHub", "github.com/taylorquinn-labs", True, False]]

    # a link unticked before stays unticked; one newly saved (another address) starts ticked
    remembered_start = seen["remembered"]["start"]["boxes"]
    assert [box[2] for box in remembered_start] == [True, False, True]
    assert seen["remembered"]["other resume"]["boxes"] == [["GitHub", "github.com/taylorquinn-labs", True, False]]

    # no saved link: no boxes, and no choice sent (every saved link is printed)
    nothing = seen["no saved links"]
    assert nothing["start"]["shown"] is False and nothing["start"]["boxes"] == []
    assert nothing["made"]["calls"] == [["/api/tailor", {**LINK_JOB, "profile_id": 8}]]
    assert nothing["made"]["current"]["links"] is None

    # another account: no boxes and no choice, as before
    user = seen["user"]
    assert user["start"] == {"shown": False, "boxes": [], "unticked": None, "messages": [], "calls": ["/api/resumes"]}
    assert user["made"]["calls"] == [["/api/tailor", LINK_JOB]] and user["made"]["current"]["links"] is None
    assert user["letter again"]["calls"] == [
        ["/api/runs/7/cover-letter", {"job_link": "https://jobs.example.com/data-engineer"}]]


def test_the_tailor_page_sends_the_choice_with_the_job(client):
    page = client.get("/tailor").text
    assert 'class="links hidden" id="links"' in page and 'id="link-rows"' in page
    assert "...(links ? { links } : {})" in page and "sameLinks(current.links, chosenLinks())" in page
    assert "company_name: company, job_link: link" in page and "cover_letter: true" in page  # as before
