"""Clickable contact details. In the PDF (resume and cover letter) the email, the LinkedIn address and the website
(GitHub or portfolio) are link annotations over exactly their own text, and in the Word copy real hyperlinks; both look
exactly as before (the same text, colour, sizes, centring and line breaks), and the phone and the place stay plain
text. A link is made only from a value that is safe to link. The rule-based reader reads a GitHub, GitLab or portfolio
address from a resume's header into the website, and the Profile page shows it. Every person, address and company
here is invented."""
import io
import json
import random
import shutil
import subprocess
from pathlib import Path

import pdfplumber
import pytest
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml.ns import qn
from docx.shared import RGBColor
from pypdf import PdfReader

from app.parsing.heuristic import parse_resume_text
from app.parsing.resume_parser import extract_text
from app.pipeline import validate
from app.pipeline.fit import LADDER, styles
from app.render import pdf as pdf_module
from app.render.docx import cover_docx, resume_docx
from app.render.pdf import (
    CAP_HEIGHT,
    CONTACT_SIZE,
    DESIGN_SIZE,
    LINE,
    MUTED,
    contact_line,
    contact_lines,
    render_cover_pdf,
    render_resume_pdf,
)
from app.schemas import Contact, CoverLetter, FitStyle

from .test_api import save_profile
from .test_api import tailor as tailor_request
from .test_validate import make_output

CONTACT = Contact(email="taylor@example.com", phone="(555) 010-4477",
                  linkedin="https://www.linkedin.com/in/taylorquinn", website="github.com/taylorquinn",
                  location="Austin, TX")
# each clickable item as printed -> the address it opens
LINKS = {"taylor@example.com": "mailto:taylor@example.com",
         "linkedin.com/in/taylorquinn": "https://www.linkedin.com/in/taylorquinn",
         "github.com/taylorquinn": "https://github.com/taylorquinn"}
PLAIN = ("Austin, TX", "(555) 010-4477")  # never links
LETTER = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["I build data pipelines at Northwind Labs."],
                     closing="Sincerely,")
# long enough to break between its items: the email ends the first line, LinkedIn and GitHub make the second
LONG = Contact(email="taylor.quinn.data.engineering@example.com", phone="+1 (555) 010-4477",
               linkedin="linkedin.com/in/taylor-quinn-data-engineer-austin",
               website="https://github.com/taylorquinn-data-engineering", location="San Francisco Bay Area, CA")
LONG_LINKS = {"taylor.quinn.data.engineering@example.com": "mailto:taylor.quinn.data.engineering@example.com",
              "linkedin.com/in/taylor-quinn-data-engineer-austin":
                  "https://linkedin.com/in/taylor-quinn-data-engineer-austin",
              "github.com/taylorquinn-data-engineering": "https://github.com/taylorquinn-data-engineering"}


def taylor(profile, contact: Contact = CONTACT):
    """The shared resume as Taylor Quinn's, with these contact details."""
    person = profile.model_copy(deep=True, update={"name": "Taylor Quinn", "contact": contact})
    return validate.build_resume(person, make_output())


@pytest.fixture
def resume(profile):
    return taylor(profile)


def page_links(pdf: bytes, page: int = 0) -> list[tuple[str, list[float]]]:
    """(the address, its rectangle [x0, y0, x1, y1] in points from the bottom left) of each link on a page, in order.
    Every link is an invisible one: no border is drawn, so the page looks as it did without it."""
    found = []
    for ref in PdfReader(io.BytesIO(pdf)).pages[page].get("/Annots") or []:
        annot = ref.get_object()
        assert annot["/Subtype"] == "/Link" and [float(width) for width in annot["/Border"]] == [0, 0, 0]
        found.append((str(annot["/A"]["/URI"]), [float(edge) for edge in annot["/Rect"]]))
    return found


def drawn(pdf: bytes, texts) -> dict[str, tuple[float, float, float, float]]:
    """Where each text is drawn on page 1, from its characters: (left, baseline, right, size), in points from the
    bottom left as a link's rectangle is."""
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        chars = document.pages[0].chars
    page = "".join(char["text"] for char in chars)
    found = {}
    for text in texts:
        start = page.index(text)
        first, last = chars[start], chars[start + len(text) - 1]
        found[text] = (first["x0"], first["matrix"][5], last["x1"], first["size"])
    return found


def assert_links_on_their_text(pdf: bytes, links: dict[str, str]) -> None:
    """Exactly one link per clickable item, in reading order, each over its own text: the same left and right edges,
    and its line's height around the letters (from below the baseline to above the capitals). None on the phone or
    the place."""
    found = page_links(pdf)
    assert [target for target, _rect in found] == list(links.values())
    places = drawn(pdf, [*links, *PLAIN])
    for (target, (x0, y0, x1, y1)), text in zip(found, links):
        left, baseline, right, size = places[text]
        assert x0 == pytest.approx(left, abs=0.02) and x1 == pytest.approx(right, abs=0.02), (text, target)
        assert y0 < baseline and baseline + CAP_HEIGHT * size < y1, (text, target)
        assert y1 - y0 == pytest.approx(LINE * size, abs=0.02)  # the line's own height, never the next line's
    for text in PLAIN:
        left, _baseline, right, _size = places[text]
        assert all(right <= x0 or x1 <= left for _target, (x0, _y0, x1, _y1) in found), text


# ------------------------------------------------------------------ the PDF
@pytest.mark.parametrize("step", [0, len(LADDER) + 1])
@pytest.mark.parametrize("paper", ["Letter", "A4"])
def test_the_resume_links_the_email_linkedin_and_github_over_their_text(resume, paper, step):
    """At the design's sizes and at the last step, on both papers."""
    pdf, pages = render_resume_pdf(resume, styles(paper)[step])
    assert pages == 1
    assert_links_on_their_text(pdf, LINKS)
    first = PdfReader(io.BytesIO(pdf)).pages[0].extract_text().splitlines()
    assert first[2] == contact_line(CONTACT)  # the text as before, short: no https:// or www. for a viewer to link


def test_the_cover_letter_links_them_too(resume):
    pdf, pages = render_cover_pdf(resume, LETTER)
    assert pages == 1
    assert_links_on_their_text(pdf, LINKS)


def test_a_contact_line_broken_in_two_has_each_link_on_its_own_line(profile):
    """The line breaks between items as before; each link is on the line its item is drawn on."""
    resume = taylor(profile, LONG)
    lines = contact_lines(LONG, CONTACT_SIZE * DESIGN_SIZE, 612 - 2 * 72 * styles("Letter")[0].margin_in)
    assert lines == ["San Francisco Bay Area, CA · +1 (555) 010-4477 · taylor.quinn.data.engineering@example.com",
                     "linkedin.com/in/taylor-quinn-data-engineer-austin · github.com/taylorquinn-data-engineering"]
    for pdf in (render_resume_pdf(resume, styles("Letter")[0])[0], render_cover_pdf(resume, LETTER)[0]):
        found = page_links(pdf)
        assert [target for target, _rect in found] == list(LONG_LINKS.values())
        email, linkedin, github = (rect for _target, rect in found)
        assert email[1] > linkedin[3] - 0.01 and linkedin[1:4:2] == github[1:4:2]  # line 1, then both on line 2
        with pdfplumber.open(io.BytesIO(pdf)) as document:
            text = document.pages[0].extract_text().splitlines()
        assert text[2:4] == lines


def test_the_links_change_nothing_drawn(resume, monkeypatch):
    """The pages are drawn exactly as without links: each page's drawing (its content stream) is byte for byte the
    one the same resume and letter get with every link turned off, so the text, colours, sizes, centring and line
    breaks are the same. Only the link annotations are added."""
    def files():
        return [render_resume_pdf(resume, style)[0] for style in (styles("Letter")[0], FitStyle())] + [
            render_cover_pdf(resume, LETTER)[0]]

    linked = files()
    monkeypatch.setattr(pdf_module, "email_link", lambda _value: "")
    monkeypatch.setattr(pdf_module, "web_link", lambda _value: "")
    plain = files()
    for with_links, without in zip(linked, plain):
        assert len(page_links(with_links)) == 3 and page_links(without) == []
        a, b = PdfReader(io.BytesIO(with_links)), PdfReader(io.BytesIO(without))
        assert [page.get_contents().get_data() for page in a.pages] == [
            page.get_contents().get_data() for page in b.pages]


def test_a_value_unsafe_to_link_stays_plain_text(profile):
    """A website or LinkedIn value with another scheme, a space or angle brackets is printed as before, unlinked."""
    contact = CONTACT.model_copy(update={"website": "javascript:alert(1)", "linkedin": "linkedin.com/in/<taylor>"})
    pdf = render_resume_pdf(taylor(profile, contact), FitStyle())[0]
    assert [target for target, _rect in page_links(pdf)] == ["mailto:taylor@example.com"]
    first = PdfReader(io.BytesIO(pdf)).pages[0].extract_text().splitlines()
    assert first[2] == contact_line(contact) and "javascript:alert(1)" in first[2]


# ------------------------------------------------------------------ the Word copy
def contact_paragraph(data: bytes, email: str = "taylor@example.com"):
    doc = Document(io.BytesIO(data))
    return doc, next(paragraph for paragraph in doc.paragraphs if email in paragraph.text)


@pytest.mark.parametrize("kind", ["resume", "cover letter"])
def test_word_has_real_hyperlinks_in_the_lines_own_look(resume, kind):
    """w:hyperlink elements on external relationships, around runs in the contact line's own font size and muted
    colour, not bold, with no Hyperlink style and no underline: the line looks as before."""
    data = resume_docx(resume, FitStyle()) if kind == "resume" else cover_docx(resume, LETTER)
    doc, paragraph = contact_paragraph(data)
    assert paragraph.text == contact_line(CONTACT)  # the same text as before
    links = paragraph.hyperlinks
    assert [(link.text, link.address) for link in links] == list(LINKS.items())
    for link in links:
        relationship = doc.part.rels[link._hyperlink.rId]
        assert relationship.is_external and relationship.reltype == RELATIONSHIP_TYPE.HYPERLINK
    plain = [run for run in paragraph.runs if run.text]
    linked = [run for link in links for run in link.runs]
    assert "".join(run.text for run in plain) == "Austin, TX · (555) 010-4477 ·  ·  · "
    size = (CONTACT_SIZE * (FitStyle().font_pt if kind == "resume" else DESIGN_SIZE) * 2) // 1 / 2
    for run in plain + linked:
        assert run.font.size.pt == size and run.font.color.rgb == RGBColor.from_string(MUTED[1:]) and not run.bold
    for run in linked:
        assert run._r.find(f"{qn('w:rPr')}/{qn('w:rStyle')}") is None  # never Word's blue Hyperlink style
        assert run.font.underline is False  # w:u none: no underline, whatever a program does with links


def test_word_links_on_a_contact_line_broken_in_two(profile):
    resume = taylor(profile, LONG)
    _doc, paragraph = contact_paragraph(resume_docx(resume, styles("Letter")[0]), LONG.email)
    assert paragraph.text == "\n".join(contact_lines(LONG, CONTACT_SIZE * styles("Letter")[0].font_pt,
                                                     612 - 2 * 72 * styles("Letter")[0].margin_in))
    assert [(link.text, link.address) for link in paragraph.hyperlinks] == list(LONG_LINKS.items())


# ------------------------------------------------------------------ what may become a link
@pytest.mark.parametrize("value,target", [
    ("taylor@example.com", "mailto:taylor@example.com"),
    ("  taylor.quinn+jobs@example.co.uk ", "mailto:taylor.quinn+jobs@example.co.uk"),
    ("josé@example.com", "mailto:jos%C3%A9@example.com"),  # a link holds ASCII only: the rest percent-encoded
    # anything else is no email the app reads: no link
    ("Taylor Quinn <taylor@example.com>", ""), ("<taylor@example.com>", ""), ("taylor@example", ""),
    ("mailto:taylor@example.com", ""), ("taylor@example.com?subject=Hello", ""), ('"taylor"@example.com', ""),
    ("tay lor@example.com", ""), ("taylor@example.com\u200b", ""), ("javascript:alert(1)//taylor@example.com", ""),
    ("", ""),
])
def test_an_email_is_linked_only_when_the_app_reads_it_as_one(value, target):
    assert pdf_module.email_link(value) == target


@pytest.mark.parametrize("value,target", [
    # no scheme: https:// and the form printed
    ("github.com/taylorquinn", "https://github.com/taylorquinn"),
    ("www.github.com/taylorquinn/", "https://github.com/taylorquinn"),
    ("linkedin.com/in/taylorquinn", "https://linkedin.com/in/taylorquinn"),
    ("taylorquinn.dev/#projects", "https://taylorquinn.dev/#projects"),
    ("linkedin.com/in/josé-quinn", "https://linkedin.com/in/jos%C3%A9-quinn"),
    # http:// or https:// kept as written (a site may have no https)
    ("https://github.com/taylorquinn", "https://github.com/taylorquinn"),
    ("https://www.linkedin.com/in/taylorquinn/", "https://www.linkedin.com/in/taylorquinn/"),
    ("HTTPS://WWW.LinkedIn.com/in/TaylorQuinn", "https://WWW.LinkedIn.com/in/TaylorQuinn"),
    ("http://taylorquinn.dev", "http://taylorquinn.dev"),
    ("  https://taylorquinn.dev:8443/cv?lang=en  ", "https://taylorquinn.dev:8443/cv?lang=en"),
    # another scheme: no link
    ("javascript:alert(1)", ""), ("JavaScript:alert(document.cookie)", ""), ("vbscript:msgbox(1)", ""),
    ("data:text/html,<script>alert(1)</script>", ""), ("data:text/html;base64,PHNjcmlwdD4=", ""),
    ("file:///etc/passwd", ""), ("file://server/share/cv.pdf", ""), ("ftp://example.com/cv.pdf", ""),
    ("mailto:taylor@example.com", ""), ("https:github.com/taylorquinn", ""), ("localhost:8000/admin", ""),
    # spaces, quotes, angle brackets and other characters a link must not hold: no link
    ("github.com/taylor quinn", ""), ('github.com/taylorquinn"onmouseover="alert(1)', ""),
    ("github.com/<b>taylor</b>", ""), ("github.com/taylor'quinn", ""), ("github.com/taylor`quinn", ""),
    ("github.com/taylor\\quinn", ""), ("github.com/taylor\u200bquinn", ""), ("github.com/taylor\nquinn", ""),
    ("github.com/taylor\u00a0quinn", ""), ("github.com/“taylorquinn”", ""),
    # no host name a browser could open: no link
    ("N/A", ""), ("//evil.example/x", ""), ("https://", ""), ("https://taylor@evil.example/", ""),
    ("taylor@example.com", ""), ("https://[::1]/", ""), ("https://[::1", ""), ("", ""),
])
def test_a_web_address_is_linked_only_when_it_is_safe(value, target):
    assert pdf_module.web_link(value) == target


def test_every_link_made_is_a_web_or_mail_address_in_plain_ascii():
    """Whatever a value holds, a link made of it opens a web page or writes an email, and holds no space, quote,
    angle bracket or other character outside printable ASCII."""
    rng = random.Random(11)
    starts = ["", "", "https://", "http://", "HTTPS://", "www.", "//", "javascript:", "data:", "file:", "mailto:",
              "taylor@", " ", '"', "<"]
    hosts = ["github.com", "example.com", "taylorquinn.dev", "münchen.example", "localhost", "N/A", "[::1]"]
    pieces = ["/", "taylorquinn", "@", "example.com", ".", " ", '"', "'", "`", "<", ">", "\\", ":", "%0a", "é",
              "\u200b", "\u00a0", "#", "?", "x=1&y=2", "\t", "\n", "(", ")", "[", "]", "+", "-", "_", "王",
              "javascript:"]
    made = 0
    for _ in range(4000):
        value = rng.choice(starts) + rng.choice(hosts) + "".join(rng.choice(pieces) for _ in range(rng.randint(0, 4)))
        for target in (pdf_module.web_link(value), pdf_module.email_link(value)):
            if target:
                made += 1
                assert target.startswith(("https://", "http://", "mailto:")), (value, target)
                assert target.isascii() and target.isprintable(), (value, target)
                assert not set(target) & set(" \t\n\"'`<>\\"), (value, target)
    assert made > 100  # the pieces make links often enough for this to test something


# ------------------------------------------------------------------ the rule-based reader
BODY = ["SUMMARY", "Data engineer with ten years of pipelines.", "EXPERIENCE",
        "Data Engineer · Northwind Labs · Remote   Mar 2022 – Present", "• Built the pipelines."]


def read(*header: str, body: list[str] = BODY):
    return parse_resume_text("\n".join(["Taylor Quinn", *header, *body]))


@pytest.mark.parametrize("header,website", [
    (["Austin, TX | taylor@example.com | (555) 010-4477 | linkedin.com/in/taylorquinn | github.com/taylorquinn"],
     "github.com/taylorquinn"),
    (["Austin, TX · (555) 010-4477 · taylor@example.com",
      "linkedin.com/in/taylorquinn · https://github.com/taylorquinn"], "https://github.com/taylorquinn"),
    (["taylor@example.com | GitHub: github.com/taylorquinn"], "github.com/taylorquinn"),
    (["taylor@example.com, www.github.com/taylorquinn/"], "www.github.com/taylorquinn/"),
    (["taylor@example.com", "gitlab.com/taylorquinn"], "gitlab.com/taylorquinn"),
    (["taylor@example.com | Portfolio: https://taylorquinn.dev"], "https://taylorquinn.dev"),
    (["Austin, TX | www.taylorquinn.dev | taylor@example.com"], "www.taylorquinn.dev"),
    # a GitHub or GitLab address first, wherever it is in the header
    (["https://taylorquinn.dev | taylor@example.com", "github.com/taylorquinn"], "github.com/taylorquinn"),
    # LinkedIn's address is the LinkedIn one, never the website
    (["taylor@example.com | https://www.linkedin.com/in/taylorquinn"], ""),
    # conservative: no address read from a bare name with a dot, nor from words around it
    (["taylor@example.com | taylorquinn.dev"], ""),
    (["taylor@example.com", "Maintainer of github.com/northwind/etl-kit, used by 300 teams."], ""),
    (["taylor@example.com | see https://taylorquinn.dev for my work"], ""),
])
def test_the_rules_reader_reads_the_website_from_the_header(header, website):
    profile = read(*header)
    assert profile.contact.website == website
    assert profile.name == "Taylor Quinn" and profile.contact.email == "taylor@example.com"


def test_the_website_is_read_only_from_the_header():
    profile = read("taylor@example.com", body=[*BODY, "• Published the loader at github.com/taylorquinn/loader."])
    assert profile.contact.website == ""


def test_a_web_address_is_kept_out_of_the_place():
    profile = read("www.taylorquinn.dev Austin, TX", "taylor@example.com")
    assert profile.contact.location == "Austin, TX"


@pytest.mark.parametrize("first", ["https://taylorquinn.dev", "GitHub: github.com/taylorquinn"])
def test_a_line_of_web_addresses_is_never_the_name(first):
    profile = parse_resume_text("\n".join([first, "Taylor Quinn", "taylor@example.com", *BODY]))
    assert profile.name == "Taylor Quinn" and profile.contact.website == first.removeprefix("GitHub: ")


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_the_apps_own_files_read_back_with_the_github_address(resume, kind):
    data = render_resume_pdf(resume, FitStyle())[0] if kind == "pdf" else resume_docx(resume, FitStyle())
    contact = parse_resume_text(extract_text(data, f"resume.{kind}")).contact
    assert (contact.email, contact.phone, contact.linkedin, contact.website, contact.location) == (
        "taylor@example.com", "(555) 010-4477", "linkedin.com/in/taylorquinn", "github.com/taylorquinn", "Austin, TX")


# ------------------------------------------------------------------ the Profile page and the whole way through
PAGE = Path(__file__).parents[1] / "app/web/templates/profile.html"
CARD_HARNESS = r"""
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const code = page.slice(page.indexOf("// The resume card's contact line"), page.indexOf('function resumeCard('));
const contactText = new Function(code + '\nreturn contactText;')();
console.log(JSON.stringify(JSON.parse(fs.readFileSync(process.argv[3], 'utf8')).map(contactText)));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="runs the Profile page's own script with Node.js")
def test_the_profile_card_shows_the_website(tmp_path):
    cases = [CONTACT.model_dump(), Contact(email="taylor@example.com", website="github.com/taylorquinn").model_dump(),
             Contact(email="taylor@example.com", phone="(555) 010-4477").model_dump()]
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "card.js").write_text(CARD_HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(tmp_path / "card.js"), str(PAGE), str(tmp_path / "cases.json")],
                         capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 0, run.stderr
    everything = ("taylor@example.com | (555) 010-4477 | https://www.linkedin.com/in/taylorquinn | "
                  "github.com/taylorquinn | Austin, TX")  # the website after LinkedIn
    assert json.loads(run.stdout) == [everything, "taylor@example.com | github.com/taylorquinn",
                                      "taylor@example.com | (555) 010-4477"]
    assert "const contact = contactText(c);" in PAGE.read_text(encoding="utf-8")


def test_an_uploaded_github_address_is_clickable_in_the_tailored_files(client):
    """The owner's case: a resume with a GitHub address, uploaded and read with the rules, then tailored. The saved
    profile holds the address, and the resume PDF, its Word copy and the letter link it like the email."""
    header = "Taylor Quinn\nAustin, TX | taylor@example.com | (555) 010-4477 | github.com/taylorquinn\n\n"
    text = header + "EXPERIENCE\nNorthwind Labs | Remote\nMar 2022 - Present\n\nEDUCATION\nState University\n"
    entry = save_profile(client, text=text)
    assert entry["profile"]["contact"]["website"] == "github.com/taylorquinn"
    run = tailor_request(client)
    assert run.status_code == 200, run.text
    run_id = run.json()["run_id"]
    pdf = client.get(f"/api/runs/{run_id}/resume.pdf").content
    assert [target for target, _rect in page_links(pdf)] == ["mailto:taylor@example.com",
                                                             "https://github.com/taylorquinn"]
    word = client.get(f"/api/runs/{run_id}/resume.docx").content
    assert [link.address for link in contact_paragraph(word)[1].hyperlinks] == [
        "mailto:taylor@example.com", "https://github.com/taylorquinn"]
