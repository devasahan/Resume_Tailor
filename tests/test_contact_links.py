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


@pytest.mark.parametrize("change", [
    {"location": "\U0001F4CD Austin, TX"},  # an emoji the AI reader copied from a template's header
    {"phone": "\u260e (555) 010-4477"},  # a symbol: the phone comes before the email
    {"location": "\uf041 Austin, TX"},  # an icon font's character, in text taken from a PDF
    {"location": "東京都 Austin, TX"},  # a script Inter lacks: three characters
], ids=["emoji", "symbol", "icon font", "Japanese"])
def test_links_after_a_character_the_font_lacks_stay_on_their_text(profile, change):
    """fpdf2 leaves a character Inter lacks out of the page (no letter and no room), while measuring gives it the
    .notdef width (0.656 em): each link after one is still over its own text, not that much to the right of it."""
    resume = taylor(profile, CONTACT.model_copy(update=change))
    for pdf in (render_resume_pdf(resume, styles("Letter")[0])[0], render_resume_pdf(resume, FitStyle())[0],
                render_cover_pdf(resume, LETTER)[0]):
        assert_links_on_their_text(pdf, LINKS)


def test_a_link_whose_own_text_holds_a_character_the_font_lacks_ends_where_its_text_does(profile):
    """The link covers the letters drawn ('linkedin.com/in/-taylor': the two Japanese ones are left out of the page)
    and opens the address as written; the link after it is over its own text too."""
    resume = taylor(profile, CONTACT.model_copy(update={"linkedin": "linkedin.com/in/田中-taylor"}))
    links = {"taylor@example.com": "mailto:taylor@example.com",
             "linkedin.com/in/-taylor": "https://linkedin.com/in/%E7%94%B0%E4%B8%AD-taylor",
             "github.com/taylorquinn": "https://github.com/taylorquinn"}
    for pdf in (render_resume_pdf(resume, styles("Letter")[0])[0], render_resume_pdf(resume, FitStyle())[0],
                render_cover_pdf(resume, LETTER)[0]):
        assert_links_on_their_text(pdf, links)


@pytest.mark.parametrize("website,target", [("www.taylorquinn.dev", "https://www.taylorquinn.dev"),
                                            ("https://www.taylorquinn.dev/", "https://www.taylorquinn.dev/"),
                                            ("taylorquinn.dev", "https://taylorquinn.dev")])
def test_a_website_opens_the_host_it_was_written_with(profile, website, target):
    """Printed short ('taylorquinn.dev') whatever was written, and linked to the address as written, so a site that
    answers only on www opens the same with or without https:// before it: in the PDF and in the Word file."""
    resume = taylor(profile, CONTACT.model_copy(update={"website": website}))
    pdf = render_resume_pdf(resume, FitStyle())[0]
    assert page_links(pdf)[-1][0] == target
    assert PdfReader(io.BytesIO(pdf)).pages[0].extract_text().splitlines()[2].endswith(" · taylorquinn.dev")
    assert contact_paragraph(resume_docx(resume, FitStyle()))[1].hyperlinks[-1].address == target


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
    # no scheme: https:// and the value as written, never the shorter form printed: 'www.' and a last '/' stay, as
    # a site may answer only on www
    ("github.com/taylorquinn", "https://github.com/taylorquinn"),
    ("www.github.com/taylorquinn/", "https://www.github.com/taylorquinn/"),
    ("www.taylorquinn.dev", "https://www.taylorquinn.dev"),
    ("www.taylorquinn.dev/cv/", "https://www.taylorquinn.dev/cv/"),
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
    # no host name a browser could open: no link ('www.' and one name is none either, with or without https://)
    ("N/A", ""), ("//evil.example/x", ""), ("https://", ""), ("https://taylor@evil.example/", ""),
    ("taylor@example.com", ""), ("https://[::1]/", ""), ("https://[::1", ""), ("", ""),
    ("www.localhost", ""), ("https://www.localhost/", ""), ("www.mysite:8080/cv", ""),
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
    # an address written without https:// or www., as the app's own files print the website: as a last choice, and
    # only from a contact line (one with the email, a phone, LinkedIn or another address), or alone on the line after
    # one, where a contact line that wrapped puts it
    (["taylor@example.com | taylorquinn.dev"], "taylorquinn.dev"),
    (["Austin, TX · (555) 010-4477 · taylor@example.com", "linkedin.com/in/taylorquinn · taylorquinn.dev"],
     "taylorquinn.dev"),
    (["New York, NY · (555) 010-4477 · taylor@example.com · linkedin.com/in/taylor-quinn-data",
      "taylorquinn.github.io/portfolio"], "taylorquinn.github.io/portfolio"),
    (["Austin, TX · (555) 010-4477 · taylorquinn.dev", "taylor@example.com"], "taylorquinn.dev"),
    (["taylor@example.com | Personal website: taylorquinn.dev"], "taylorquinn.dev"),
    (["taylor@example.com | taylorquinn.dev", "github.com/taylorquinn"], "github.com/taylorquinn"),
    # ... never a name with a dot that is no site, nor one after another label or among words
    (["Austin, TX | taylor@example.com | Python, Node.js, ASP.NET"], ""), (["St.Louis, MO | taylor@example.com"], ""),
    (["Taylor Quinn, M.Sc | taylor@example.com"], ""), (["taylor@example.com | Skype: taylor.quinn"], ""),
    (["taylor@example.com", "Maintainer of etl-kit.dev, used by 300 teams."], ""),
    # never words of a sentence, nor an item of a line that reads as one: a summary or a tagline under the contact
    # lines holds an employer's site or a project's repository, not the person's own
    (["taylor@example.com", "Maintainer of github.com/northwind/etl-kit, used by 300 teams."], ""),
    (["taylor@example.com | see https://taylorquinn.dev for my work"], ""),
    (["Austin, TX | taylor@example.com", "Data engineer at Northwind Labs, www.northwind.example, since 2022."], ""),
    (["Austin, TX | taylor@example.com",
      "Data engineer and creator of etl-kit — github.com/northwind/etl-kit — used by 300 teams."], ""),
    (["taylor@example.com", "Most proud of: github.com/northwind/etl-kit"], ""),
    (["Austin, TX | taylor@example.com | https://taylorquinn.dev",
      "Data engineer and creator of etl-kit — github.com/northwind/etl-kit — used by 300 teams."],
     "https://taylorquinn.dev"),
    # ... while a headline, a place or a line of links is no sentence, and a contact line never is one
    (["Senior Data Engineer | github.com/taylorquinn", "taylor@example.com | (555) 010-4477"],
     "github.com/taylorquinn"),
    (["Berlin | github.com/taylorquinn", "taylor@example.com"], "github.com/taylorquinn"),
    (["taylor@example.com | (555) 010-4477", "Portfolio: taylorquinn.dev | GitHub: github.com/taylorquinn"],
     "github.com/taylorquinn"),
    (["taylor@example.com | My GitHub page: github.com/taylorquinn"], "github.com/taylorquinn"),
    # a label naming a site is no words of a sentence, however long
    (["taylor@example.com", "My GitHub page: github.com/taylorquinn · https://taylorquinn.dev"],
     "github.com/taylorquinn"),
    (["taylor@example.com", "My personal website: taylorquinn.dev"], "taylorquinn.dev"),
])
def test_the_rules_reader_reads_the_website_from_the_header(header, website):
    profile = read(*header)
    assert profile.contact.website == website
    assert profile.name == "Taylor Quinn" and profile.contact.email == "taylor@example.com"


def test_the_website_is_read_only_from_the_header():
    profile = read("taylor@example.com", body=[*BODY, "• Published the loader at github.com/taylorquinn/loader."])
    assert profile.contact.website == ""


@pytest.mark.parametrize("text", [
    # the contact details in a Word table, whose text comes after the paragraphs': no line before the first heading
    ("SUMMARY\nData engineer with ten years of pipelines.\nEXPERIENCE\n"
     "Data Engineer · Northwind Labs · Remote   Mar 2022 – Present\n"
     "• Open-sourced the team's loader — github.com/northwind/loader\n• Built the pipelines.\n"
     "EDUCATION\nState University\nTaylor Quinn\nAustin, TX | taylor@example.com | github.com/taylorquinn"),
    # in the Word file's page header, which is not in its text
    ("EXPERIENCE\nData Engineer · Northwind Labs · Remote   Mar 2022 – Present\n"
     "• Published the loader: https://github.com/northwind/loader\n• Built the pipelines.\n"),
])
def test_no_website_is_read_from_the_body_when_the_text_starts_with_a_heading(text):
    assert parse_resume_text(text).contact.website == ""


def test_a_word_resume_with_its_contact_details_in_a_table_takes_no_bullets_address():
    """A common template: the name and contact details in a table at the top, which the text of a Word file has after
    the paragraphs. The first job's bullet names a project's repository: it is never the person's website."""
    document = Document()
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Taylor Quinn"
    table.cell(0, 1).text = "Austin, TX | taylor@example.com | github.com/taylorquinn"
    for line in ["SUMMARY", "Data engineer with ten years of pipelines.", "EXPERIENCE",
                 "Data Engineer · Northwind Labs · Remote\tMar 2022 – Present",
                 "• Open-sourced the team's loader — github.com/northwind/loader", "• Built the pipelines.",
                 "EDUCATION", "State University", "B.S. in Computer Science, 2016"]:
        document.add_paragraph(line)
    data = io.BytesIO()
    document.save(data)
    text = extract_text(data.getvalue(), "resume.docx")
    assert "github.com/northwind/loader" in text and text.index("SUMMARY") < text.index("taylor@example.com")
    assert parse_resume_text(text).contact.website == ""


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


@pytest.mark.parametrize("kind", ["pdf", "docx"])
@pytest.mark.parametrize("website", ["https://taylorquinn.dev", "www.taylorquinn.dev", "taylorquinn.github.io/cv/"])
def test_the_apps_own_files_read_back_with_a_portfolio(profile, kind, website):
    """The files print a portfolio without https:// or www. ('taylorquinn.dev'): a re-upload of one reads it back in
    that form, so the next resumes still print and link it."""
    resume = taylor(profile, CONTACT.model_copy(update={"website": website}))
    data = render_resume_pdf(resume, FitStyle())[0] if kind == "pdf" else resume_docx(resume, FitStyle())
    contact = parse_resume_text(extract_text(data, f"resume.{kind}")).contact
    printed = website.removeprefix("https://").removeprefix("www.").rstrip("/")
    assert (contact.email, contact.linkedin, contact.website, contact.location) == (
        "taylor@example.com", "linkedin.com/in/taylorquinn", printed, "Austin, TX")


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


TAILOR_PAGE = Path(__file__).parents[1] / "app/web/templates/tailor.html"
# Runs the Tailor page's whole script with stand-ins for the page's elements, the helpers of app.js and the browser's
# location and history, then does what each case asks and prints what the page shows.
TAILOR_HARNESS = r"""
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const code = page.slice(page.indexOf('<script>') + '<script>'.length, page.lastIndexOf('</script>'));
const HIDDEN = ['#result', '#toggles', '#timing', '#certs', '#steps'];  // hidden when the page opens

class Classes {
  constructor(names) { this.names = new Set(names); }
  add(name) { this.names.add(name); }
  remove(name) { this.names.delete(name); }
  toggle(name, on = !this.names.has(name)) { if (on) this.names.add(name); else this.names.delete(name); return on; }
  contains(name) { return this.names.has(name); }
}
function node(names = []) {
  const parts = {};
  return {
    nodeType: 1, classList: new Classes(names), children: [], attributes: {}, style: {}, textContent: '', value: '',
    src: '', disabled: false, readOnly: false,
    set className(value) { this.classList = new Classes(String(value).split(' ').filter(Boolean)); },
    part(selector) { return parts[selector] || (parts[selector] = node()); },
    replaceChildren(...nodes) {
      this.children = nodes.flat().filter(n => n !== null && n !== undefined && n !== false);
    },
    append(...nodes) { this.children.push(...nodes); },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name]; },
    addEventListener() {}, focus() {}, scrollIntoView() {},
    get firstChild() { return this.children[0]; },
  };
}
function el(tag, attrs = {}, ...children) {
  const made = node();
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value === null || value === undefined || key.startsWith('on')) continue;
    if (key === 'class') made.className = value;
    else if (key === 'text') made.textContent = value;
    else made.setAttribute(key, value === true ? '' : value);
  }
  made.replaceChildren(...children);
  return made;
}

async function show(c) {
  const elements = {};
  const $ = (selector, root) => root ? root.part(selector)
    : elements[selector] || (elements[selector] = node(HIDDEN.includes(selector) ? ['hidden'] : []));
  const urls = [];
  const history = { state: c.state, replaceState(state, _title, url) { this.state = state; urls.push(url); } };
  const api = async path => {
    if (path === '/api/resumes') return { resumes: [] };
    if (c.runs[path] === 'offline') throw Object.assign(new Error('Lost contact.'), { transient: true });
    if (c.runs[path] === null || !(path in c.runs)) throw new Error('Run not found');
    return c.runs[path];
  };
  const exposed = '\nreturn { showResult, clearResult, '
    + 'set current(value) { current = value; }, get current() { return current; } };';
  const tailor = new Function('$', '$$', 'el', 'api', 'finished', 'notice', 'chip', 'money', 'ME', 'recallValue',
    'rememberValue', 'location', 'history', 'window', 'document', code + exposed)(
    $, () => [], el, api, async data => data, (kind, text) => el('div', { class: `notice ${kind}`, text }),
    text => el('span', { text }), String, { id: '1', admin: false }, () => null, () => {}, { search: c.search },
    history, { matchMedia: () => ({ matches: false }), addEventListener() {} }, { addEventListener() {} });
  await new Promise(resolve => setTimeout(resolve, 50));  // the page's own init() has finished
  if (c.shown) {  // a run the page made: what create() keeps of it, then the result on screen
    tailor.current = c.shown;
    tailor.showResult(c.shown.run, 'resume');
  }
  const seen = (state = history.state) => ({
    urls: [...urls], state, result: !$('#result').classList.contains('hidden'),
    empty: !$('#empty').classList.contains('hidden'), preview: $('#pdf').src.split('?')[0],
    downloads: $('#downloads').children.map(link => link.attributes.href || link.attributes.id),
    current: tailor.current && { run: tailor.current.run.run_id, jd: tailor.current.jd,
      profileId: tailor.current.profileId, spelling: tailor.current.spelling, company: tailor.current.company },
    eu: $('#spell-eu').classList.contains('on'), messages: $('#messages').children.map(n => n.textContent) });
  const before = seen();
  tailor.clearResult();
  return { before, cleared: seen() };
}
(async () => {
  const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const out = {};
  for (const [name, c] of Object.entries(cases)) out[name] = await show(c);
  console.log(JSON.stringify(out));
})().catch(err => { console.error(err); process.exit(1); });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="runs the Tailor page's own script with Node.js")
def test_the_tailor_page_shows_its_result_again_after_a_link_in_the_preview(tmp_path):
    """The preview's links are clickable, and a browser's PDF viewer opens one in the page's own tab (Chrome does):
    the page then comes back without its result (Back loads it afresh). The page's address names the run it shows
    ('/tailor?run=7'), with what the page needs to tell whether the job's boxes still hold the same job, so the run
    and its downloads are shown again after Back or a reload. A run that is gone or failed drops the address; no
    result on screen, no run in the address."""
    def run(run_id, letter=True, spelling="us"):
        return {"run_id": run_id, "status": "ok", "pages": 1, "fits": True, "coverage": {"score": 0.9},
                "cover_letter": {"paragraphs": ["I build data pipelines."]} if letter else None, "spelling": spelling,
                "certifications": []}

    made = {"run": run(7, letter=False), "jd": "Senior Data Engineer at Fabrikam ...", "profileId": 3,
            "spelling": "us", "company": "Fabrikam"}
    job = {"jd": made["jd"], "profileId": 3, "spelling": "eu", "company": "Fabrikam"}  # what run 7 was made from
    cases = {
        "made": {"search": "", "state": None, "runs": {}, "shown": made},
        # Back to the resume shown (a link clicked in it), though the run has a letter too
        "back": {"search": "?run=7", "state": {"run": 7, "view": "resume", **job},
                 "runs": {"/api/runs/7": run(7, spelling="eu")}},
        "back to the letter": {"search": "?run=7", "state": {"run": 7, "view": "cover", **job},
                               "runs": {"/api/runs/7": run(7, spelling="eu")}},
        "address only": {"search": "?run=7", "state": None, "runs": {"/api/runs/7": run(7)}},
        "gone": {"search": "?run=8", "state": None, "runs": {"/api/runs/8": None}},
        "failed": {"search": "?run=9", "state": None, "runs": {"/api/runs/9": {**run(9), "status": "error"}}},
        "no run": {"search": "?run=x", "state": None, "runs": {}},
        "offline": {"search": "?run=7", "state": None, "runs": {"/api/runs/7": "offline"}},
    }
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "tailor.js").write_text(TAILOR_HARNESS, encoding="utf-8")
    done = subprocess.run(["node", str(tmp_path / "tailor.js"), str(TAILOR_PAGE), str(tmp_path / "cases.json")],
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    resume_only = ["/api/runs/7/resume.pdf?download=1", "add-cover"]
    both = ["/api/runs/7/resume.pdf?download=1", "/api/runs/7/cover.pdf?download=1"]

    made_job = {key: made[key] for key in ("jd", "profileId", "spelling", "company")}
    assert seen["made"]["before"] == {
        "urls": ["/tailor?run=7"], "state": {"run": 7, "view": "resume", **made_job}, "result": True, "empty": False,
        "preview": "/api/runs/7/resume.pdf", "downloads": resume_only, "current": {"run": 7, **made_job}, "eu": False,
        "messages": []}
    assert seen["made"]["cleared"]["urls"] == ["/tailor?run=7", "/tailor"] and seen["made"]["cleared"]["state"] is None
    assert not seen["made"]["cleared"]["result"] and seen["made"]["cleared"]["empty"]

    # Back: the run as it was, with the job it was made from, in its style and on the preview that was shown
    for name, view in (("back", "resume"), ("back to the letter", "cover")):
        back = seen[name]["before"]
        assert back["result"] and not back["empty"] and back["preview"] == f"/api/runs/7/{view}.pdf", name
        assert back["downloads"] == both and back["current"] == {"run": 7, **job} and back["eu"], name
        assert back["urls"] == ["/tailor?run=7"] and back["state"] == {"run": 7, "view": view, **job}, name
    # the address alone (typed, or kept by another browser): shown, but never taken for the job now in the boxes
    alone = seen["address only"]["before"]
    assert alone["result"] and alone["downloads"] == both and alone["preview"] == "/api/runs/7/cover.pdf"
    assert alone["current"] == {"run": 7, "jd": None, "profileId": None, "spelling": "us", "company": None}
    # no run to show: the address drops it, unless the server could not be reached (a reload tries again)
    for name in ("gone", "failed", "no run", "offline"):
        shown = seen[name]["before"]
        assert not shown["result"] and shown["empty"] and shown["current"] is None, name
        assert shown["messages"] == [] and shown["urls"] == ([] if name in ("no run", "offline") else ["/tailor"]), name


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
