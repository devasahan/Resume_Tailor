Inter 4.000 (Regular, Medium, SemiBold and Bold) is bundled so the PDFs look the same on every computer.

Source: the Inter 4.000 OpenType files of the Debian package fonts-inter 4.0+ds-1
(/usr/share/fonts/opentype/inter), made by the Inter Project Authors: https://github.com/rsms/inter

Why the files are converted: those OpenType files have CFF (PostScript) outlines, and fpdf2 2.9 embeds such a font
as /CIDFontType2 with /FontFile2, which the PDF specification keeps for TrueType outlines: some readers then show
nothing or the wrong letters. make_inter_ttf.py (in this folder) converts each file to TrueType (glyf) outlines with
fontTools: cubic curves become quadratic ones at most 1 font unit (of 2048 per em) away, and the glyphs, character
map, advance widths, kerning and names stay as they are. The script checks the character map and every advance width
against the source file. To make the files again:

    python app/render/fonts/make_inter_ttf.py --source /usr/share/fonts/opentype/inter --out app/render/fonts

License: SIL Open Font License 1.1, with no Reserved Font Name: see OFL-Inter.txt (Inter's copyright line and the full
license text). The converted files are a Modified Version under the OFL and stay under it.

SHA-256 of the bundled files:
7eb9642f3a0f91fe405c8bce8c6854d8d7b201581ccdb7c47e75b0b47eab7cae  Inter-Regular.ttf
17e82f885edf35f91ba5a99cad04953e826e703c379b90a3a09cc63978bb966e  Inter-Medium.ttf
90092e1e133cdf50d7195a5d2eeed43c9c7e2b25e945dd0fa60e130d1e10b0f5  Inter-SemiBold.ttf
4c7027ada6103b74f884ba64fe2fec38fa74b3219e5e20066e2bb75ee654a8b1  Inter-Bold.ttf
