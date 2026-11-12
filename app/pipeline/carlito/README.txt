Carlito's advance widths (Regular, Bold and Italic), for measuring the resume layout from before the restyle.

What it is: widths.json holds, for each of the three styles, the width of every character Carlito has, in thousandths
of an em, rounded as fpdf2 2.9 rounds them, and the width fpdf2 gives a character the font does not have. Numbers
only: no glyphs, outlines or anything else of the font.

Why it is here: until the restyle the resume was set in Carlito (metrically compatible with Calibri). The bullet plan
and the page count of a fitted resume are still worked out on that layout (app/pipeline/old_layout.py), so the restyle
changes only the look: the model is asked for as many bullets as before, and no resume takes more pages than it did.
The font files themselves are no longer shipped; these widths are all that layout needs.

How it was made: make_widths.py (in this folder) read the three font files the app shipped before the restyle
(app/render/fonts/Carlito-Regular.ttf, -Bold.ttf and -Italic.ttf at commit aa88a1a, taken from git history) and
checked every width against fpdf2's own get_string_width before writing the file:

    python app/pipeline/carlito/make_widths.py --regular Carlito-Regular.ttf --bold Carlito-Bold.ttf \
        --italic Carlito-Italic.ttf --out app/pipeline/carlito/widths.json

SHA-256 of the font files read:
f6418f708baede9789daef5d458c0f53d2a888af9820e8062934e504fedc6595  Carlito-Regular.ttf
bb5d20f79b82599ec72983597437373a80f2d2085fa91fc144fd74e876a594db  Carlito-Bold.ttf
0b019225e58d702bfedcbd35c21696769f8ee115cb6343f84c2f240312450d1c  Carlito-Italic.ttf

License: Carlito is Copyright (c) 2010-2013 by tyPoland Lukasz Dziedzic with Reserved Font Name "Carlito", licensed
under the SIL Open Font License 1.1: see OFL-Carlito.txt (the full license text, as the app shipped it with the font).
Source: https://github.com/googlefonts/carlito
