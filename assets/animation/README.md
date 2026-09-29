Install `python -m pip install -r requirements.txt` and provide Poppler (`pdftocairo`) and librsvg (`rsvg-convert`).
Run `python export_story_gif.py` in this directory to regenerate `../evidence-credit.gif` (54 seconds, 900 px, 10 fps).
The renderer uses locally installed Comic Sans MS; set `COMIC_FONT` and `COMIC_BOLD_FONT` to your font paths if needed. No font files are redistributed.
