# Aksharam

## Configuration

Tool paths (Tesseract, Poppler) and input/output filenames are stored in a local
configuration module that is **not** checked in to source control.

1. Copy `config_local.py` (provided as a template) to the workspace root.
2. Edit the constants to point at your installed tools and any custom filenames.
3. `config_local.py` is listed in `.gitignore` so it won't be committed.

The scripts will automatically fall back to sensible defaults if the config is
missing.
