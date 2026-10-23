# Presentation authoring sources

These scripts and original input assets were moved out of the temporary build directory during repository cleanup. Final decks remain in [output/presentation](../../output/presentation/).

| Source | Purpose |
|---|---|
| `build_deck.mjs` | Build the 31-slide editable research presentation |
| `finish_deck.mjs`, `finalize_ready.mjs` | Finalize and render the static deck |
| `embed_smooth_video.ps1` | Embed the smooth steering MP4 on slide 19 through PowerPoint |
| `check_embedded_video.py`, `preserve_deck_and_embed.py` | Check the embedded MP4 and preserve all other original deck parts |
| `finalize_video_deck.mjs` | Validate and write the final deck with video |
| `inspect_final_video_deck.ps1`, `review_video_deck.py` | Inspect the saved deck and slide renders |
| `extract_final_script_sources.py` | Read slide text and speaker notes |
| `assets/` | Original cover, agenda artwork, prototype screenshot, and extracted research tables |

The Node authoring scripts require `@oai/artifact-tool` and the Codex presentation skill. `artifact_runtime.mjs` resolves a local SDK installation or the bundled runtime selected by `RUNTIME_NODE_MODULES`. Set `AUTODRIVE_PRESENTATION_SKILL_DIR` for the static deck scripts when the installed skill path differs. Some scripts retain the original Windows project and reference-PDF paths, which must be adjusted on another machine. PowerPoint automation requires Windows with Microsoft PowerPoint installed.

Generated drafts and slide renders go under `tmp/`, which is disposable and excluded from Git. The application does not depend on this directory. Rebuilding the original static deck requires the original reference PDFs and recorded experiment outputs. The video embedding scripts require the static ready deck, the final MP4 and its first-frame poster.
