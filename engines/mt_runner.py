"""Run MangaTranslator's CLI inside its own venv with machine-readable progress.

MangaTranslator's CLI does not pass a progress callback to
`batch_translate_images`; this wrapper injects one that prints JSON lines
(`{"progress": 0.0-1.0, "message": "..."}`) prefixed with `@@CT@@` so the
parent process can parse them out of the regular log stream.

Usage: python mt_runner.py <MangaTranslator dir> -- <main.py arguments...>
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    mt_dir = sys.argv[1]
    if sys.argv[2] != "--":
        raise SystemExit("usage: mt_runner.py <dir> -- <args>")
    args = sys.argv[3:]
    os.chdir(mt_dir)
    sys.path.insert(0, mt_dir)

    import core.pipeline as mt_pipeline  # noqa: PLC0415  (MangaTranslator package)
    import main as mt_main  # noqa: PLC0415  (MangaTranslator's main.py)
    import core.services.translation as mt_translation  # noqa: PLC0415
    from utils.exceptions import TranslationError  # noqa: PLC0415

    # The proxy owns retries. MangaTranslator otherwise retries each 429 or
    # connection failure five more times, multiplying the proxy's six attempts.
    # Its HTTP timeout must outlive six full 600s upstream requests plus 135s
    # backoff; a final proxy error must fail the page rather than render as text.
    original_endpoint = mt_translation.call_openai_compatible_endpoint

    def proxy_endpoint(*call_args, **call_kwargs):
        call_kwargs["max_retries"] = 0
        call_kwargs["timeout"] = 3780
        try:
            return original_endpoint(*call_args, **call_kwargs)
        except TranslationError as exc:
            raise TranslationError(f"API failed: {exc}") from exc

    mt_translation.call_openai_compatible_endpoint = proxy_endpoint

    original = mt_pipeline.batch_translate_images

    def emit(fraction: float, message: str) -> None:
        print("@@CT@@" + json.dumps({"progress": fraction, "message": message}, ensure_ascii=False), flush=True)

    def with_progress(*call_args, **call_kwargs):
        call_kwargs["progress_callback"] = emit
        return original(*call_args, **call_kwargs)

    # main() imports the function from core.pipeline at call time.
    mt_pipeline.batch_translate_images = with_progress

    # v1.24.7 CLI bug: clamp_settings() walks _CONFIG_ATTR_PATHS and indexes
    # SETTING_CONSTRAINTS for each key, but the boolean
    # `outside_text_osb_auto_vertical_text` has no constraint -> KeyError on
    # every CLI run. Non-numeric keys have nothing to clamp; drop them.
    import core.validation as mt_validation  # noqa: PLC0415

    for key in list(mt_validation._CONFIG_ATTR_PATHS):
        if key not in mt_validation.SETTING_CONSTRAINTS:
            del mt_validation._CONFIG_ATTR_PATHS[key]

    sys.argv = ["main.py", *args]
    mt_main.main()



if __name__ == "__main__":
    main()
