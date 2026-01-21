# -*- coding: utf-8 -*-
"""
sitecustomize.py

Auto-loaded by Python on startup (if present on sys.path).

This project relies on AkShare + requests/urllib3 to fetch market data.
In some environments, system/registry proxies (or env HTTP(S)_PROXY) can
break these calls (hang, 401/login, SSL issues). We default to bypassing
proxies to keep data fetching stable.
"""

from __future__ import annotations

import os


def _disable_proxy_env() -> None:
    # Clear common proxy env vars (case-insensitive).
    for k in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        if k in os.environ:
            os.environ.pop(k, None)

    # Force bypass for all hosts.
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _patch_requests_no_proxy() -> None:
    # requests reads system proxy (Windows registry) via urllib when trust_env=True.
    # Patch Session.__init__ to default to trust_env=False for all future sessions.
    try:
        import requests.sessions  # type: ignore

        Session = requests.sessions.Session  # type: ignore[attr-defined]
        if getattr(Session, "_laowang_no_proxy_patched", False):
            return

        old_init = Session.__init__

        def new_init(self, *args, **kwargs):  # noqa: ANN001
            old_init(self, *args, **kwargs)
            try:
                self.trust_env = False
            except Exception:
                pass

        Session.__init__ = new_init  # type: ignore[assignment]
        Session._laowang_no_proxy_patched = True  # type: ignore[attr-defined]
    except Exception:
        return


_disable_proxy_env()
_patch_requests_no_proxy()

