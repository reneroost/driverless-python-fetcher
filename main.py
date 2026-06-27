"""
nodriver sidecar — FastAPI service for CDP-native undetected browser fetching.

VERSION: 1.1.0  (switch to zendriver — fixes ProxyForwarder NameError)

WHY NODRIVER OVER PATCHRIGHT
-----------------------------
patchright patches the CDP Runtime.enable leak and clears --enable-automation,
but it still runs within the Playwright framework's abstraction layer, which
introduces its own detection surface (init-script timing, CDP ordering, browser
context fingerprint). Crucially, patchright's evasion requires active patching
whose presence can itself be probed via Function.prototype.toString().

nodriver communicates directly with Chrome over CDP without any intermediary
framework. It has no Playwright layer, no init-script injection, and no
Runtime.enable broadcast — Chrome manages its own execution contexts natively.
The result is a browser that looks and behaves identically to real Chrome,
because the only thing between Java and the wire is a thin asyncio CDP client.

LIBRARY: ZENDRIVER (nodriver fork)
------------------------------------
This sidecar uses zendriver, not the upstream nodriver package.

The architecture is correct: browser.create_context(proxy_server=...) is
nodriver's documented, official API for per-context proxy isolation. It passes
proxy_server via CDP target.create_browser_context(), which Chrome handles
natively per context — each context gets its own cookie jar, network stack, and
proxy routing. This is not a Playwright API blend; it is in nodriver's own
quickstart and browser class docs.

However, nodriver's ProxyForwarder class (which handles authenticated HTTP
proxies) contains an unfixed NameError: inside util.py, ProxyForwarder.__init__
calls util.free_port() when it should call free_port() — the class was moved
into util.py during a refactor but the self-referential module prefix was never
removed. The upstream maintainer acknowledged the bug but never merged the fix,
and nodriver's PR queue has been frozen for months.

zendriver is the community-maintained fork created specifically to compile these
unmerged bugfixes. Its API is identical to nodriver (same create_context,
Browser.create, Tab.get, cdp module paths). Switching is a one-line import
change. The ProxyForwarder NameError is fixed in zendriver.

Install: pip install zendriver

ARCHITECTURE
------------
- One shared Chrome process, launched at startup via zendriver.
- asyncio.Semaphore(MAX_CONCURRENT_FETCHES) limits concurrency.
- Per-request: new browser context (with isolated proxy) → navigate → extract
  → close context. No cross-request cookie or state leakage.
- Java orchestrates: sends URL, proxy URL, locale, timezone, referrer hint, and
  navigation timing. The sidecar is responsible only for browser context
  creation, navigation, dwell, and HTML extraction.

PROXY HANDLING
--------------
browser.create_context(proxy_server=...) is the correct API for per-context
proxy isolation. For authenticated HTTP proxies (DataImpulse format), zendriver's
ProxyForwarder starts a local forwarder that Chrome can reach without embedding
credentials in --proxy-server (which Chrome ignores for HTTP proxies with auth).
Each create_context() call with a distinct proxy URL gets its own isolated
forwarding connection and cookie jar.

PROCESS CLEANUP
---------------
Chrome's parent process is stubborn to terminate. On shutdown, the sidecar:
  1. Closes all open contexts via browser.stop().
  2. Sends SIGTERM to browser._process.
  3. Waits up to 3 seconds; if still alive, sends SIGKILL.
  4. Reaps the process with wait() to avoid zombie processes.
This four-step sequence is required because browser.stop() alone does not
guarantee the OS process exits in all environments.

HEADLESS / HEADED MODE
-----------------------
Set NODRIVER_HEADLESS=0 and wrap ExecStart with xvfb-run for headed mode
(higher stealth; requires xorg-x11-server-Xvfb on AlmaLinux).

ENDPOINTS
---------
  POST /api/v1/fetch            — fetch a URL
  GET  /api/v1/browser-version  — {"version": "148.0.7778.96", "major": "148"}
  GET  /health                  — liveness probe
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import signal
from contextlib import asynccontextmanager
from enum import Enum
from urllib.parse import urlparse

import zendriver as nodriver
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_CONCURRENT_FETCHES  = 1
DEFAULT_TIMEOUT_SECONDS = 30

# How long to wait for Chrome to terminate gracefully before SIGKILL.
_SHUTDOWN_GRACE_SECONDS = 3

# ── ReferrerHint ──────────────────────────────────────────────────────────────


class ReferrerHint(str, Enum):
    """
    Mirror of the Java ReferrerHint enum.
    Values must stay in sync with ReferrerHint.sidecarValue() in Java.
    """
    GOOGLE_SEARCH = "google_search"
    GOOGLE_NEWS   = "google_news"
    TWITTER       = "twitter"
    DIRECT        = "direct"


_REFERRER_URL: dict[ReferrerHint, str] = {
    ReferrerHint.GOOGLE_SEARCH: "https://www.google.com/",
    ReferrerHint.GOOGLE_NEWS:   "https://news.google.com/",
    ReferrerHint.TWITTER:       "https://t.co/",
    ReferrerHint.DIRECT:        "",
}

# ── DTOs ──────────────────────────────────────────────────────────────────────


class NoDriverFetchRequest(BaseModel):
    """
    Wire-format request from Java NoDriverSidecarFetcher.

    Java sends only the signals that vary per attempt. Everything else is
    handled natively by Chrome (UA via --user-agent arg, locale via lang,
    Sec-Fetch-* via Chrome's own navigation logic).
    """
    url:                 str
    proxy_url:           str | None = Field(default=None)
    locale_tag:          str        = Field(default="en-US")
    timezone_id:         str        = Field(default="America/New_York")
    accept_language:     str        = Field(default="en-US,en;q=0.9")
    referrer_hint:       str        = Field(default="direct")
    default_wait_millis: int        = Field(default=1500, ge=0)
    timeout_seconds:     int        = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=120)
    wait_for_selector:   str | None = Field(default=None)


class NoDriverFetchResponse(BaseModel):
    html:        str | None = None
    final_url:   str | None = None
    status_code: int


# ── Application state ─────────────────────────────────────────────────────────

_browser:          nodriver.Browser | None = None
_fetch_semaphore:  asyncio.Semaphore | None = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Manages the singleton Chrome process for the lifetime of the sidecar.

    nodriver.Browser.create() is the correct async entrypoint (nodriver.start()
    is a convenience wrapper that calls loop().run_until_complete, which is
    incompatible with an already-running asyncio event loop inside uvicorn).
    """
    global _browser, _fetch_semaphore

    _fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    headless: bool    = os.environ.get("NODRIVER_HEADLESS", "1").strip() != "0"
    executable: str   = os.environ.get("NODRIVER_CHROME_EXECUTABLE", "").strip()

    browser_args: list[str] = [
        # Required on VPS: prevents Chrome crashing when /dev/shm is too small.
        "--disable-dev-shm-usage",
        # Skip first-run dialogs — no-op in headless; eliminates startup delay
        # in headed mode.
        "--no-first-run",
        "--no-default-browser-check",
        # Suppress audio subsystem initialisation — irrelevant for scraping.
        "--mute-audio",
    ]

    if not headless:
        browser_args += [
            "--window-size=1920,1080",
            "--window-position=0,0",
        ]

    # --no-sandbox is required when running as root (uid 0).
    # The service file specifies a non-root user; this guard is for container
    # environments that still map to uid 0.
    if os.getuid() == 0:
        browser_args.append("--no-sandbox")
        log.warning(
            "Running as root — appending --no-sandbox. "
            "Prefer a non-root User= in the systemd service file."
        )

    config_kwargs: dict = {
        "headless":     headless,
        "browser_args": browser_args,
    }
    if executable:
        config_kwargs["browser_executable_path"] = executable
        log.info("Using Chrome executable: %s", executable)
    else:
        log.warning(
            "NODRIVER_CHROME_EXECUTABLE not set — falling back to zendriver's "
            "auto-detected Chrome. Prefer NODRIVER_CHROME_EXECUTABLE="
            "/usr/bin/google-chrome for production."
        )

    config = nodriver.Config(**config_kwargs)

    _browser = await nodriver.Browser.create(config)
    log.info(
        "nodriver / Chrome launched. headless=%s | executable=%s",
        headless,
        executable or "auto-detected",
    )

    yield

    # ── Shutdown sequence ────────────────────────────────────────────────────
    #
    # nodriver's browser.stop() closes the CDP connection and attempts to
    # terminate Chrome, but the OS process can outlive the call. A four-step
    # sequence guarantees the process is reaped:
    #   1. stop() — graceful CDP disconnect + Chrome SIGTERM via nodriver.
    #   2. SIGTERM  — belt-and-suspenders OS-level signal if still alive.
    #   3. wait (3 s) — grace period for clean exit.
    #   4. SIGKILL  — forceful termination if still running after grace period.
    #   5. wait()   — reap the zombie to free the PID table entry.

    log.info("Shutting down nodriver sidecar...")

    process: asyncio.subprocess.Process | None = getattr(_browser, "_process", None)

    try:
        await _browser.stop()
    except Exception as exc:
        log.warning("browser.stop() raised during shutdown: %s", exc)

    if process is not None and process.returncode is None:
        try:
            process.send_signal(signal.SIGTERM)
            await asyncio.wait_for(process.wait(), timeout=_SHUTDOWN_GRACE_SECONDS)
            log.info("Chrome process terminated cleanly after SIGTERM.")
        except asyncio.TimeoutError:
            log.warning(
                "Chrome process did not exit within %ds after SIGTERM — sending SIGKILL.",
                _SHUTDOWN_GRACE_SECONDS,
            )
            try:
                process.send_signal(signal.SIGKILL)
                await process.wait()
                log.info("Chrome process killed with SIGKILL.")
            except Exception as exc:
                log.error("Failed to SIGKILL Chrome process: %s", exc)
        except Exception as exc:
            log.error("Unexpected error during Chrome shutdown: %s", exc)

    _browser = None
    log.info("nodriver sidecar shutdown complete.")


app = FastAPI(title="nodriver-sidecar", version="1.1.0", lifespan=lifespan)


# ── Core fetch logic ──────────────────────────────────────────────────────────

def _resolve_referrer_hint(raw: str) -> ReferrerHint:
    try:
        return ReferrerHint(raw)
    except ValueError:
        log.warning("Unknown referrer_hint %r — defaulting to DIRECT.", raw)
        return ReferrerHint.DIRECT


async def _dwell(tab: nodriver.Tab, base_wait_ms: int) -> None:
    """
    Scripted dwell that produces a minimal but non-mechanical behavioural signal.

    Mirrors the patchright sidecar's dwell implementation: two-phase with
    independent jitter so the timing histogram does not cluster into a
    recognisable machine pattern.
    """
    try:
        jitter      = random.randint(0, base_wait_ms // 3)
        total_wait  = base_wait_ms + jitter
        split_ratio = random.uniform(0.45, 0.65)
        pre_scroll  = int(total_wait * split_ratio)
        post_scroll = total_wait - pre_scroll

        await asyncio.sleep(pre_scroll / 1000)

        scroll_px = random.randint(100, 350)
        await tab.evaluate(f"window.scrollBy(0, {scroll_px})")

        await asyncio.sleep(post_scroll / 1000)
    except Exception as exc:
        log.debug("Dwell interrupted — proceeding with content extraction: %s", exc)


async def _execute_fetch(
    browser: nodriver.Browser,
    req:     NoDriverFetchRequest,
) -> NoDriverFetchResponse:
    """
    Creates a per-request isolated browser context, navigates to the target
    URL, applies dwell or selector wait, extracts HTML, and closes the context.

    Each context has its own cookie jar and proxy configuration. No state
    leaks between fetch attempts.

    PROXY NOTE: create_context() with proxy_server starts nodriver's internal
    ProxyForwarder for authenticated HTTP proxies. Chrome's --proxy-server arg
    does not support embedded credentials; the forwarder intercepts the CONNECT
    tunnel and injects auth. This is nodriver's documented mechanism and
    requires no Chrome extension.
    """
    hint        = _resolve_referrer_hint(req.referrer_hint)
    referrer    = _REFERRER_URL[hint]
    nav_timeout = req.timeout_seconds * 1000  # nodriver accepts milliseconds

    create_context_kwargs: dict = {
        "new_window": False,
    }
    if req.proxy_url:
        create_context_kwargs["proxy_server"] = req.proxy_url

    tab = await browser.create_context(**create_context_kwargs)
    try:
        # Set Accept-Language via CDP so it is transmitted on every request
        # from this context without overriding Chrome's native header ordering.
        await tab.send(
            nodriver.cdp.network.set_extra_http_headers(
                headers=nodriver.cdp.network.Headers(
                    {"Accept-Language": req.accept_language}
                )
            )
        )

        # Navigate with referrer when the hint is not DIRECT. Passing the
        # referrer here causes Chrome to set both the Referer header and
        # Sec-Fetch-Site natively and correctly, without any manual injection.
        navigate_kwargs: dict = {"timeout": nav_timeout}
        if referrer:
            navigate_kwargs["referrer"] = referrer

        response = await tab.get(req.url, **navigate_kwargs)

        # tab.get() returns the Tab itself (navigation is implicit in nodriver).
        # The HTTP status is available on the navigation response event, but
        # nodriver does not expose it directly from get(). Use the CDP response
        # from the network event instead.
        status_code = await _resolve_status_code(tab, req.url)
        final_url   = tab.url

        log.info("navigation complete  status=%d  url=%s", status_code, req.url)

        if req.wait_for_selector:
            try:
                await asyncio.wait_for(
                    tab.select(req.wait_for_selector),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, Exception):
                log.debug(
                    "wait_for_selector '%s' timed out for %s — proceeding with content",
                    req.wait_for_selector,
                    req.url,
                )
        elif req.default_wait_millis > 0:
            await _dwell(tab, req.default_wait_millis)

        content: str = await tab.get_content()
        if not content or not content.strip():
            log.error("get_content() returned empty DOM for %s", req.url)
            return NoDriverFetchResponse(
                html=None, final_url=final_url, status_code=status_code
            )

        return NoDriverFetchResponse(
            html=content, final_url=final_url, status_code=status_code
        )

    finally:
        # close() disposes the browser context and its cookie jar. Raising an
        # exception in this block must not suppress the original exception, so
        # the failure is logged and swallowed.
        try:
            await tab.close()
        except Exception as exc:
            log.warning("Error closing tab/context after fetch: %s", exc)


async def _resolve_status_code(tab: nodriver.Tab, url: str) -> int:
    """
    Attempts to resolve the HTTP status code of the main frame navigation.

    nodriver does not expose the navigation response object from tab.get().
    This helper fires a CDP Runtime.evaluate to read
    window.performance.getEntriesByType('navigation')[0].responseStatus, which
    Chrome populates for the main-frame request. Falls back to 200 on any
    error so a missing status code never blocks extraction.
    """
    try:
        result = await tab.evaluate(
            "(() => { const e = window.performance.getEntriesByType('navigation')[0]; "
            "return e ? e.responseStatus : null; })()"
        )
        if isinstance(result, (int, float)) and result > 0:
            return int(result)
    except Exception as exc:
        log.debug("Status code resolution via performance API failed for %s: %s", url, exc)
    return 200


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/fetch", response_model=NoDriverFetchResponse)
async def execute_fetch(req: NoDriverFetchRequest) -> NoDriverFetchResponse:
    """
    Fetches a URL using a nodriver Chrome context.

    The asyncio semaphore limits execution to MAX_CONCURRENT_FETCHES (1),
    mirroring the Java NoDriverSidecarFetcher's Semaphore(1). Returns
    status_code=502 in the body on execution failures so the Java retry /
    circuit-breaker path fires rather than the HTTP error path.
    """
    if _browser is None:
        raise HTTPException(status_code=503, detail="Browser not available")

    async with _fetch_semaphore:
        try:
            return await _execute_fetch(_browser, req)
        except Exception as exc:
            log.exception("nodriver fetch failed for %s: %s", req.url, exc)
            return NoDriverFetchResponse(html=None, final_url=None, status_code=502)


@app.get("/api/v1/browser-version")
async def get_browser_version() -> dict[str, str]:
    """
    Returns the version of the running Chrome binary.

    Called at Java startup by NoDriverSidecarValidator to confirm the sidecar
    is live and retrieve the Chromium major version for UA pool selection.
    """
    if _browser is None:
        raise HTTPException(status_code=503, detail="Browser not available")
    try:
        version_info = await _browser.connection.send(
            nodriver.cdp.browser.get_version()
        )
        # version_info is a tuple: (protocolVersion, product, revision, userAgent, jsVersion)
        # product looks like "Chrome/148.0.7778.96"
        product: str = version_info[1] if len(version_info) > 1 else ""
        version: str = product.split("/")[1] if "/" in product else product
        major:   str = version.split(".")[0] if "." in version else version
        return {"version": version, "major": major}
    except Exception as exc:
        log.warning("Could not retrieve browser version: %s", exc)
        raise HTTPException(status_code=503, detail="Browser version unavailable") from exc


@app.get("/health")
async def health() -> dict[str, object]:
    """
    Liveness probe for systemd and the Java NoDriverSidecarValidator.

    Returns {"status": "ok"} when the browser is present and its CDP
    connection is alive. Returns {"status": "degraded"} otherwise.
    """
    connected: bool = _browser is not None and _browser.connection is not None
    return {"status": "ok" if connected else "degraded", "browser_connected": connected}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("NODRIVER_HOST", "127.0.0.1")
    port = int(os.getenv("NODRIVER_PORT", "8083"))
    uvicorn.run(app, host=host, port=port, log_level="info")
