"""
zendriver sidecar — FastAPI service for CDP-native Chrome fetching.

VERSION: 2.0.0

WHY THIS EXISTS
---------------
news-scraper's Chrome engine, beside the camoufox (Firefox) sidecar, and one of
the two browser fallbacks for curl_cffi. zendriver (the maintained fork of
nodriver) drives a real Google Chrome directly over CDP: no Playwright layer,
no init scripts, no patched getters, and no Runtime.enable broadcast — Chrome
manages its own execution contexts.

AS CURL_CFFI'S FALLBACK
-----------------------
On 2026-10-11 every strategy prod routes to curl_cffi was run through it
(news-scraper's bench/: its own strategies via /trend/test and /content/test,
3 attempts, DataImpulse): 43 of 43 trend lists and 48 of 49 article pages,
against 43/43 and 49/49 for curl_cffi on the same pages. The miss is wsj.com
(DataDome). It also passed Sky News's Akamai on 7 of 10 single attempts, which
neither curl_cffi nor camoufox has. Costs: median 7 s a fetch against 2.5 s,
~0.9-1.0 MB of proxy traffic against ~77 KB, ~0.66 GB of memory (0.76 GB
peak). news-scraper's FALLBACK.md is the runbook for switching to it.

VERSIONS
--------
zendriver 0.17.1 (requirements.txt). The browser is the host's Google Chrome
(NODRIVER_CHROME_EXECUTABLE=/usr/bin/google-chrome), which updates with the
OS packages — a real, current Chrome is the point.

HEADED, NOT HEADLESS
--------------------
Run it headed on a virtual display (the systemd unit wraps it in xvfb-run).
Measured 2026-10-11, headless Chrome under zendriver presents a Chrome user
agent with NO Client Hints (no Sec-CH-UA headers, empty
navigator.userAgentData) and an 800x600 screen. zendriver strips "Headless"
from the UA with an override that carries no UA metadata, and real Chrome never
looks like that. Headed Chrome reports its own complete, consistent identity,
with nothing overridden. NODRIVER_HEADLESS=1 still works for quick local runs.

Checked on the VPS itself (2026-10-11; Chrome 154 under xwfb-run, no GPU, 2
cores): full Client Hints, a 1920x1080 screen, and 27 of 42 single-attempt
fetches of the hard sources passed (Sky News 5 of 6) — in line with the laptop.
WebGL is unavailable there; enabling SwiftShader (NODRIVER_EXTRA_ARGS=
--enable-unsafe-swiftshader) made no difference (14/21 vs 13/21), so Chrome's
default stands. NODRIVER_EXTRA_ARGS takes any extra Chrome flags, space-separated.

PER FETCH
---------
1. Timezone from the exit IP: one lookup at ip-api.com through the same proxy
   session the page will use. Production exits are random, so the region
   timezone Java sends would contradict the IP; it is used only without a
   proxy. If the lookup fails, the fetch fails and Java retries on a fresh exit.
2. A new browser context with its own proxy (the local forwarder below) and
   cookie jar.
3. Network events: the main document's real status, headers and body come from
   CDP, not from the page.
4. Page.navigate with the referrer, so Chrome itself sets Referer and a
   consistent Sec-Fetch-Site. Accept-Language is Chrome's own, consistent with
   navigator.languages.
5. A Cloudflare interstitial (403/503 with cf-mitigated: challenge, or a
   challenge marker in the page) is given time to solve itself and reload; the
   status returned is the reloaded page's.
6. A non-HTML document (a JSON API) is returned as served — Chrome wraps JSON
   in an HTML viewer page otherwise.
7. Dwell or selector wait.
8. The HTML, once the page has settled: some pages reload themselves right
   after loading (globalnews.ca), and read mid-reload the document stops
   inside <head>.
Images, media, fonts and ad/tracking requests are failed inside the browser
throughout, which halved the proxy traffic per fetch (1.6-1.8 MB to ~1 MB).
Everything spends from one deadline (timeout_seconds), because Java's read
timeout counts as the sidecar being down; a page that is not parsed or settled
by then fails the fetch, so Java retries on a fresh exit rather than receiving a
half-built page.

BROWSER LIFECYCLE
-----------------
One Chrome process, relaunched by the next fetch if it dies and recycled every
BROWSER_RECYCLE_AFTER_FETCHES fetches.

PROXY HANDLING
--------------
Chrome cannot use embedded credentials for an HTTP proxy. _AsyncProxyForwarder
starts a local CONNECT tunnel per fetch that injects Proxy-Authorization towards
DataImpulse, and the context uses it as its proxy. Stopped after the fetch.

ENDPOINTS
---------
  POST /api/v1/fetch            — fetch a URL
  GET  /api/v1/browser-version  — {"version": "155.0.8059.39", "major": "155"}
  GET  /health                  — liveness probe
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import signal
import urllib.request
from contextlib import asynccontextmanager
from enum import Enum
from urllib.parse import urlparse

import uvicorn
import zendriver as nodriver
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

# Recycle Chrome after this many fetches, so per-process growth stays bounded.
BROWSER_RECYCLE_AFTER_FETCHES = 50

# The proxy's exit IP and its timezone, asked through the proxy itself.
EXIT_GEO_URL             = "http://ip-api.com/json?fields=status,message,query,timezone"
EXIT_GEO_TIMEOUT_SECONDS = 10

# Cloudflare's managed challenge answers with HTTP 403 (or 503) and an
# interstitial a real browser solves on its own, then reloads into the page.
# Recognised by Cloudflare's own `cf-mitigated: challenge` header, else by a
# marker in the page: the inline `window._cf_chl_opt = {...}` of a managed
# challenge, or the Turnstile script a site's own wall loads. Only a 403/503 is
# checked, so a real page that embeds Turnstile is never mistaken for a wall.
CHALLENGE_STATUSES = frozenset({403, 503})
CHALLENGE_MARKERS  = ("_cf_chl_opt", "challenges.cloudflare.com/turnstile")
CHALLENGE_WAIT_MS  = 15_000
POLL_SECONDS       = 0.1

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
    Wire-format request: news-scraper's NativeBrowserFetchRequest, sent by
    BrowserSidecarFetcher.

    timezone_id is used only when there is no proxy_url (see PER FETCH);
    locale_tag and accept_language are accepted and ignored, because Chrome's
    own Accept-Language stays consistent with navigator.languages and a header
    override would not.
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

_state: dict[str, object] = {}
_fetch_semaphore: asyncio.Semaphore | None = None


# ── Browser lifecycle ─────────────────────────────────────────────────────────

def _browser_config() -> nodriver.Config:
    headless: bool  = os.environ.get("NODRIVER_HEADLESS", "1").strip() != "0"
    executable: str = os.environ.get("NODRIVER_CHROME_EXECUTABLE", "").strip()

    browser_args: list[str] = [
        # Required on VPS: prevents Chrome crashing when /dev/shm is too small.
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--mute-audio",
    ]
    if not headless:
        browser_args += ["--window-size=1920,1080", "--window-position=0,0"]
    # Extra Chrome flags for one deployment, space-separated (the systemd unit sets them).
    browser_args += os.environ.get("NODRIVER_EXTRA_ARGS", "").split()

    # --no-sandbox is required when running as root (uid 0). The service file
    # specifies a non-root user; this guard is for container environments.
    if os.getuid() == 0:
        browser_args.append("--no-sandbox")
        log.warning("Running as root — appending --no-sandbox. Prefer a non-root User=.")

    config_kwargs: dict = {"headless": headless, "browser_args": browser_args}
    if executable:
        config_kwargs["browser_executable_path"] = executable
    else:
        log.warning("NODRIVER_CHROME_EXECUTABLE not set — using zendriver's auto-detected Chrome.")
    return nodriver.Config(**config_kwargs)


async def _launch_browser() -> None:
    config  = _browser_config()
    browser = await nodriver.Browser.create(config)
    _state["browser"] = browser
    _state["fetches"] = 0
    log.info("zendriver / Chrome launched. headless=%s | executable=%s",
             config.headless, config.browser_executable_path)


async def _close_browser() -> None:
    """
    Stops Chrome and makes sure the OS process is gone: browser.stop() closes
    the CDP connection and terminates Chrome, but the process can outlive the
    call, so it is signalled and reaped here too.
    """
    browser = _state.pop("browser", None)
    if browser is None:
        return
    process = getattr(browser, "_process", None)
    try:
        await browser.stop()
    except Exception as exc:
        log.warning("browser.stop() raised: %s", exc)
    if process is not None and process.poll() is None:
        try:
            process.send_signal(signal.SIGTERM)
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=_SHUTDOWN_GRACE_SECONDS)
        except asyncio.TimeoutError:
            log.warning("Chrome did not exit within %ds after SIGTERM — sending SIGKILL.",
                        _SHUTDOWN_GRACE_SECONDS)
            process.kill()
            await asyncio.to_thread(process.wait)
        except Exception as exc:
            log.error("Unexpected error while stopping Chrome: %s", exc)


def _browser_alive(browser) -> bool:
    connection = getattr(browser, "connection", None)
    return (browser is not None and not browser.stopped
            and connection is not None and not connection.closed)


async def _ensure_browser():
    """
    The live browser, relaunched first if the previous one has gone or has
    served BROWSER_RECYCLE_AFTER_FETCHES fetches. Chrome can die under the
    sidecar (a crash, or the unit's MemoryMax killing it) while this process
    stays up; without this every later fetch would fail until a manual
    restart. Called under the fetch semaphore, so a relaunch never races a fetch.
    """
    browser = _state.get("browser")
    if _browser_alive(browser):
        if _state["fetches"] < BROWSER_RECYCLE_AFTER_FETCHES:
            _state["fetches"] += 1
            return browser
        log.info("Recycling the browser after %d fetches.", _state["fetches"])
    else:
        log.error("Browser is not connected — relaunching it.")
    await _close_browser()
    await _launch_browser()
    _state["fetches"] = 1
    return _state["browser"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _fetch_semaphore
    _fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    await _launch_browser()
    try:
        yield
    finally:
        log.info("Shutting down zendriver sidecar...")
        await _close_browser()
        log.info("zendriver sidecar shutdown complete.")


app = FastAPI(title="zendriver-sidecar", version="2.0.0", lifespan=lifespan)


# ── Core fetch logic ──────────────────────────────────────────────────────────

def _resolve_referrer_hint(raw: str) -> ReferrerHint:
    try:
        return ReferrerHint(raw)
    except ValueError:
        log.warning("Unknown referrer_hint %r — defaulting to DIRECT.", raw)
        return ReferrerHint.DIRECT


def _remaining(deadline: float) -> float:
    return deadline - asyncio.get_running_loop().time()


def _exit_timezone(proxy_url: str) -> str:
    """The proxy exit's timezone, asked through the proxy. Blocking: run in a thread."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy_url}))
    with opener.open(EXIT_GEO_URL, timeout=EXIT_GEO_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read())
    if data.get("status") != "success" or not data.get("timezone"):
        raise RuntimeError(f"exit geo lookup failed: {data.get('message') or data}")
    return data["timezone"]


async def _dwell(tab: nodriver.Tab, base_wait_ms: int) -> None:
    """
    Scripted dwell with a randomised split around one natural scroll, so the
    timing histogram does not cluster into a recognisable machine pattern.
    """
    try:
        jitter      = random.randint(0, base_wait_ms // 3)
        total_wait  = base_wait_ms + jitter
        split_ratio = random.uniform(0.45, 0.65)
        pre_scroll  = int(total_wait * split_ratio)
        post_scroll = total_wait - pre_scroll

        await asyncio.sleep(pre_scroll / 1000)
        await tab.evaluate(f"window.scrollBy(0, {random.randint(100, 350)})")
        await asyncio.sleep(post_scroll / 1000)
    except Exception as exc:
        log.debug("Dwell interrupted — proceeding with content extraction: %s", exc)


async def _wait_for_document(documents: list, count: int, deadline: float) -> None:
    """Waits until at least `count` main-frame document responses have arrived."""
    while len(documents) < count:
        if _remaining(deadline) <= 0:
            raise asyncio.TimeoutError("no document response before the deadline")
        await asyncio.sleep(POLL_SECONDS)


async def _wait_until_parsed(tab: nodriver.Tab, deadline: float, leaving: str) -> None:
    """
    Waits until the new document has replaced the page at `leaving` and been
    parsed. A document's response arrives before it commits, so readyState read
    too early is the previous page's — which is how a referred fetch once
    returned the blank referrer page instead of the target.
    """
    while _remaining(deadline) > 0:
        try:
            href, state = await tab.evaluate("[location.href, document.readyState]")
            if href != leaving and state in ("interactive", "complete"):
                return
        except Exception:
            pass
        await asyncio.sleep(POLL_SECONDS)
    raise asyncio.TimeoutError("page not parsed before the deadline")


async def _wait_until_settled(tab: nodriver.Tab, documents: list, deadline: float) -> None:
    """
    Waits until the page is parsed and no newer main-frame document has arrived
    meanwhile. Some pages reload themselves right after their first load
    (globalnews.ca does, setting a cookie): read during the reload, the HTML
    stopped inside <head> and the trend list came back empty.
    """
    while _remaining(deadline) > 0:
        seen = len(documents)
        try:
            state = await tab.evaluate("document.readyState")
        except Exception:
            state = None
        if state in ("interactive", "complete") and len(documents) == seen:
            return
        await asyncio.sleep(POLL_SECONDS)
    raise asyncio.TimeoutError("page did not settle before the deadline")


async def _is_challenge(tab: nodriver.Tab, document) -> bool:
    headers = {k.lower(): v for k, v in (document.response.headers or {}).items()}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True
    try:
        content = await tab.get_content()
    except Exception:
        return False
    return any(marker in content for marker in CHALLENGE_MARKERS)


async def _await_challenge(tab: nodriver.Tab, url: str, documents: list, deadline: float):
    """
    Waits for a Cloudflare interstitial to clear: the challenge script reloads
    the main frame, which shows up as a newer main-frame document response.
    Returns that document, or the challenge's own if it never clears, so Java
    treats the fetch as blocked and retries it on a fresh exit.
    """
    seen   = len(documents)
    budget = min(CHALLENGE_WAIT_MS / 1000, _remaining(deadline) - 1)
    log.info("Cloudflare interstitial on %s — waiting up to %d ms for it to clear", url, max(budget, 0) * 1000)
    give_up_at = asyncio.get_running_loop().time() + max(budget, 0)
    while asyncio.get_running_loop().time() < give_up_at:
        await asyncio.sleep(0.5)
        newer = documents[seen:]
        if newer and newer[-1].response.status < 400:
            # The reload usually keeps the URL, so wait for the challenge markup to go instead.
            while _remaining(deadline) > 0:
                try:
                    content = await tab.get_content()
                    if not any(marker in content for marker in CHALLENGE_MARKERS):
                        break
                except Exception:
                    pass
                await asyncio.sleep(POLL_SECONDS)
            log.info("Cloudflare interstitial cleared for %s  status=%d", url, newer[-1].response.status)
            return newer[-1]
    log.info("Cloudflare interstitial did not clear for %s", url)
    return documents[-1]


# Not worth the proxy traffic: no anti-bot sensor is delivered as an image, a
# video or a font, and the ad/tracking networks below are not sensors either
# ("analytics" alone is deliberately absent — vendor sensors may carry it).
BLOCKED_RESOURCE_TYPES = ("Image", "Media", "Font")
BLOCKED_URL_KEYWORDS = ("google-analytics", "doubleclick", "scorecardresearch", "taboola",
                        "outbrain", "adnxs", "facebook.net", "adsystem")


async def _intercept(tab: nodriver.Tab, referrer: str) -> None:
    """
    One Fetch-domain handler for the whole fetch: it answers the referrer URL
    with an empty page (see _navigate_from) and fails images, media, fonts and
    ad/tracking requests. The patterns limit what pauses; anything else that
    pauses is continued untouched. The handler stays registered while Fetch is
    enabled — zendriver re-enables, with no URL pattern, any domain that has a
    handler but was not enabled by hand, which would pause every request.
    """
    cdp = nodriver.cdp

    async def _on_paused(event: cdp.fetch.RequestPaused) -> None:
        url = event.request.url
        try:
            if referrer and url == referrer:
                await tab.send(cdp.fetch.fulfill_request(
                    request_id=event.request_id,
                    response_code=200,
                    response_headers=[cdp.fetch.HeaderEntry(name="Content-Type", value="text/html; charset=utf-8")],
                    body=base64.b64encode(b"<!doctype html><title></title>").decode(),
                ))
            elif (event.resource_type.value in BLOCKED_RESOURCE_TYPES
                  or any(keyword in url for keyword in BLOCKED_URL_KEYWORDS)):
                await tab.send(cdp.fetch.fail_request(
                    request_id=event.request_id, error_reason=cdp.network.ErrorReason.BLOCKED_BY_CLIENT))
            else:
                await tab.send(cdp.fetch.continue_request(request_id=event.request_id))
        except Exception as exc:
            log.debug("Fetch interception for %s failed: %s", url, exc)

    patterns = [cdp.fetch.RequestPattern(resource_type=cdp.network.ResourceType(t)) for t in BLOCKED_RESOURCE_TYPES]
    patterns += [cdp.fetch.RequestPattern(url_pattern=f"*{keyword}*") for keyword in BLOCKED_URL_KEYWORDS]
    if referrer:
        patterns.append(cdp.fetch.RequestPattern(url_pattern=referrer))
    tab.add_handler(cdp.fetch.RequestPaused, _on_paused)
    await tab.send(cdp.fetch.enable(patterns=patterns))


async def _navigate_from(tab: nodriver.Tab, referrer: str, url: str, deadline: float) -> None:
    """
    Navigates to `url` the way a click on the referrer's page does.

    Page.navigate is browser-initiated: given a referrer, Chrome sends the
    Referer header but Sec-Fetch-Site: none and no Sec-Fetch-User, a pairing
    no real click produces. So the referrer URL is answered inside the browser
    with an empty page (by _intercept — nothing reaches the network), and the
    target is opened from that page under a user gesture. Chrome then sets
    Referer, Sec-Fetch-Site: cross-site and Sec-Fetch-User: ?1 itself.
    """
    await tab.send(nodriver.cdp.page.navigate(url=referrer))
    while _remaining(deadline) > 0:
        try:
            if await tab.evaluate("location.href") == referrer:
                break
        except Exception:
            pass
        await asyncio.sleep(POLL_SECONDS)
    await tab.send(nodriver.cdp.runtime.evaluate(
        expression=f"location.href = {json.dumps(url)}", user_gesture=True))


async def _execute_fetch(browser, req: NoDriverFetchRequest) -> NoDriverFetchResponse:
    loop     = asyncio.get_running_loop()
    deadline = loop.time() + req.timeout_seconds
    referrer = _REFERRER_URL[_resolve_referrer_hint(req.referrer_hint)]

    forwarder: _AsyncProxyForwarder | None = None
    tab:       nodriver.Tab | None         = None
    try:
        # ── 1. Exit timezone + local proxy forwarder ─────────────────────────
        timezone = req.timezone_id
        local_proxy: str | None = None
        if req.proxy_url:
            timezone = await asyncio.wait_for(
                asyncio.to_thread(_exit_timezone, req.proxy_url),
                timeout=min(EXIT_GEO_TIMEOUT_SECONDS, max(_remaining(deadline), 1)))
            forwarder = _AsyncProxyForwarder(req.proxy_url)
            await forwarder.start()
            local_proxy = forwarder.proxy_server

        # ── 2. Isolated context with its own proxy and cookie jar ────────────
        tab = await browser.create_context(proxy_server=local_proxy)

        # ── 3. Main-frame document responses, in arrival order ───────────────
        main_frame = str(tab.target.target_id)  # a top-level target's main frame id is its target id
        documents: list = []

        def _on_response(event: nodriver.cdp.network.ResponseReceived) -> None:
            if (event.type_ == nodriver.cdp.network.ResourceType.DOCUMENT
                    and str(event.frame_id) == main_frame
                    and not (referrer and event.response.url.startswith(referrer))):
                documents.append(event)

        tab.add_handler(nodriver.cdp.network.ResponseReceived, _on_response)
        await tab.send(nodriver.cdp.network.enable())
        await _intercept(tab, referrer)
        await tab.send(nodriver.cdp.emulation.set_timezone_override(timezone_id=timezone))

        # ── 4. Navigate — from the referrer's own page when there is one ─────
        if referrer:
            await _navigate_from(tab, referrer, req.url, deadline)
        else:
            await tab.send(nodriver.cdp.page.navigate(url=req.url))
        await _wait_for_document(documents, 1, deadline)
        await _wait_until_parsed(tab, deadline, leaving=referrer or "about:blank")

        document = documents[-1]
        if document.response.status in CHALLENGE_STATUSES and await _is_challenge(tab, document):
            document = await _await_challenge(tab, req.url, documents, deadline)

        status_code = document.response.status
        final_url   = document.response.url or tab.url
        log.info("navigation complete  status=%d  url=%s", status_code, req.url)

        # ── 5. A non-HTML document is the payload itself ─────────────────────
        mime = (document.response.mime_type or "").lower()
        if mime and "html" not in mime:
            body, is_base64 = await tab.send(
                nodriver.cdp.network.get_response_body(request_id=document.request_id))
            if is_base64:
                body = base64.b64decode(body).decode("utf-8", "replace")
            return NoDriverFetchResponse(html=body, final_url=final_url, status_code=status_code)

        # ── 6. Dwell / selector wait ─────────────────────────────────────────
        if req.wait_for_selector:
            try:
                await asyncio.wait_for(tab.select(req.wait_for_selector),
                                       timeout=min(5.0, max(_remaining(deadline), 1)))
            except Exception:
                log.debug("wait_for_selector '%s' timed out for %s — proceeding",
                          req.wait_for_selector, req.url)
        elif req.default_wait_millis > 0:
            await _dwell(tab, req.default_wait_millis)

        # ── 7. Extract HTML, once the page has stopped changing documents ────
        await _wait_until_settled(tab, documents, deadline)
        if documents[-1] is not document:
            document    = documents[-1]
            status_code = document.response.status
            final_url   = document.response.url or final_url
            log.info("page reloaded itself  status=%d  url=%s", status_code, req.url)
        content: str = await tab.get_content()
        if not content or not content.strip():
            log.error("get_content() returned empty DOM for %s", req.url)
            return NoDriverFetchResponse(html=None, final_url=final_url, status_code=status_code)
        return NoDriverFetchResponse(html=content, final_url=final_url, status_code=status_code)

    finally:
        if tab is not None:
            context_id = tab.target.browser_context_id
            try:
                await tab.close()
            except Exception as exc:
                log.warning("Error closing tab: %s", exc)
            if context_id is not None:
                try:
                    await browser.connection.send(
                        nodriver.cdp.target.dispose_browser_context(browser_context_id=context_id))
                except Exception as exc:
                    log.debug("Context %s already disposed: %s", context_id, exc)
        if forwarder is not None:
            try:
                await forwarder.stop()
            except Exception as exc:
                log.warning("Error stopping proxy forwarder: %s", exc)


class _AsyncProxyForwarder:
    """
    Minimal asyncio HTTP CONNECT tunnel for authenticated upstream proxies.

    Chrome silently ignores embedded credentials for HTTP proxies. This
    forwarder accepts unauthenticated CONNECT requests from Chrome on a random
    localhost port and re-issues them to the upstream DataImpulse proxy with a
    Proxy-Authorization header injected. One per fetch; stopped after it.
    """

    _BUFFER_SIZE = 65536
    _CONNECT_TIMEOUT = 10.0

    def __init__(self, upstream_url: str) -> None:
        parsed              = urlparse(upstream_url)
        self._upstream_host = parsed.hostname or ""
        self._upstream_port = parsed.port or 8080
        self._auth_header: str | None = None
        if parsed.username and parsed.password:
            from urllib.parse import unquote
            creds = f"{unquote(parsed.username)}:{unquote(parsed.password)}"
            self._auth_header = "Basic " + base64.b64encode(creds.encode()).decode()
        self._server: asyncio.Server | None = None
        self._local_port: int = 0
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_connection, "127.0.0.1", 0)
        self._local_port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """
        Closes the listener and every tunnel still open. Chrome keeps its proxy
        tunnels alive after the tab closes, and since Python 3.12
        Server.wait_closed() waits for all of them — so without closing them
        here, stop() hung until the fetch's whole budget ran out.
        """
        if self._server is None:
            return
        self._server.close()
        for writer in list(self._writers):
            try:
                writer.close()
            except Exception:
                pass
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout=2)
        except asyncio.TimeoutError:
            log.debug("Proxy forwarder still had connections after close")
        self._server = None

    @property
    def proxy_server(self) -> str:
        return f"http://127.0.0.1:{self._local_port}"

    async def _handle_connection(self, client_reader: asyncio.StreamReader,
                                 client_writer: asyncio.StreamWriter) -> None:
        """
        Reads Chrome's CONNECT, re-issues it to the upstream proxy with
        Proxy-Authorization, and pipes bytes both ways once the upstream accepts.
        """
        self._writers.add(client_writer)
        try:
            request_line = await client_reader.readline()
            if not request_line:
                return
            while True:
                header_line = await client_reader.readline()
                if header_line in (b"\r\n", b"\n", b""):
                    break

            parts = request_line.decode(errors="replace").split()
            if len(parts) < 2 or parts[0].upper() != "CONNECT":
                client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await client_writer.drain()
                return
            host_port = parts[1]

            try:
                up_reader, up_writer = await asyncio.wait_for(
                    asyncio.open_connection(self._upstream_host, self._upstream_port),
                    timeout=self._CONNECT_TIMEOUT)
            except (asyncio.TimeoutError, OSError) as exc:
                log.warning("Upstream proxy connection failed: %s", exc)
                client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await client_writer.drain()
                return

            connect_lines = [f"CONNECT {host_port} HTTP/1.1\r\n", f"Host: {host_port}\r\n"]
            if self._auth_header:
                connect_lines.append(f"Proxy-Authorization: {self._auth_header}\r\n")
            connect_lines.append("\r\n")
            up_writer.write("".join(connect_lines).encode())
            await up_writer.drain()

            upstream_response = await up_reader.readline()
            while True:
                line = await up_reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            if b"200" not in upstream_response:
                log.warning("Upstream proxy rejected CONNECT: %s",
                            upstream_response.decode(errors="replace").strip())
                client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await client_writer.drain()
                up_writer.close()
                return

            self._writers.add(up_writer)
            client_writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await client_writer.drain()
            await asyncio.gather(self._pipe(client_reader, up_writer),
                                 self._pipe(up_reader, client_writer),
                                 return_exceptions=True)
        except Exception as exc:
            log.debug("CONNECT handler error: %s", exc)
        finally:
            self._writers.discard(client_writer)
            try:
                client_writer.close()
            except Exception:
                pass

    async def _pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(self._BUFFER_SIZE):
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/fetch", response_model=NoDriverFetchResponse)
async def execute_fetch(req: NoDriverFetchRequest) -> NoDriverFetchResponse:
    """
    Fetches a URL in a fresh Chrome context. The semaphore mirrors the Java
    BrowserSidecarFetcher's Semaphore(1). Execution failures return
    status_code=502 in the body, so Java's retry path handles them; a browser
    that cannot be relaunched is a 503, which Java counts against the sidecar.
    """
    async with _fetch_semaphore:
        try:
            browser = await _ensure_browser()
        except Exception as exc:
            log.exception("Could not relaunch the browser: %s", exc)
            raise HTTPException(status_code=503, detail="Browser not available") from exc
        try:
            # The deadline inside bounds every wait; this bounds a CDP call that never answers.
            return await asyncio.wait_for(_execute_fetch(browser, req), timeout=req.timeout_seconds + 2)
        except Exception as exc:
            log.exception("zendriver fetch failed for %s: %s", req.url, exc)
            return NoDriverFetchResponse(html=None, final_url=None, status_code=502)


@app.get("/api/v1/browser-version")
async def get_browser_version() -> dict[str, str]:
    """The running Chrome's version; BrowserSidecarValidator logs it at Java startup."""
    browser = _state.get("browser")
    if not _browser_alive(browser):
        raise HTTPException(status_code=503, detail="Browser not available")
    try:
        version_info = await browser.connection.send(nodriver.cdp.browser.get_version())
        # (protocolVersion, product, revision, userAgent, jsVersion); product is "Chrome/155.0.8059.39"
        product: str = version_info[1] if len(version_info) > 1 else ""
        version: str = product.split("/")[1] if "/" in product else product
        return {"version": version, "major": version.split(".")[0]}
    except Exception as exc:
        log.warning("Could not retrieve browser version: %s", exc)
        raise HTTPException(status_code=503, detail="Browser version unavailable") from exc


@app.get("/health")
async def health() -> dict[str, object]:
    """{"status": "ok"} while Chrome is up and its CDP connection open."""
    connected = _browser_alive(_state.get("browser"))
    return {"status": "ok" if connected else "degraded", "browser_connected": connected}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("NODRIVER_HOST", "127.0.0.1")
    port = int(os.getenv("NODRIVER_PORT", "8084"))
    uvicorn.run(app, host=host, port=port, log_level="info")
