"""
nodriver sidecar — FastAPI service for CDP-native undetected browser fetching.

VERSION: 1.2.0  (per-request proxy isolation via raw CDP + local CONNECT forwarder)

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

zendriver is the community-maintained fork created to compile unmerged bugfixes.
Its core API mirrors nodriver exactly, but its Browser class does not expose
create_context() as a Python method. Per-request proxy isolation is achieved by
calling the underlying CDP commands directly:
  - Target.createBrowserContext(proxyServer=...) — creates an isolated context
  - Target.createTarget(browserContextId=...) — opens a tab in that context
  - Target.disposeBrowserContext(...) — cleans up after the fetch

PROXY HANDLING (per-request isolation)
---------------------------------------
Chrome cannot use embedded credentials in --proxy-server for HTTP proxies.
_AsyncProxyForwarder solves this: it starts a local asyncio TCP server on a
random localhost port that speaks the HTTP CONNECT protocol, then forwards
tunnels to the upstream DataImpulse proxy, injecting Proxy-Authorization.

Per request:
  1. _AsyncProxyForwarder starts on 127.0.0.1:RANDOM_PORT
  2. CDP Target.createBrowserContext(proxyServer="http://127.0.0.1:RANDOM_PORT")
  3. Chrome connects to the forwarder; forwarder injects auth to DataImpulse
  4. After fetch: dispose browser context + stop forwarder

This gives every fetch attempt its own exit IP and cookie jar, matching the
proxy isolation behaviour of the other fetcher transports.

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
import base64
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
    Creates a per-request isolated CDP browser context with its own proxy,
    navigates to the target URL, applies dwell or selector wait, extracts
    HTML, and disposes the context.

    PROXY ISOLATION:
    _AsyncProxyForwarder starts a local CONNECT tunnel for the request's
    DataImpulse sticky-session URL, injecting Proxy-Authorization so Chrome
    can reach the upstream without embedded credentials in --proxy-server.
    Target.createBrowserContext binds Chrome's context to the local forwarder,
    giving each attempt its own exit IP and isolated cookie jar.

    ZENDRIVER NOTE:
    zendriver's Browser class does not expose create_context(). The equivalent
    is achieved by sending Target.createBrowserContext and Target.createTarget
    directly over the CDP connection, then locating the resulting tab via
    browser.targets once it is autodiscovered.
    """
    hint     = _resolve_referrer_hint(req.referrer_hint)
    referrer = _REFERRER_URL[hint]

    forwarder:          _AsyncProxyForwarder | None = None
    browser_context_id: str | None                  = None
    tab:                nodriver.Tab | None          = None

    try:
        # ── 1. Start local proxy forwarder ────────────────────────────────────
        local_proxy_server: str | None = None
        if req.proxy_url:
            forwarder = _AsyncProxyForwarder(req.proxy_url)
            await forwarder.start()
            local_proxy_server = forwarder.proxy_server
            log.debug("Local CONNECT forwarder started on %s → %s",
                      local_proxy_server, req.proxy_url.split("@")[-1])

        # ── 2. Create isolated CDP browser context with proxy ─────────────────
        browser_context_id = await browser.connection.send(
            nodriver.cdp.target.create_browser_context(
                proxy_server=local_proxy_server,
            )
        )

        # ── 3. Create a tab (target) inside this context ──────────────────────
        target_id = await browser.connection.send(
            nodriver.cdp.target.create_target(
                url="about:blank",
                browser_context_id=browser_context_id,
                new_window=True,
            )
        )

        # Allow zendriver's autodiscover loop to register the new target.
        deadline = asyncio.get_event_loop().time() + 3.0
        while asyncio.get_event_loop().time() < deadline:
            for t in browser.targets:
                if str(t.target_id) == str(target_id):
                    tab = t
                    break
            if tab is not None:
                break
            await asyncio.sleep(0.05)

        if tab is None:
            raise RuntimeError(
                f"Tab for target {target_id} not found in browser.targets "
                f"within 3 s. Context: {browser_context_id}"
            )

        # ── 4. Set Accept-Language and Referer before navigating ─────────────
        # zendriver's Tab.get() dropped the referrer parameter that upstream
        # nodriver exposed. Referer is injected via CDP set_extra_http_headers
        # instead. Note: this approach does not trigger Chrome's native
        # Sec-Fetch-Site logic (which only fires when referrer flows through
        # the navigation context), so Sec-Fetch-Site will be "none" regardless
        # of the referrer hint. The Referer header itself is still transmitted.
        extra_headers: dict[str, str] = {"Accept-Language": req.accept_language}
        if referrer:
            extra_headers["Referer"] = referrer

        await tab.send(
            nodriver.cdp.network.set_extra_http_headers(
                headers=nodriver.cdp.network.Headers(extra_headers)
            )
        )

        # ── 5. Navigate ───────────────────────────────────────────────────────
        await tab.get(req.url)

        status_code = await _resolve_status_code(tab, req.url)
        final_url   = tab.url
        log.info("navigation complete  status=%d  url=%s", status_code, req.url)

        # ── 6. Dwell / selector wait ──────────────────────────────────────────
        if req.wait_for_selector:
            try:
                await asyncio.wait_for(
                    tab.select(req.wait_for_selector),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, Exception):
                log.debug(
                    "wait_for_selector '%s' timed out for %s — proceeding",
                    req.wait_for_selector, req.url,
                )
        elif req.default_wait_millis > 0:
            await _dwell(tab, req.default_wait_millis)

        # ── 7. Extract HTML ───────────────────────────────────────────────────
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
        # ── Cleanup: tab → context → forwarder ───────────────────────────────
        # Each step is individually guarded so a failure in one does not
        # prevent the others from running.
        if tab is not None:
            try:
                await tab.close()
            except Exception as exc:
                log.warning("Error closing tab: %s", exc)

        if browser_context_id is not None:
            try:
                await browser.connection.send(
                    nodriver.cdp.target.dispose_browser_context(
                        browser_context_id=browser_context_id,
                    )
                )
            except Exception as exc:
                log.warning("Error disposing browser context %s: %s",
                            browser_context_id, exc)

        if forwarder is not None:
            try:
                await forwarder.stop()
            except Exception as exc:
                log.warning("Error stopping proxy forwarder: %s", exc)


class _AsyncProxyForwarder:
    """
    Minimal asyncio HTTP CONNECT tunnel for authenticated upstream proxies.

    Chrome's --proxy-server and CDP Target.createBrowserContext(proxyServer=...)
    both accept proxy URLs, but Chrome silently ignores embedded credentials
    for HTTP proxies. This forwarder accepts unauthenticated CONNECT requests
    from Chrome on a random localhost port and re-issues them to the upstream
    DataImpulse proxy with a Proxy-Authorization header injected.

    Lifecycle:
      forwarder = _AsyncProxyForwarder(proxy_url)
      await forwarder.start()
      # use forwarder.proxy_server as the proxy URL for CDP context creation
      await forwarder.stop()

    One forwarder per fetch attempt; MAX_CONCURRENT_FETCHES=1 ensures no
    concurrent forwarders share the same lifecycle.
    """

    _BUFFER_SIZE = 65536
    _CONNECT_TIMEOUT = 10.0

    def __init__(self, upstream_url: str) -> None:
        parsed              = urlparse(upstream_url)
        self._upstream_host = parsed.hostname or ""
        self._upstream_port = parsed.port or 8080
        self._auth_header: str | None = None
        if parsed.username and parsed.password:
            creds = f"{parsed.username}:{parsed.password}"
            encoded = base64.b64encode(creds.encode()).decode()
            self._auth_header = f"Basic {encoded}"
        self._server: asyncio.Server | None = None
        self._local_port: int = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, "127.0.0.1", 0
        )
        self._local_port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def proxy_server(self) -> str:
        return f"http://127.0.0.1:{self._local_port}"

    async def _handle_connection(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handles one CONNECT tunnel from Chrome.

        Reads the CONNECT request line, opens a TCP connection to the upstream
        proxy, re-issues the CONNECT with Proxy-Authorization injected, waits
        for the upstream 200 Connection established response, then pipes bytes
        bidirectionally until either side closes.
        """
        try:
            # Read Chrome's CONNECT request line and headers
            request_line = await client_reader.readline()
            if not request_line:
                return

            # Consume remaining request headers from Chrome (we don't forward them)
            while True:
                header_line = await client_reader.readline()
                if header_line in (b"\r\n", b"\n", b""):
                    break

            # Parse: CONNECT host:port HTTP/1.1
            parts = request_line.decode(errors="replace").split()
            if len(parts) < 2 or parts[0].upper() != "CONNECT":
                client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await client_writer.drain()
                return

            host_port = parts[1]
            if ":" in host_port:
                target_host, target_port_str = host_port.rsplit(":", 1)
                target_port = int(target_port_str)
            else:
                target_host = host_port
                target_port = 443

            # Connect to upstream proxy
            try:
                up_reader, up_writer = await asyncio.wait_for(
                    asyncio.open_connection(self._upstream_host, self._upstream_port),
                    timeout=self._CONNECT_TIMEOUT,
                )
            except (asyncio.TimeoutError, OSError) as exc:
                log.warning("Upstream proxy connection failed: %s", exc)
                client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await client_writer.drain()
                return

            # Re-issue CONNECT to upstream with auth injected
            connect_lines = [
                f"CONNECT {host_port} HTTP/1.1\r\n",
                f"Host: {host_port}\r\n",
            ]
            if self._auth_header:
                connect_lines.append(f"Proxy-Authorization: {self._auth_header}\r\n")
            connect_lines.append("\r\n")
            up_writer.write("".join(connect_lines).encode())
            await up_writer.drain()

            # Read upstream's response to our CONNECT
            upstream_response = await up_reader.readline()
            # Consume the rest of upstream's response headers
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

            # Tell Chrome the tunnel is established
            client_writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await client_writer.drain()

            # Pipe bytes bidirectionally until either side closes
            await asyncio.gather(
                self._pipe(client_reader, up_writer),
                self._pipe(up_reader, client_writer),
                return_exceptions=True,
            )

        except Exception as exc:
            log.debug("CONNECT handler error: %s", exc)
        finally:
            try:
                client_writer.close()
            except Exception:
                pass

    async def _pipe(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Copies bytes from reader to writer until EOF or error."""
        try:
            while True:
                data = await reader.read(self._BUFFER_SIZE)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass


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
