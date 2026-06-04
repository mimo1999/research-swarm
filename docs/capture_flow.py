"""
Playwright script to capture the Research Swarm UI flow.
Run with: python docs/capture_flow.py
Screenshots saved to docs/screenshots/
"""
import asyncio
import sys
from pathlib import Path

# Force UTF-8 output on Windows so emoji in page text don't crash print()
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DOCS = Path(__file__).parent
SHOTS = DOCS / "screenshots"
SHOTS.mkdir(exist_ok=True)

PDF1 = str(DOCS / "1-s2.0-S0377221723005027-main.pdf")
PDF2 = str(DOCS / "Methods Ecol Evol - 2022 - Clark - Dynamic generalised additive models  DGAMs  for forecasting discrete ecological time.pdf")
TOPIC = "GAMs for temporal data"


async def wait_for_text(page, text: str, timeout: int = 600_000):
    print(f"  waiting for: {text!r}")
    await page.wait_for_selector(f"text={text}", timeout=timeout)


async def shot(page, name: str):
    path = str(SHOTS / name)
    await page.screenshot(path=path, full_page=False)
    print(f"  saved: {path}")


async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, slow_mo=300)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto("http://localhost:8501", wait_until="networkidle", timeout=60_000)
        await page.wait_for_timeout(4000)

        # ── 1. Landing page ───────────────────────────────────────────────
        print("Step 1: landing")
        await shot(page, "01_landing.png")

        # ── 2. Fill topic ─────────────────────────────────────────────────
        print("Step 2: fill topic")
        topic_input = page.get_by_placeholder("e.g.  Impact of large language models")
        await topic_input.fill(TOPIC)
        await page.wait_for_timeout(500)
        await shot(page, "02_topic_filled.png")

        # ── 3. Enable HITL ────────────────────────────────────────────────
        print("Step 3: enable HITL")
        hitl = page.locator('input[type="checkbox"][aria-label="Pause before writing (HITL)"]')
        if not await hitl.is_checked():
            await hitl.check()
        await page.wait_for_timeout(500)

        # ── 4. Upload PDFs ────────────────────────────────────────────────
        print("Step 4: upload PDFs")
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files([PDF1, PDF2])
        await page.wait_for_timeout(2000)
        await shot(page, "03_docs_uploaded.png")

        # ── 5. Start research ─────────────────────────────────────────────
        print("Step 5: start research")
        await page.get_by_role("button", name="Start Research").click()
        await page.wait_for_timeout(3000)
        await shot(page, "04_research_started.png")

        # ── 6. Poll for agent trace content ──────────────────────────────
        print("Step 6: polling for agent trace (up to 3 min)...")
        for _ in range(36):  # 36 × 5 s = 3 min
            await page.wait_for_timeout(5000)
            body = await page.evaluate("document.body.innerText")
            print(f"  body snippet: {body[:120]!r}")
            if any(k in body for k in ["Supervisor", "supervisor", "Routing", "routing", "finding", "Researcher"]):
                break
        await shot(page, "05_agent_trace.png")

        # ── 7. Poll for HITL or report-ready ─────────────────────────────
        print("Step 7: polling for HITL / completion (up to 10 min)...")
        hitl_seen = False
        for _ in range(120):  # 120 × 5 s = 10 min
            await page.wait_for_timeout(5000)
            body = await page.evaluate("document.body.innerText")
            if "Human Review Required" in body:
                print("  HITL panel detected")
                await shot(page, "06_hitl_panel.png")
                hitl_seen = True
                break
            if "Report ready" in body or "Report complete" in body:
                print("  Report ready without HITL")
                await shot(page, "06_report_ready.png")
                break
            if _ % 6 == 5:
                print(f"  still running... snippet: {body[:80]!r}")

        # ── 8. Approve HITL if shown ───────────────────────────────────────
        if hitl_seen:
            print("Step 8: clicking Approve & Write...")
            try:
                approve_btn = page.get_by_role("button", name="Approve & Write")
                await approve_btn.wait_for(timeout=10_000)
                await shot(page, "07_hitl_full.png")
                await approve_btn.click()
                await page.wait_for_timeout(3000)
            except Exception as e:
                print(f"  approve click error: {e}")

        # ── 9. Wait for final report ───────────────────────────────────────
        print("Step 9: waiting for final report...")
        for _ in range(60):  # 60 × 5 s = 5 min
            await page.wait_for_timeout(5000)
            body = await page.evaluate("document.body.innerText")
            if "Report ready" in body or "Report complete" in body or "Switch to the" in body:
                await shot(page, "08_report_ready.png")
                break

        try:
            await page.get_by_role("tab", name="Report").click()
            await page.wait_for_timeout(3000)
            await shot(page, "09_report_tab.png")
        except Exception as e:
            print(f"  report tab error: {e}")

        # ── 10. Sessions tab ───────────────────────────────────────────────
        print("Step 10: sessions tab")
        try:
            await page.get_by_role("tab", name="Sessions").click()
            await page.wait_for_timeout(1500)
            await shot(page, "10_sessions_tab.png")
        except Exception as e:
            print(f"  sessions tab error: {e}")

        await browser.close()
        print("\nAll screenshots saved to docs/screenshots/")


if __name__ == "__main__":
    asyncio.run(main())
