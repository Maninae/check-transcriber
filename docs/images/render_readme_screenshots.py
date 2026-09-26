import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "tests"))
from browser_test_helpers import serve_directory, paste_image_file, wait_for_debug_state, wait_for_engines_ready
from playwright.sync_api import sync_playwright
scene = Path(sys.argv[1]); out = Path("/tmp/ct_readme")
hints = {k: {"opens": 9, "dismissed": True} for k in ["pasteFromGmail","fixOutlines","reviewGuide"]}
with serve_directory() as url, sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width":1366,"height":768}, device_scale_factor=1)
    pg = ctx.new_page()
    pg.goto(url)
    pg.evaluate("(v)=>localStorage.setItem('checkTranscriber.v1.firstRunHints', v)", json.dumps(hints))
    pg.reload()
    wait_for_engines_ready(pg, 180000)
    paste_image_file(pg, scene)
    st = wait_for_debug_state(pg, "state.step === 'count'", 120000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(out/"count_full.png"))
    pg.locator("#count-step").screenshot(path=str(out/"count_el.png"))
    pg.click("#count-continue-button")
    wait_for_debug_state(pg, "state.timings.gridCompleteAt > state.timings.continuedAt", 300000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(out/"review_full.png"))
    pg.locator("#review-step").screenshot(path=str(out/"review_el.png"))
    print(json.dumps({k: st.get(k) for k in ("step",)}), pg.locator(".review-row").count())
    b.close()
