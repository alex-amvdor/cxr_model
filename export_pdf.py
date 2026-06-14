import sys, asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.set_event_loop_policy = lambda *a, **k: None  # block jupyter from reverting it

from nbconvert.nbconvertapp import main

sys.argv = ["jupyter-nbconvert", "--to", "webpdf", "--output-dir", "results", "cxr_analysis_zhai.ipynb"]
main()