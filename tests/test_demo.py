"""spatialmem.demo on one small HSSD dev scene: offline HTML, embedded object count == the prediction's."""
import json
import re

import pytest

from spatialmem import demo

SCENE = "104348253_171513237"   # smallest dev scene


def test_demo_html_offline(tmp_path, monkeypatch):
    src = demo.data_dir()
    if not (src / "baselines" / "cg" / "runs" / "hssd" / SCENE / "objects.npz").exists():
        pytest.skip("HSSD / CG data not available")
    html = demo.build(SCENE, src)
    text = html.read_text(encoding="utf-8")
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", text)
    assert "http://" not in text and "https://" not in text
    assert "<link" not in text and not re.search(r"<script[^>]*\bsrc=", text)
    d = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S).group(1).replace(r"<\/", "</"))
    pred, _, _, _ = demo.load(SCENE, src)
    assert len(d["pred"]) == len(pred.objects) == d["pts"]["off"].__len__() - 1
    assert len(d["prel"]) == len(pred.relations)
