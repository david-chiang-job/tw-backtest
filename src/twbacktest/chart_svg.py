"""從四份圖表資料檔畫 SVG（只用標準庫、不需要行情或重跑回測）：
py -m twbacktest.chart_svg <資料夾>"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from html import escape

COLORS = ("#1462a3", "#a34b0e", "#6042a6", "#06715c", "#a23e78", "#ad2222", "#635321")
GRAYS = ("#444", "#777", "#999", "#555", "#888")
FONTS = '"Microsoft JhengHei","PingFang TC",Arial,"Noto Sans CJK TC",sans-serif'

LEFT, RIGHT, TOP, BOTTOM = 105, 1130, 85, 560
WIDTH = 1200


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _hover(day, center, left, right, candle, values, bottom):
    """每個交易日一塊游標目標，說明用純 CSS 的 :hover 顯示。"""
    preferred = center + 12 if center < 600 else center - 652
    box_x = min(max(preferred, LEFT), 490) - center
    height = 66 + 23 * ((len(values) + 2) // 3)
    ohlc = "   ".join(f"{lab} {float(candle[k]):,.2f}" for k, lab in zip(("open", "high", "low", "close"), "OHLC"))
    out = [f'<g class="hover-day" data-date="{escape(day)}" transform="translate({center:.2f} 0)">',
           f'<rect x="{left - center:.3f}" y="{TOP}" width="{right - left:.3f}" height="{bottom - TOP}" fill="transparent"/>',
           '<g class="hover-detail">',
           f'<path d="M0 {TOP}V{bottom}" stroke="#424f61" stroke-width="1.4" stroke-dasharray="4 3"/>',
           f'<g transform="translate({box_x:.2f} {TOP + 7})">',
           f'<rect width="640" height="{height}" rx="6" fill="#fff" stroke="#65748b"/>',
           f'<text x="13" y="23" font-weight="bold">{escape(day)}</text>',
           f'<text x="13" y="46">{ohlc}</text>']
    for i, v in enumerate(values):
        out.append(f'<text x="{13 + 208 * (i % 3)}" y="{69 + 23 * (i // 3)}">{escape(v)}</text>')
    out.append("</g></g></g>")
    return "\n".join(out)


def _draw(window, info, prices, levels, events):
    start, end, strategy = window["start"], window["end"], window["strategy"]
    cycle = window["cycle"]
    days = sorted({r["date"] for r in prices if start <= r["date"] <= end})
    if not days:
        raise ValueError(f"策略 {strategy} {start} 至 {end} 沒有行情")
    index = {d: i for i, d in enumerate(days)}
    px = {r["date"]: r for r in prices if r["date"] in index}
    lv = [r for r in levels if r["strategy"] == strategy and r["date"] in index]
    ev = [r for r in events if r["strategy"] == strategy and r["cycle"] == cycle and r["window_start"] == start
          and r["trade_date"] in index]
    tiers, lines = info["tiers"], [x["line"] for x in info["lines"]]
    names = {t["slot"]: t["name"] for t in tiers}

    vals = [float(px[d][k]) for d in days for k in ("high", "low")]
    vals += [float(r[k]) for r in lv for k in ("moving_average", "threshold") if r[k]]
    vals += [float(r["price"]) for r in ev]
    lo, hi = min(vals), max(vals)
    margin = (hi - lo) * 0.06 or max(abs(hi) * 0.01, 1)
    lo, hi = lo - margin, hi + margin
    y = lambda v: BOTTOM - (v - lo) / (hi - lo) * (BOTTOM - TOP)
    x = lambda d: LEFT + index[d] * (RIGHT - LEFT) / max(len(days) - 1, 1)

    legend_rows = len(tiers) + len(lines) + 1
    height = BOTTOM + 70 + 25 * legend_rows
    title = f"Strategy {strategy} · {window['kind']}{(' ' + cycle) if cycle else ''}: {start} to {end}"
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
         f'<rect width="{WIDTH}" height="{height}" fill="white"/>',
         f'<style>text{{font-family:{FONTS};fill:#222;font-size:15px}}'
         '.title{font-size:21px;font-weight:bold}.grid{stroke:#ddd;stroke-width:1}'
         '.hover-detail{display:none;pointer-events:none}'
         '.hover-day:hover>.hover-detail{display:inline}</style>',
         f'<text class="title" x="{LEFT}" y="30">{escape(title)}</text>',
         f'<text x="{LEFT}" y="62">Total return index (index points)</text>']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        yy = y(v)
        p.append(f'<line class="grid" x1="{LEFT}" x2="{RIGHT}" y1="{yy:.2f}" y2="{yy:.2f}"/>')
        p.append(f'<text x="95" y="{yy + 5:.2f}" text-anchor="end">{v:,.0f}</text>')
    p.append(f'<line x1="{LEFT}" x2="{LEFT}" y1="{TOP}" y2="{BOTTOM}" stroke="#555"/>')

    width = min(6, max(0.55, 650 / len(days)))
    for d in days:
        r = px[d]
        xx = x(d)
        op, cl = float(r["open"]), float(r["close"])
        color = "#b33a3a" if cl >= op else "#168257"
        yo, yc = y(op), y(cl)
        p.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{y(float(r["high"])):.2f}" y2="{y(float(r["low"])):.2f}" stroke="{color}"/>')
        p.append(f'<rect x="{xx - width / 2:.2f}" y="{min(yo, yc):.2f}" width="{width:.2f}" '
                 f'height="{max(abs(yo - yc), 0.8):.2f}" fill="{color}"/>')

    def polyline(points, color, w, dash=""):
        """points 是 [(日期, 值或 None)]；None 斷開。"""
        run = []
        for d, v in points + [(None, None)]:
            if v is None:
                if len(run) > 1:
                    pts = " ".join(f"{x(a):.2f},{y(b):.2f}" for a, b in run)
                    p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}"{dash}/>')
                run = []
            else:
                run.append((d, v))

    polyline([(d, float(px[d]["close"])) for d in days], "#111", 2)
    series = defaultdict(dict)  # (level, slot) -> {date: 值}
    for r in lv:
        if r["level"] == "line":
            series[("line", r["slot"])][r["date"]] = float(r["moving_average"]) if r["moving_average"] else None
        else:
            series[("tier", int(r["slot"]))][r["date"]] = float(r["threshold"]) if r["threshold"] else None
    for k, name in enumerate(lines):
        polyline([(d, series[("line", name)].get(d)) for d in days], GRAYS[k % len(GRAYS)], 1.6, ' stroke-dasharray="5 3"')
    for t in tiers:
        s = t["slot"]
        polyline([(d, series[("tier", s)].get(d)) for d in days], COLORS[(s - 1) % len(COLORS)], 1.4)

    for e in ev:
        day, side = e["trade_date"], e["side"]
        xx, yy = x(day), y(float(e["price"]))
        color = "#c23030" if side == "buy" else "#187645"
        d = -1 if side == "buy" else 1
        pts = f"{xx:.2f},{yy + d * 7:.2f} {xx - 6:.2f},{yy - d * 5:.2f} {xx + 6:.2f},{yy - d * 5:.2f}"
        p.append(f'<polygon data-kind="trade" data-date="{escape(day)}" data-slot="{escape(e["slot"])}" '
                 f'data-price="{escape(e["price"])}" points="{pts}" fill="{color}" stroke="white"/>')
        sig = e["signal_date"]
        if sig and sig != day and sig in index:
            p.append(f'<circle data-kind="signal" data-date="{escape(sig)}" data-slot="{escape(e["slot"])}" '
                     f'cx="{x(sig):.2f}" cy="{y(float(px[sig]["close"])):.2f}" r="4" fill="white" '
                     'stroke="#333" stroke-width="1.5"/>')
    for i in sorted({round(j * (len(days) - 1) / 6) for j in range(7)}):
        p.append(f'<text x="{x(days[i]):.2f}" y="{BOTTOM + 25}" text-anchor="middle">{escape(days[i])}</text>')

    ly = BOTTOM + 60
    for t in tiers:
        color = COLORS[(t["slot"] - 1) % len(COLORS)]
        label = f"Slot {t['slot']} {t['name']}: {t['side']} → {t['target_stock'] * 100:g}% stock"
        p.append(f'<line x1="{LEFT}" x2="{LEFT + 28}" y1="{ly - 5}" y2="{ly - 5}" stroke="{color}" stroke-width="2"/>'
                 f'<text x="{LEFT + 34}" y="{ly}">{escape(label)}</text>')
        ly += 25
    for k, name in enumerate(lines):
        p.append(f'<line x1="{LEFT}" x2="{LEFT + 28}" y1="{ly - 5}" y2="{ly - 5}" stroke="{GRAYS[k % len(GRAYS)]}" '
                 f'stroke-width="2" stroke-dasharray="5 3"/><text x="{LEFT + 34}" y="{ly}">{escape(name)}</text>')
        ly += 25
    p.append(f'<text x="{LEFT}" y="{ly}">Black: close. Red ▲ buy. Green ▼ sell. ○ signal day of a filled trade.</text>')

    by_day = defaultdict(list)
    for r in lv:
        by_day[r["date"]].append(r)
    centers = [x(d) for d in days]
    edges = [LEFT, *((a + b) / 2 for a, b in zip(centers, centers[1:])), RIGHT]
    for i, d in enumerate(days):
        values = [f"{r['slot']} {float(r['moving_average']):,.2f}" for r in by_day[d] if r["level"] == "line" and r["moving_average"]]
        values += [f"{names[int(r['slot'])]} trigger {float(r['threshold']):,.2f}" for r in by_day[d]
                   if r["level"] != "line" and r["threshold"]]
        p.append(_hover(d, centers[i], edges[i], edges[i + 1], px[d], values, BOTTOM))
    p.append("</svg>")
    return "\n".join(p) + "\n"


def render_charts(out_dir):
    """照 Chart Data.json 的 windows 逐張畫，回傳 [(window, file)]。"""
    with open(os.path.join(out_dir, "Chart Data.json"), encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version") != 1:
        raise ValueError(f"不支援的圖表資料版本：{doc.get('version')!r}")
    prices = _read_csv(os.path.join(out_dir, "Chart Prices.csv"))
    levels = _read_csv(os.path.join(out_dir, "Chart Levels.csv"))
    events = _read_csv(os.path.join(out_dir, "Chart Events.csv"))
    out = []
    for w in doc["windows"]:
        svg = _draw(w, doc["strategies"][w["strategy"]], prices, levels, events)
        with open(os.path.join(out_dir, w["file"]), "w", encoding="utf-8", newline="") as f:
            f.write(svg)
        out.append((w, w["file"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twbacktest.chart_svg", description="從圖表資料四份檔重畫 SVG")
    ap.add_argument("folder")
    a = ap.parse_args(argv)
    try:
        n = len(render_charts(a.folder))
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"圖 {n} 張")
    return 0


if __name__ == "__main__":
    sys.exit(main())
