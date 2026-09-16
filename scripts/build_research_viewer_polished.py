"""Build a polished, data-driven offline research viewer."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import compute_stats

ROOT = Path(__file__).resolve().parents[1] / "reports"
OUT = ROOT / "research_viewer.html"

def make_payload() -> dict:
    stats = pd.read_csv(ROOT / "backtest_stats.csv").to_dict("records")
    strategy_labels = {
        "raw": "Trend-following raw funds",
        "neutral": "Trend-following factor-neutral funds",
    }
    for row in stats:
        row["strategy"] = strategy_labels.get(row["strategy"], row["strategy"])
    alpha_stats = pd.read_csv(ROOT / "alpha_backtest_stats.csv").to_dict("records")[0]
    alpha_equity = pd.read_csv(ROOT / "alpha_backtest_equity.csv", index_col=0).iloc[:, 0].fillna(1).tolist()
    trend_equity = pd.read_csv(ROOT / "backtest_equity.csv", index_col=0).fillna(1.0).tail(len(alpha_equity))
    trend_equity = trend_equity.div(trend_equity.iloc[0])
    factors = pd.read_csv(ROOT / "oos_latent_factor_moves.csv", index_col=0)
    fitted = pd.read_csv(ROOT / "strategy_oos_fitted.csv", index_col=0)
    residual = pd.read_csv(ROOT / "strategy_oos_residuals.csv", index_col=0)
    residual_equal_weight_returns = residual.fillna(0.0).mean(axis=1)
    residual_equal_weight_equity = (1.0 + residual_equal_weight_returns).cumprod()
    residual_vol, residual_cagr, residual_drawdown, residual_sharpe, _ = compute_stats(
        residual_equal_weight_equity, residual_equal_weight_returns
    )
    raw = pd.read_csv(ROOT.parent / "data/raw/daily_returns.csv", index_col=0)[fitted.columns].loc[fitted.index]
    loads = pd.read_csv(ROOT / "frozen_factor_loadings.csv", index_col=0)
    fund_betas = pd.read_csv(ROOT / "strategy_train_exposures.csv", index_col=0)
    cluster = pd.read_csv(ROOT / "benchmark_cluster_order.csv")["ticker"].tolist()
    raw_corr, resid_corr = raw.corr(), residual.corr()
    points = []
    for i, a in enumerate(raw.columns):
        for b in raw.columns[i + 1:]:
            x, y = float(raw_corr.loc[a, b]), float(resid_corr.loc[a, b])
            points.append({"a": a, "b": b, "raw": x, "residual": y, "delta": y - x})
    return {
        "stats": stats, "alpha": alpha_stats, "alpha_equity": alpha_equity,
        "trend_equity": trend_equity.to_dict("list"),
        "residual_equal_weight_stats": {
            "strategy": "Naive equal-weight residual diagnostic",
            "target_vol": 0.0,
            "annual_vol": residual_vol,
            "cagr": residual_cagr,
            "max_drawdown": -residual_drawdown,
            "sharpe": residual_sharpe,
        },
        "factors": {"dates": [str(x) for x in factors.index], "names": list(factors), "values": factors.fillna(0).to_numpy().tolist()},
        "funds": {"dates": [str(x) for x in fitted.index], "names": list(fitted), "raw": raw.fillna(0).to_numpy().tolist(), "fitted": fitted.fillna(0).to_numpy().tolist(), "residual": residual.fillna(0).to_numpy().tolist()},
        "loadings": loads.to_dict("index"), "betas": fund_betas.to_dict("index"), "benchmarks": cluster, "scatter": points,
    }

def render(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Factor Model Research Viewer</title>
<style>
:root{{--ink:#17282e;--muted:#68787b;--paper:#f4f1e8;--panel:#fffdf8;--line:#d5d4ca;--teal:#176b87;--green:#2f855a;--red:#b94a48;--blue:#2e67a5;--gold:#b87919}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Georgia,serif;background-image:linear-gradient(rgba(23,107,135,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(23,107,135,.035) 1px,transparent 1px);background-size:34px 34px}}.shell{{max-width:1480px;margin:auto;padding:26px 32px 60px}}header{{border-bottom:2px solid var(--ink);padding-bottom:20px;display:flex;justify-content:space-between;gap:24px}}.kicker,.ui{{font:700 11px 'Segoe UI',sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--teal)}}h1{{font-size:clamp(35px,5vw,68px);line-height:.98;font-weight:500;margin:10px 0;max-width:850px}}.lede,.intro{{color:var(--muted);font-size:17px;max-width:950px}}.meta{{font:12px 'Segoe UI',sans-serif;color:var(--muted);text-align:right}}.layout{{display:grid;grid-template-columns:220px 1fr;gap:28px;margin-top:25px}}nav{{position:sticky;top:15px;align-self:start;display:grid;gap:7px}}nav button,.control button{{border:1px solid var(--line);background:var(--panel);padding:10px 12px;text-align:left;cursor:pointer;color:var(--ink)}}nav button.active,nav button:hover{{background:var(--ink);color:#fff;border-color:var(--ink)}}.chapter{{display:none}}.chapter.active{{display:block;animation:up .3s ease}}@keyframes up{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:none}}}}h2{{font-size:37px;line-height:1.05;font-weight:500;margin:0 0 8px}}h3{{font-size:21px;font-weight:500;margin:26px 0 8px}}.flow,.cards{{display:grid;grid-template-columns:repeat(5,1fr);border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);margin:24px 0}}.cards{{grid-template-columns:repeat(3,1fr);border:0;gap:10px}}.step,.card{{padding:15px;border-right:1px solid var(--line);background:rgba(255,253,248,.55)}}.step:last-child{{border:0}}.step b{{color:var(--teal);font:700 11px 'Segoe UI',sans-serif}}.step span{{display:block;margin-top:8px}}.card{{border:1px solid var(--line);background:var(--panel)}}.card b{{font-size:28px;font-weight:500;display:block;margin:5px 0}}.card small{{color:var(--muted);font:12px 'Segoe UI',sans-serif}}.note{{border-left:4px solid var(--teal);background:#e5efed;padding:13px 16px;color:#345057}}.actions,.control,.legend{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:17px 0}}.actions a{{color:var(--teal);font:13px 'Segoe UI',sans-serif;border-bottom:1px solid currentColor;text-decoration:none}}select{{border:1px solid var(--line);background:var(--panel);padding:10px;min-width:220px}}.chart,.tablewrap{{background:var(--panel);border:1px solid var(--line);padding:12px;margin-top:14px;overflow:auto}}svg{{display:block;width:100%;height:auto;min-height:300px}}.legend button{{border:1px solid var(--line);background:var(--panel);padding:6px 9px;cursor:pointer}}.legend button.off{{opacity:.38;text-decoration:line-through}}.swatch{{display:inline-block;width:11px;height:11px;border:1px solid #718187;margin-right:5px}}table{{border-collapse:collapse;width:100%;font:12px 'Segoe UI',sans-serif}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}@media(max-width:850px){{.shell{{padding:18px 14px}}header,.layout{{display:block}}.meta{{text-align:left;margin-top:16px}}nav{{position:static;display:flex;overflow:auto;margin-bottom:20px}}nav button{{white-space:nowrap}}.flow,.cards{{grid-template-columns:1fr}}.step{{border-right:0;border-bottom:1px solid var(--line)}}}}
</style></head><body><div class="shell"><header><div><div class="kicker">Statistical factor risk model · research viewer</div><h1>From benchmark structure to residual alpha.</h1><p class="lede">An interactive audit of the benchmark universe, frozen factor construction, hidden fund betas, residual performance, and the equal-risk alpha portfolio.</p></div><div class="meta">Offline HTML<br>Frozen loading space<br>Out-of-sample scoring</div></header><div class="layout"><nav><button class="active" data-tab="start">01 · Start here</button><button data-tab="universe">02 · Universes</button><button data-tab="factors">03 · Frozen factors</button><button data-tab="funds">04 · Fund selector</button><button data-tab="correlations">05 · Correlations</button><button data-tab="strategies">06 · Strategies</button></nav><main>
<section id="start" class="chapter active"><div class="kicker">Research map</div><h2>What is being explained, and what is left?</h2><p class="intro">Benchmark assets define the hidden risk space. Investible funds are held out from factor construction, regressed on daily latent factor moves, and stripped to residual returns. The alpha sleeve trades only that residual panel.</p><div class="flow"><div class="step"><b>BENCHMARKS</b><span>Clustered assets define the risk vocabulary.</span></div><div class="step"><b>LOADINGS</b><span>Chunked PCA loadings are aligned, averaged, and frozen.</span></div><div class="step"><b>FACTOR MOVES</b><span>Daily benchmark regressions estimate latent innovations OOS.</span></div><div class="step"><b>FUNDS</b><span>Training betas explain raw OOS performance.</span></div><div class="step"><b>ALPHA</b><span>Residuals feed the trend follower.</span></div></div><div class="cards"><article class="card"><span class="ui">Raw sleeve</span><b>10.2% vol</b><small>10.7% CAGR · 1.25x max leverage</small></article><article class="card"><span class="ui">Neutral sleeve</span><b>9.9% vol</b><small>2.1% CAGR · 2.50x max leverage</small></article><article class="card"><span class="ui">Residual alpha</span><b>32.7% CAGR</b><small>10.6% vol · Sharpe 2.72 · 9.0% drawdown</small></article></div><p class="note"><b>Reading guide:</b> start with the clustered universes, inspect the loading matrix and OOS factor moves, select any fund to inspect its hidden beta composition, then compare equal-weight raw, equal-weight residual, and residual trend-following performance.</p></section>
<section id="universe" class="chapter"><div class="kicker">Chapter 02 · Inputs</div><h2>Two universes, two jobs.</h2><p class="intro">The benchmark panel is the only source of factor definitions. The non-benchmark universe is investible research material and never defines the factor space.</p><div class="cards"><article class="card"><span class="ui">Benchmark assets</span><b id="benchmarkCount"></b><small>held out as the factor construction universe</small></article><article class="card"><span class="ui">Investible funds</span><b id="fundCount"></b><small>residuals become alpha inputs</small></article><article class="card"><span class="ui">OOS observations</span><b id="obsCount"></b><small>daily latent moves and fund scoring</small></article></div><div id="benchTable" class="tablewrap"></div><p class="note">The benchmark universe is clustered for model diagnostics, then used to construct the frozen loading space. The actual clustered order and loadings are shown together on the next page, where they can be read as model inputs rather than as a standalone ranking.</p></section>
<section id="factors" class="chapter"><div class="kicker">Chapter 03 · Frozen factor model</div><h2>Loadings first. Factor moves second.</h2><p class="intro">Historical benchmark blocks are PCA-decomposed, aligned, averaged, and frozen. Each OOS day is then projected onto that fixed loading space by linear regression. The chart below compounds the daily latent moves so factor persistence is visible through time.</p><div class="actions"><a href="frozen_factor_loadings.csv">Frozen loading matrix CSV</a><a href="oos_latent_factor_moves.csv">OOS factor moves CSV</a></div><div id="loadTable" class="tablewrap"></div><div class="chart"><div id="factorLegend" class="legend"></div><svg id="factorChart" viewBox="0 0 1000 440"></svg></div></section>
<section id="funds" class="chapter"><div class="kicker">Chapter 04 · Hidden betas</div><h2>Choose a fund. See the explanation.</h2><p class="intro">The dropdown drives the raw, factor-fitted, and residual performance chart. Legend controls hide and restore each series without changing the others.</p><div class="control"><label class="ui">Fund</label><select id="fundSelect"></select></div><div class="fundGrid"><div><div id="fundLegend" class="legend"></div><div class="chart"><svg id="fundChart" viewBox="0 0 1000 430"></svg></div></div><div id="fundStats" class="tablewrap"></div></div><div id="betaTable" class="tablewrap"></div></section>
<section id="strategies" class="chapter"><div class="kicker">Chapter 05 · Portfolio construction</div><h2>Equal-weight funds, then trend follow.</h2><p class="intro">Raw and residual equal-weight fund benchmarks are compared with the trend follower on the residual matrix. Signals and volatility estimates are lagged one day.</p><div id="strategyLegend" class="legend"></div><div class="chart"><svg id="strategyChart" viewBox="0 0 1000 430"></svg></div><div id="strategyTable" class="tablewrap"></div><div class="actions"><a href="alpha_backtest_stats.csv">Alpha statistics CSV</a><a href="figures/residual_alpha_trend_follower.html">Detailed alpha report</a></div></section>
<section id="correlations" class="chapter"><div class="kicker">Chapter 06 · Correlation movement</div><h2>What changes when hidden beta is removed?</h2><p class="intro">Each point is a fund pair. The x-axis is raw correlation, the y-axis is residual correlation, and color shows the change. Red means correlation increased; blue means it decreased. Borders keep white points visible.</p><div class="chart"><svg id="scatter" viewBox="0 0 1000 620"></svg></div></section>
</main></div></div><script>const DATA={payload};
const $=id=>document.getElementById(id), COLORS=['#176b87','#2f855a','#b87919','#b94a48','#2e67a5','#6b4f9e','#127c78','#9a5b13'];
function esc(v){{return String(v).replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]))}}
function cum(values){{let x=1;return values.map(v=>{{x*=1+(v||0);return x}})}}
function draw(id,series,state,legendId){{const svg=$(id), keys=Object.keys(series), shown=keys.filter(k=>state[k]), all=shown.flatMap(k=>series[k]).filter(Number.isFinite);if(!all.length)return;const lo=Math.min(...all),hi=Math.max(...all);svg.innerHTML='';$(legendId).innerHTML='';keys.forEach((key,i)=>{{const btn=document.createElement('button');btn.className=state[key]?'':'off';btn.innerHTML=`<span class="swatch" style="background:${{COLORS[i%COLORS.length]}}"></span>${{esc(key)}}`;btn.onclick=()=>{{state[key]=!state[key];draw(id,series,state,legendId)}};$(legendId).appendChild(btn);if(!state[key])return;const p=document.createElementNS('http://www.w3.org/2000/svg','polyline');p.setAttribute('points',series[key].map((v,j)=>`${{62+876*j/Math.max(1,series[key].length-1)}},${{35+355*(hi-v)/Math.max(1e-9,hi-lo)}}`).join(' '));p.setAttribute('fill','none');p.setAttribute('stroke',COLORS[i%COLORS.length]);const isReplicable=key.includes('Replicable');p.setAttribute('stroke-width',isReplicable?'5':'3');svg.appendChild(p)}})}}
function fundView(){{const f=DATA.funds,i=f.names.indexOf($('fundSelect').value),rows=f.raw.map((_,j)=>({{raw:f.raw[j][i],fitted:f.fitted[j][i],residual:f.residual[j][i]}}));draw('fundChart',{{'Raw performance':cum(rows.map(x=>x.raw)),'Factor-fitted performance':cum(rows.map(x=>x.fitted)),'Residual performance':cum(rows.map(x=>x.residual))}},{{'Raw performance':true,'Factor-fitted performance':true,'Residual performance':true}},'fundLegend');const b=DATA.betas[f.names[i]]||{{}};let h='<table><thead><tr><th>Frozen factor loading</th><th>Beta</th></tr></thead><tbody>';Object.entries(b).forEach(([k,v])=>h+=`<tr><td>${{esc(k)}}</td><td>${{Number(v).toFixed(5)}}</td></tr>`);$('betaTable').innerHTML=h+'</tbody></table>'}}
function scatter(){{const svg=$('scatter');svg.innerHTML='<line x1="70" y1="550" x2="950" y2="550" stroke="#17282e"/><line x1="70" y1="40" x2="70" y2="550" stroke="#17282e"/><text x="500" y="610" text-anchor="middle">Raw correlation</text><text x="18" y="300" transform="rotate(-90 18 300)">Residual correlation</text>';DATA.scatter.forEach(p=>{{const x=70+880*(p.raw+1)/2,y=550-510*(p.residual+1)/2,t=Math.max(-1,Math.min(1,p.delta)),pos=t>=0,r=pos?190+55*t:45+70*(1+t),g=pos?75+110*(1-t):110+90*(1+t),b=pos?75+110*(1-t):175+50*t,c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',x);c.setAttribute('cy',y);c.setAttribute('r',6);c.setAttribute('fill',`rgb(${{r}},${{g}},${{b}})`);c.setAttribute('stroke','#53636a');c.setAttribute('stroke-width','1.5');c.innerHTML=`<title>${{p.a}} / ${{p.b}} · raw ${{p.raw.toFixed(2)}} · residual ${{p.residual.toFixed(2)}} · change ${{p.delta.toFixed(2)}}</title>`;svg.appendChild(c)}})}}
function init(){{DATA.funds.names.forEach(n=>$('fundSelect').add(new Option(n,n)));$('fundSelect').onchange=fundView;$('benchmarkCount').textContent=DATA.benchmarks.length;$('fundCount').textContent=DATA.funds.names.length;$('obsCount').textContent=DATA.funds.dates.length;$('benchTable').innerHTML='<table><thead><tr><th>Cluster order</th><th>Benchmark</th></tr></thead><tbody>'+DATA.benchmarks.map((x,i)=>`<tr><td>${{i+1}}</td><td>${{x}}</td></tr>`).join('')+'</tbody></table>';$('loadTable').innerHTML='<table><thead><tr><th>Benchmark</th>'+Object.keys(DATA.betas[DATA.benchmarks[0]]||{{}}).map(x=>`<th>${{x}}</th>`).join('')+'</tr></thead><tbody>'+DATA.benchmarks.map(x=>`<tr><td>${{x}}</td>`+Object.keys(DATA.betas[x]||{{}}).map(k=>`<td>${{Number(DATA.betas[x][k]).toFixed(4)}}</td>`).join('')+'</tr>').join('')+'</tbody></table>';draw('factorChart',Object.fromEntries(DATA.factors.names.map((n,j)=>[n,DATA.factors.values.map(r=>r[j])])),Object.fromEntries(DATA.factors.names.map(n=>[n,true])),'factorLegend');const raw=DATA.funds.raw.map(r=>r.reduce((a,v)=>a+v,0)/DATA.funds.names.length),res=DATA.funds.residual.map(r=>r.reduce((a,v)=>a+v,0)/DATA.funds.names.length);draw('strategyChart',{{'Equal-weight raw funds':cum(raw),'Equal-weight residual funds':cum(res),'Residual alpha trend follower':DATA.alpha_equity}},{{'Equal-weight raw funds':true,'Equal-weight residual funds':true,'Residual alpha trend follower':true}},'strategyLegend');let h='<table><thead><tr><th>Strategy</th><th>Target vol</th><th>Realized vol</th><th>CAGR</th><th>Max DD</th><th>Sharpe</th></tr></thead><tbody>';DATA.stats.forEach(r=>h+=`<tr><td>${{r.strategy}}</td><td>${{(r.target_vol*100).toFixed(1)}}%</td><td>${{(r.annual_vol*100).toFixed(1)}}%</td><td>${{(r.cagr*100).toFixed(1)}}%</td><td>${{(r.max_drawdown*100).toFixed(1)}}%</td><td>${{r.sharpe.toFixed(2)}}</td></tr>`);h+=`<tr><td>Residual alpha</td><td>${{(DATA.alpha.target_vol*100).toFixed(1)}}%</td><td>${{(DATA.alpha.annual_vol*100).toFixed(1)}}%</td><td>${{(DATA.alpha.cagr*100).toFixed(1)}}%</td><td>${{(DATA.alpha.max_drawdown*100).toFixed(1)}}%</td><td>${{DATA.alpha.sharpe.toFixed(2)}}</td></tr></tbody></table>`;$('strategyTable').innerHTML=h;fundView();scatter()}}const buttons=[...document.querySelectorAll('nav button')],chapters=[...document.querySelectorAll('.chapter')];buttons.forEach(b=>b.onclick=()=>{{buttons.forEach(x=>x.classList.toggle('active',x===b));chapters.forEach(c=>c.classList.toggle('active',c.id===b.dataset.tab));history.replaceState(null,'','#'+b.dataset.tab)}});init();</script></body></html>'''

    diagnostic = data["residual_equal_weight_stats"]
    diagnostic_row = (
        f"<tr><td>{diagnostic['strategy']}</td><td>n/a</td>"
        f"<td>{diagnostic['annual_vol'] * 100:.1f}%</td>"
        f"<td>{diagnostic['cagr'] * 100:.1f}%</td>"
        f"<td>{diagnostic['max_drawdown'] * 100:.1f}%</td>"
        f"<td>{diagnostic['sharpe']:.2f}</td></tr>"
    )
    html = html.replace(
        '<div id="strategyLegend" class="legend"></div>',
        '<div class="axis-label">Y-axis: cumulative wealth (1.0 = start)</div>'
        '<div id="strategyLegend" class="legend"></div>',
    )
    html = html.replace(
        '</svg></div><div id="strategyTable"',
        '</svg><div class="axis-label">X-axis: out-of-sample date</div></div>'
        '<div id="strategyTable"',
    )
    html = html.replace(
        "$('strategyTable').innerHTML=h;",
        "$('strategyTable').innerHTML=h.replace('</tbody>', "
        + json.dumps(diagnostic_row)
        + " + '</tbody>');",
    )
    html = html.replace(
        "draw('strategyChart',{'Equal-weight raw funds':cum(raw),'Equal-weight residual funds':cum(res),'Residual alpha trend follower':DATA.alpha_equity},{'Equal-weight raw funds':true,'Equal-weight residual funds':true,'Residual alpha trend follower':true},'strategyLegend')",
        "draw('strategyChart',{'Trend-following raw funds':DATA.trend_equity.raw,'Trend-following factor-neutral funds':DATA.trend_equity.neutral,'Replicable residual fund + benchmark hedge':DATA.alpha_equity,'Naive equal-weight residual diagnostic':cum(res)},{'Trend-following raw funds':true,'Trend-following factor-neutral funds':true,'Replicable residual fund + benchmark hedge':true,'Naive equal-weight residual diagnostic':true},'strategyLegend')",
    )
    html = html.replace(
        "Equal-weight raw funds",
        "Equal-weight raw fund benchmark",
    )
    html = html.replace(
        "Equal-weight residual funds",
        "Naive equal-weight residual diagnostic",
    )
    html = html.replace(
        "Residual alpha trend follower",
        "Replicable residual fund + benchmark hedge",
    )
    html = html.replace('<td>raw</td>', '<td>Trend-following raw funds</td>')
    html = html.replace('<td>neutral</td>', '<td>Trend-following factor-neutral funds</td>')
    html = html.replace(
        '<td>Residual alpha</td>',
        '<td>Replicable residual fund + benchmark hedge</td>',
    )
    html = html.replace(
        "Raw and residual equal-weight fund benchmarks are compared with the trend follower on the residual matrix. Signals and volatility estimates are lagged one day.",
        "The diagnostic residual average is shown beside a modeled tradeable implementation: long fund positions plus a benchmark ETF hedge, with lagged signals, weekly rebalancing, volatility targeting, and a leverage cap.",
    )
    html = html.replace(
        '<div id="strategyTable" class="tablewrap">',
        '<p class="note">The replicable line uses the generated fund holdings and benchmark hedge exposures below. It is the executable approximation; the naive residual line is a diagnostic only.</p><div id="strategyTable" class="tablewrap">',
    )
    html = html.replace(
        '<a href="alpha_backtest_stats.csv">Alpha statistics CSV</a>',
        '<a href="alpha_backtest_stats.csv">Tradeable strategy statistics</a><a href="trend_residual_fund_weights.csv">Fund holdings CSV</a><a href="trend_residual_benchmark_hedge_exposures.csv">Benchmark hedge CSV</a>',
    )
    return html


def main() -> None:
    OUT.write_text(render(make_payload()), encoding="utf-8")
    print(f"Wrote {OUT.resolve()} ({OUT.stat().st_size / 1024:.1f} KB)")
if __name__ == "__main__": main()
