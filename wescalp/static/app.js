// Painel do Wescalp: candles + VWAP/EMAs, marcas dos setups, buracos do livro, fluxo e estado do coletor.
// Sem build: Lightweight Charts 4.2.0 do unpkg e JS puro. Só lê /api/*.
'use strict';

const LC = window.LightweightCharts;
const STEP = 300; // candle de 5 min
const TZ = -new Date().getTimezoneOffset() * 60; // o gráfico mostra o horário do aparelho
const toChart = (t) => t + TZ;
const POLL_MS = 10000;
const GAPS_MS = 60000;
const MORE_CANDLES = 1500;
const MAX_MARK_SPAN = 3000 * STEP;
const FLOW_BUCKETS = [1, 2, 5, 10, 15, 30, 60, 120, 300];
const FLOW_MAX_S = 6 * 3600;
const FLOW_MAX_BUCKETS = 10800;

const css = getComputedStyle(document.documentElement);
const V = (name) => css.getPropertyValue(name).trim();
const C = {
  bg: V('--bg'), line: V('--line'), text: V('--text'), muted: V('--muted'),
  up: '#26a69a', down: '#ef5350', vwap: V('--vwap'), ema9: V('--ema9'), ema21: V('--ema21'),
  tp: V('--tp'), sl: V('--sl'), time: V('--time'), flow: V('--flow'),
  gap: 'rgba(248, 81, 73, 0.17)', gapEdge: 'rgba(248, 81, 73, 0.55)', holdout: 'rgba(88, 166, 255, 0.07)',
};
const OUT_COLOR = { tp: C.tp, sl: C.sl, time: C.time };
const OUT_NAME = { tp: 'alvo', sl: 'stop', time: 'tempo' };
const SETUP_SHORT = { breakout: 'R', vwap_reversion: 'V', pullback: 'P' };
const SETUP_NAME = { breakout: 'rompimento', vwap_reversion: 'reversão VWAP', pullback: 'pullback' };
const SIDE_NAME = { long: 'compra', short: 'venda' };

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('wescalp.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('wescalp.' + k, JSON.stringify(v)); } catch { /* sem armazenamento */ } },
};

const S = {
  items: [], // {t,o,h,l,c,v,vwap,up1,dn1,up2,dn2,e9,e21,live}
  first: null, lastStored: null, more: true, liveFrom: null, forming: null,
  marks: new Map(), holdout: null, gaps: [], flow: null, flowKey: '',
  loadingMore: false,
  show: Object.assign({ vwap: true, bands: true, ema9: true, ema21: true, marks: true, flow: true }, store.get('show', {})),
  barriers: Object.assign({ tp: 1.5, sl: 1, bars: 12, holdout: false }, store.get('barriers', {})),
};

// --- formatação (pt-BR) ---
const nf = (d) => new Intl.NumberFormat('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
const NF = [0, 1, 2, 3, 4].map(nf);
const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? '—' : NF[d].format(x));
const fmtSigned = (x, d = 1) => (x > 0 ? '+' : '') + fmt(x, d);
const dt = new Intl.DateTimeFormat('pt-BR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
const dtSec = new Intl.DateTimeFormat('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const dDay = new Intl.DateTimeFormat('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric' });
const fmtTs = (t) => dt.format(new Date(t * 1000));
const dtYear = new Intl.DateTimeFormat('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
const fmtSec = (t) => dtSec.format(new Date(t * 1000));
const fmtDay = (t) => dDay.format(new Date(t * 1000));

async function api(path, params = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== '') q.set(k, v);
  const r = await fetch(`api/${path}${q.toString() ? '?' + q : ''}`, { cache: 'no-store' });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

// --- gráfico ---
const chartEl = $('chart');
const chart = LC.createChart(chartEl, {
  autoSize: true,
  layout: { background: { type: 'solid', color: C.bg }, textColor: C.muted, fontSize: 11 },
  grid: { vertLines: { color: '#161d26' }, horzLines: { color: '#161d26' } },
  rightPriceScale: { borderColor: C.line, scaleMargins: { top: 0.12, bottom: 0.06 } },
  timeScale: { borderColor: C.line, timeVisible: true, secondsVisible: false, rightOffset: 4, barSpacing: 7 },
  crosshair: { mode: LC.CrosshairMode.Normal },
  localization: { locale: 'pt-BR', dateFormat: 'dd/MM/yy', priceFormatter: (p) => NF[1].format(p) },
  kineticScroll: { touch: true, mouse: false },
});
const candleSeries = chart.addCandlestickSeries({
  upColor: C.up, downColor: C.down, borderVisible: false, wickUpColor: C.up, wickDownColor: C.down,
  priceFormat: { type: 'price', precision: 1, minMove: 0.1 },
});
const lineOpts = { priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
// Bandas não entram na escala automática: senão os candles ficam espremidos quando a VWAP do dia está longe.
const bandOpts = { ...lineOpts, lineWidth: 1, autoscaleInfoProvider: () => null };
const lines = {
  vwap: chart.addLineSeries({ ...lineOpts, color: C.vwap, lineWidth: 2 }),
  up1: chart.addLineSeries({ ...bandOpts, color: 'rgba(240,180,41,.55)', lineStyle: LC.LineStyle.Dashed }),
  dn1: chart.addLineSeries({ ...bandOpts, color: 'rgba(240,180,41,.55)', lineStyle: LC.LineStyle.Dashed }),
  up2: chart.addLineSeries({ ...bandOpts, color: 'rgba(240,180,41,.32)', lineStyle: LC.LineStyle.Dotted }),
  dn2: chart.addLineSeries({ ...bandOpts, color: 'rgba(240,180,41,.32)', lineStyle: LC.LineStyle.Dotted }),
  e9: chart.addLineSeries({ ...lineOpts, color: C.ema9, lineWidth: 1 }),
  e21: chart.addLineSeries({ ...lineOpts, color: C.ema21, lineWidth: 1 }),
};
const LINE_GROUP = { vwap: ['vwap'], bands: ['up1', 'dn1', 'up2', 'dn2'], ema9: ['e9'], ema21: ['e21'] };
const DAY_RESET = new Set(['vwap', 'up1', 'dn1', 'up2', 'dn2']); // VWAP recomeça às 00:00 UTC: sem traço ligando os dias

// Tempo bruto (época) -> índice lógico fracionário da série de candles. Um candle i ocupa [i - 0,5; i + 0,5].
function logicalOf(t) {
  const it = S.items;
  if (!it.length) return null;
  let lo = 0, hi = it.length - 1;
  if (t < it[0].t) return -0.5 - (it[0].t - t) / STEP;
  if (t >= it[hi].t + STEP) return hi + 0.5 + (t - it[hi].t - STEP) / STEP;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (it[mid].t <= t) lo = mid; else hi = mid - 1; }
  const frac = Math.min((t - it[lo].t) / STEP, 1);
  return lo - 0.5 + frac;
}
// logicalToCoordinate da 4.2 só aceita índice inteiro: a fração é somada à mão.
function xAtLogical(l) {
  const i = Math.floor(l);
  const x = chart.timeScale().logicalToCoordinate(i);
  return x === null ? null : x + (l - i) * barSpacing();
}
function xOf(t) { const l = logicalOf(t); return l === null ? null : xAtLogical(l); }
function timeAtLogical(l) {
  const it = S.items;
  if (!it.length) return null;
  const i = Math.max(0, Math.min(it.length - 1, Math.floor(l + 0.5)));
  return it[i].t + (l + 0.5 - i) * STEP;
}
function barSpacing() {
  const ts = chart.timeScale();
  const a = ts.logicalToCoordinate(0), b = ts.logicalToCoordinate(1);
  return a === null || b === null ? 6 : b - a;
}

// Faixas (buracos, período intocado) e linhas entrada -> saída, desenhadas dentro do gráfico.
class Overlay {
  constructor() {
    const self = this;
    this.views = [
      { zOrder: () => 'bottom', renderer: () => ({ draw: (target) => self.drawShades(target) }) },
      { zOrder: () => 'top', renderer: () => ({ draw: (target) => self.drawTrades(target) }) },
    ];
  }
  attached({ requestUpdate }) { this.requestUpdate = requestUpdate; }
  detached() { this.requestUpdate = null; }
  updateAllViews() {}
  paneViews() { return this.views; }
  update() { if (this.requestUpdate) this.requestUpdate(); }

  drawShades(target) {
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      const h = mediaSize.height, w = mediaSize.width;
      if (S.holdout && S.holdout.from) {
        const x = xOf(S.holdout.from);
        if (x !== null && x < w) {
          ctx.fillStyle = C.holdout;
          ctx.fillRect(Math.max(x, 0), 0, w - Math.max(x, 0), h);
          if (x >= 0) { // início do período intocado: linha e rótulo
            ctx.fillStyle = 'rgba(88, 166, 255, 0.7)';
            ctx.fillRect(x, 0, 1, h);
            ctx.font = '11px sans-serif';
            ctx.fillText('intocado →', x + 4, h - 8);
          }
        }
      }
      for (const [a, b] of S.gaps) {
        const x1 = xOf(a), x2 = xOf(b);
        if (x1 === null || x2 === null || x2 < 0 || x1 > w) continue;
        const width = Math.max(x2 - x1, 2);
        ctx.fillStyle = C.gap;
        ctx.fillRect(x1, 0, width, h);
        ctx.fillStyle = C.gapEdge;
        ctx.fillRect(x1, h - 3, width, 3);
      }
    });
  }

  drawTrades(target) {
    if (!S.show.marks || !S.marks.size) return;
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      const w = mediaSize.width;
      ctx.lineWidth = 1.2;
      for (const m of S.marks.values()) {
        const l1 = logicalOf(m.ts), l2 = logicalOf(m.exit_ts);
        if (l1 === null || l2 === null) continue;
        const x1 = xAtLogical(l1 + 1); // entrada no fechamento: borda direita do candle
        const x2 = xAtLogical(m.outcome === 'time' ? l2 + 1 : l2 + 0.5);
        if (x1 === null || x2 === null || x2 < -10 || x1 > w + 10) continue;
        const y1 = candleSeries.priceToCoordinate(m.entry_px), y2 = candleSeries.priceToCoordinate(m.exit_px);
        if (y1 === null || y2 === null) continue;
        const color = OUT_COLOR[m.outcome];
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.85;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        ctx.fillStyle = C.bg;
        ctx.beginPath(); ctx.arc(x1, y1, 2.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(x2, y2, 3.2, 0, 2 * Math.PI); ctx.fill();
      }
    });
  }
}
const overlay = new Overlay();
candleSeries.attachPrimitive(overlay);

// --- dados dos candles ---
function rowsToItems(body) {
  const out = [];
  body.candles.forEach((c, i) => {
    out.push({
      t: c[0], o: c[1], h: c[2], l: c[3], c: c[4], v: c[5],
      vwap: body.vwap[i], up1: body.vwap_up1[i], dn1: body.vwap_dn1[i], up2: body.vwap_up2[i], dn2: body.vwap_dn2[i],
      e9: body.ema9[i], e21: body.ema21[i],
      live: body.live_from !== null && c[0] >= body.live_from,
    });
  });
  return out;
}

function mergeCandles(body) {
  const fresh = rowsToItems(body);
  const keep = S.items.filter((x) => x.t < body.from || x.t >= body.to);
  S.items = keep.concat(fresh).sort((a, b) => a.t - b.t);
  if (body.first !== null) S.first = body.first;
  if (body.last_stored !== null) S.lastStored = body.last_stored;
  if (body.to >= body.now) { S.liveFrom = body.live_from; S.forming = body.forming; }
  if (S.items.length) S.more = body.first !== null && body.first < S.items[0].t;
}

function renderCandles() {
  const it = S.items;
  candleSeries.setData(it.map((x) => {
    const d = { time: toChart(x.t), open: x.o, high: x.h, low: x.l, close: x.c };
    if (x.t === S.forming) { // em formação: só contorno
      const col = x.c >= x.o ? C.up : C.down;
      Object.assign(d, { color: 'rgba(0,0,0,0)', borderColor: col, wickColor: col });
    } else if (x.live) { // montado de negócios, ainda sem candle gravado
      Object.assign(d, x.c >= x.o ? { color: 'rgba(38,166,154,.6)' } : { color: 'rgba(239,83,80,.6)' });
    }
    return d;
  }));
  for (const [key, series] of Object.entries(lines)) {
    let prevDay = null;
    series.setData(it.map((x) => {
      const day = Math.floor(x.t / 86400);
      const brk = DAY_RESET.has(key) && prevDay !== null && day !== prevDay;
      prevDay = day;
      const v = x[key];
      return v === null || v === undefined || brk ? { time: toChart(x.t) } : { time: toChart(x.t), value: v };
    }));
  }
  renderMarkers();
  overlay.update();
}

function applyVisibility() {
  for (const [group, keys] of Object.entries(LINE_GROUP)) {
    for (const k of keys) lines[k].applyOptions({ visible: !!S.show[group] });
  }
  $('flow-wrap').hidden = !S.show.flow;
  renderMarkers();
  overlay.update();
  updateNotice();
  if (S.show.flow) { resizeFlow(); scheduleFlow(); }
}

let markerText = null;
function renderMarkers(force = true) {
  const withText = barSpacing() >= 7;
  if (!force && withText === markerText) return;
  markerText = withText;
  if (!S.show.marks) { candleSeries.setMarkers([]); return; }
  const have = new Set(S.items.map((x) => x.t));
  const ms = [...S.marks.values()].filter((m) => have.has(m.ts)).sort((a, b) => a.ts - b.ts);
  candleSeries.setMarkers(ms.map((m) => ({
    time: toChart(m.ts),
    position: m.side === 'long' ? 'belowBar' : 'aboveBar',
    shape: m.side === 'long' ? 'arrowUp' : 'arrowDown',
    color: OUT_COLOR[m.outcome],
    text: withText ? SETUP_SHORT[m.setup] : '',
    size: 1,
  })));
}

// --- carga ---
const loadingEl = $('loading');
async function loadInitial() {
  loadingEl.hidden = false;
  try {
    const now = Math.floor(Date.now() / 1000);
    const n = window.innerWidth < 640 ? 400 : 700;
    const body = await api('candles', { source: 'futures', from: now - n * STEP });
    mergeCandles(body);
    renderCandles();
    chart.timeScale().scrollToRealTime();
    await Promise.all([loadMarks(S.items.length ? S.items[0].t : null, null, true), loadGaps()]);
  } catch (e) {
    loadingEl.textContent = 'erro ao carregar: ' + e.message;
    return;
  }
  loadingEl.hidden = true;
}

async function loadMore() {
  if (S.loadingMore || !S.more || !S.items.length) return;
  S.loadingMore = true;
  loadingEl.hidden = false;
  loadingEl.textContent = 'carregando candles anteriores…';
  try {
    const to = S.items[0].t;
    const body = await api('candles', { source: 'futures', from: to - MORE_CANDLES * STEP, to });
    const before = S.items.length;
    const range = chart.timeScale().getVisibleLogicalRange();
    mergeCandles(body);
    const added = S.items.length - before;
    renderCandles();
    if (range && added > 0) chart.timeScale().setVisibleLogicalRange({ from: range.from + added, to: range.to + added });
    if (body.candles.length) await loadMarks(body.from, to, false);
  } catch (e) {
    console.warn('candles anteriores:', e);
  } finally {
    S.loadingMore = false;
    loadingEl.hidden = true;
    loadingEl.textContent = 'carregando…';
  }
}

// Marcas de [from, to) em pedaços de até 3.000 candles. reset = troca todas (barreiras novas).
async function loadMarks(from, to, reset) {
  if (from === null) return;
  const b = S.barriers;
  const end = to ?? (S.lastStored !== null ? S.lastStored + STEP : Math.floor(Date.now() / 1000));
  const got = [];
  let holdout = S.holdout;
  for (let a = from; a < end; a += MAX_MARK_SPAN) {
    const body = await api('marks', { from: a, to: Math.min(a + MAX_MARK_SPAN, end), tp: b.tp, sl: b.sl, bars: b.bars, holdout: b.holdout ? 1 : '' });
    got.push(...body.marks);
    holdout = body.holdout;
  }
  if (reset) S.marks = new Map();
  else for (const [k, m] of S.marks) if (m.ts >= from && m.ts < end) S.marks.delete(k);
  for (const m of got) S.marks.set(`${m.ts}|${m.setup}|${m.side}`, m);
  S.holdout = holdout;
  renderMarkers();
  overlay.update();
  updateNotice();
}

async function loadGaps() {
  try {
    const body = await api('gaps', { hours: 48 });
    S.gaps = body.gaps;
    overlay.update();
  } catch (e) { console.warn('buracos:', e); }
}

async function refreshTail() {
  if (!S.items.length) return loadInitial();
  const from = Math.max(S.lastStored ?? S.items[S.items.length - 1].t, S.items[S.items.length - 1].t - 2 * 3600);
  const prevStored = S.lastStored;
  const body = await api('candles', { source: 'futures', from });
  mergeCandles(body);
  renderCandles();
  if (S.lastStored !== prevStored && S.show.marks) await loadMarks(S.lastStored - 86400, null, false);
}

function updateNotice() {
  const el = $('notice');
  const h = S.holdout;
  if (!h || !h.from || !S.show.marks) { el.hidden = true; return; }
  el.hidden = false;
  el.className = 'notice' + (h.included ? ' warn' : '');
  el.textContent = h.included
    ? `Mostrando marcas no período intocado (desde ${fmtDay(h.from)}). Olhar aqui gasta o período de teste da pesquisa. `
    : `Período intocado da pesquisa desde ${fmtDay(h.from)} (últimas ${h.weeks} semanas): sem marcas, fundo azulado. `;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = h.included ? 'Ocultar' : 'Mostrar mesmo assim';
  btn.onclick = () => { $('b-holdout').checked = !h.included; applyBarriers(); };
  el.appendChild(btn);
}

// --- legenda ---
const legendEl = $('legend');
function legendFor(x) {
  if (!x) return '';
  const chg = x.c - x.o, cls = chg >= 0 ? 'up' : 'down';
  let tag = '';
  if (x.t === S.forming) tag = ' <span class="dim">em formação</span>';
  else if (x.live) tag = ' <span class="dim">de negócios</span>';
  let html = `<div class="row"><span>${fmtTs(x.t)}${tag}</span>`
    + `<span>A <b class="${cls}">${fmt(x.o)}</b></span><span>M <b class="${cls}">${fmt(x.h)}</b></span>`
    + `<span>m <b class="${cls}">${fmt(x.l)}</b></span><span>F <b class="${cls}">${fmt(x.c)}</b></span>`
    + `<span class="${cls}">${fmtSigned(chg / x.o * 1e4, 1)} bps</span></div>`;
  const parts = [];
  if (S.show.vwap && x.vwap !== null) parts.push(`<span style="color:${C.vwap}">VWAP ${fmt(x.vwap)}</span>`);
  if (S.show.ema9 && x.e9 !== null) parts.push(`<span style="color:${C.ema9}">EMA9 ${fmt(x.e9)}</span>`);
  if (S.show.ema21 && x.e21 !== null) parts.push(`<span style="color:${C.ema21}">EMA21 ${fmt(x.e21)}</span>`);
  if (parts.length) html += `<div class="row">${parts.join('')}</div>`;
  if (S.show.marks) {
    for (const m of S.marks.values()) {
      if (m.ts !== x.t && m.exit_ts !== x.t) continue;
      const verb = m.ts === x.t ? 'entra' : 'sai';
      html += `<div class="row mk" style="color:${OUT_COLOR[m.outcome]}">${SETUP_NAME[m.setup]} · ${SIDE_NAME[m.side]} · ${verb}`
        + ` ${fmt(m.entry_px)} → ${fmt(m.exit_px)} (${OUT_NAME[m.outcome]}, ${m.bars} candles)`
        + ` · líq. ${fmtSigned(m.ret_net_bps, 1)} bps</div>`;
    }
  }
  return html;
}
function itemAtLogical(l) {
  if (l === null || !S.items.length) return null;
  const i = Math.round(l);
  return i >= 0 && i < S.items.length ? S.items[i] : null;
}
chart.subscribeCrosshairMove((p) => {
  if (!p || !p.point || p.time === undefined) {
    legendEl.innerHTML = legendFor(S.items[S.items.length - 1]);
    flowCursor = null;
  } else {
    const l = chart.timeScale().coordinateToLogical(p.point.x);
    legendEl.innerHTML = legendFor(itemAtLogical(l));
    flowCursor = l === null ? null : timeAtLogical(l);
  }
  drawFlow();
});

// --- painel de fluxo ---
const flowCanvas = $('flow');
const flowInfo = $('flow-info');
let flowCursor = null;
function resizeFlow() {
  const dpr = window.devicePixelRatio || 1;
  const r = flowCanvas.getBoundingClientRect();
  flowCanvas.width = Math.round(r.width * dpr);
  flowCanvas.height = Math.round(r.height * dpr);
  legendEl.style.right = (chart.priceScale('right').width() + 8) + 'px'; // legenda não invade a escala
  drawFlow();
}

function robustMax(vals) {
  const a = vals.filter((v) => v !== null).map(Math.abs).sort((x, y) => x - y);
  if (!a.length) return 1;
  return a[Math.min(a.length - 1, Math.floor(a.length * 0.98))] || a[a.length - 1] || 1;
}

function drawFlow() {
  if (!S.show.flow) return;
  const dpr = window.devicePixelRatio || 1;
  const ctx = flowCanvas.getContext('2d');
  const W = flowCanvas.width / dpr, H = flowCanvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const plotW = W - chart.priceScale('right').width();
  const rows = [
    { key: 'ofi', name: 'OFI', y: 0, h: H * 0.40 },
    { key: 'delta', name: 'delta', y: H * 0.40, h: H * 0.25 },
    { key: 'imb5', name: 'deseq. 5 níveis', y: H * 0.65, h: H * 0.35 },
  ];
  ctx.font = '10px ' + V('--mono');
  for (const r of rows) {
    ctx.strokeStyle = C.line;
    ctx.beginPath(); ctx.moveTo(0, r.y + r.h / 2 + 0.5); ctx.lineTo(plotW, r.y + r.h / 2 + 0.5); ctx.stroke();
    if (r.y > 0) { ctx.strokeStyle = '#1e2630'; ctx.beginPath(); ctx.moveTo(0, r.y + 0.5); ctx.lineTo(W, r.y + 0.5); ctx.stroke(); }
  }
  const f = S.flow;
  const visible = [];
  if (f && f.t.length) {
    const bw = Math.max(1, (f.bucket / STEP) * barSpacing() * 0.8);
    for (let i = 0; i < f.t.length; i++) {
      const x = xOf(f.t[i] - f.bucket / 2);
      if (x === null || x < -bw || x > plotW + bw) continue;
      visible.push([i, x]);
    }
    const scale = {
      ofi: robustMax(visible.map(([i]) => f.ofi[i])),
      delta: robustMax(visible.map(([i]) => f.delta[i])),
    };
    for (const r of rows.slice(0, 2)) {
      const mid = r.y + r.h / 2, half = r.h / 2 - 3;
      for (const [i, x] of visible) {
        const v = f[r.key][i];
        if (v === null) continue;
        const hh = Math.min(Math.abs(v) / scale[r.key], 1) * half;
        ctx.fillStyle = v >= 0 ? 'rgba(63,185,80,.85)' : 'rgba(248,81,73,.85)';
        ctx.fillRect(x - bw / 2, v >= 0 ? mid - hh : mid, bw, Math.max(hh, 0.5));
      }
      ctx.fillStyle = C.muted;
      ctx.textAlign = 'right';
      ctx.fillText(fmt(scale[r.key], 0), W - 4, r.y + 11);
    }
    const r = rows[2], mid = r.y + r.h / 2, half = r.h / 2 - 3;
    ctx.strokeStyle = C.flow;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    let pen = false;
    for (const [i, x] of visible) {
      const v = f.imb5[i];
      if (v === null) { pen = false; continue; }
      const y = mid - v * half;
      if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    // Pontos: balde isolado entre buracos não vira linha; com baldes largos, cada ponto aparece.
    ctx.fillStyle = C.flow;
    visible.forEach(([i, x], k) => {
      const v = f.imb5[i];
      if (v === null) return;
      const prev = k > 0 ? f.imb5[visible[k - 1][0]] : null, next = k < visible.length - 1 ? f.imb5[visible[k + 1][0]] : null;
      if (bw >= 4 || (prev === null && next === null)) { ctx.beginPath(); ctx.arc(x, mid - v * half, 2, 0, 2 * Math.PI); ctx.fill(); }
    });
    ctx.fillStyle = C.muted;
    ctx.textAlign = 'right';
    ctx.fillText('+1', W - 4, r.y + 11);
    ctx.fillText('−1', W - 4, r.y + r.h - 3);
  }
  ctx.textAlign = 'left';
  ctx.fillStyle = C.muted;
  for (const r of rows) ctx.fillText(r.name, 4, r.y + 11);

  let info = '';
  if (!f) info = 'carregando…';
  else if (!visible.length) info = 'sem livro gravado nesta janela';
  else info = `baldes de ${f.bucket} s · ${fmtTs(f.t[visible[0][0]] - f.bucket)}–${fmtSec(f.t[visible[visible.length - 1][0]])}`
    + (f.clipped ? ` · só as últimas ${fmt(f.max_hours, 0)} h da janela` : '');
  if (f && flowCursor !== null && visible.length) {
    let best = null;
    for (const [i] of visible) if (f.t[i] - f.bucket <= flowCursor && flowCursor < f.t[i]) { best = i; break; }
    if (best !== null) {
      const x = xOf(f.t[best] - f.bucket / 2);
      ctx.strokeStyle = 'rgba(214,221,229,.35)';
      ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); ctx.stroke();
      info = `${fmtSec(f.t[best] - f.bucket)} · OFI ${fmt(f.ofi[best], 1)} · delta ${fmt(f.delta[best], 0)}`
        + ` · deseq. ${fmt(f.imb5[best], 2)} · micro ${fmt(f.micro_bps[best], 2)} bps · mid ${fmt(f.mid[best], 1)}`;
    }
  }
  flowInfo.textContent = info;
}

let flowTimer = null;
function scheduleFlow() {
  clearTimeout(flowTimer);
  flowTimer = setTimeout(() => loadFlow(false), 450);
}
async function loadFlow(force) {
  if (!S.show.flow || !S.items.length) return;
  const r = chart.timeScale().getVisibleLogicalRange();
  if (!r) return;
  const last = S.items[S.items.length - 1].t + STEP;
  const now = Math.floor(Date.now() / 1000);
  let to = Math.min(timeAtLogical(r.to), last, now);
  const from0 = Math.max(timeAtLogical(r.from), S.items[0].t);
  const pxPerSec = barSpacing() / STEP;
  const bucket = FLOW_BUCKETS.find((b) => b * pxPerSec >= 3) || 300;
  const span = Math.min(FLOW_MAX_S, bucket * FLOW_MAX_BUCKETS);
  const live = to >= now - 2 * STEP;
  if (live) to = now;
  const from = Math.max(from0, to - span);
  if (to <= from) return;
  // Já tem o pedaço com o mesmo balde e ele não está mais andando: não pede de novo.
  const f = S.flow;
  if (!force && f && f.bucket === bucket && f.from <= from + bucket && f.to >= to - bucket && !live) return;
  const key = `${Math.floor(from / bucket)}|${Math.floor(to / bucket)}|${bucket}`;
  if (!force && key === S.flowKey) return;
  S.flowKey = key;
  try {
    S.flow = await api('flow', { from: Math.floor(from), to: Math.ceil(to), bucket });
  } catch (e) {
    console.warn('fluxo:', e);
    flowInfo.textContent = 'erro no fluxo: ' + e.message;
    return;
  }
  drawFlow();
}

chart.timeScale().subscribeVisibleLogicalRangeChange((r) => {
  if (!r) return;
  if (r.from < 40) loadMore();
  renderMarkers(false);
  drawFlow();
  scheduleFlow();
});
chart.timeScale().subscribeSizeChange(() => { resizeFlow(); });
window.addEventListener('resize', () => resizeFlow());

// --- barra de estado ---
const AGE_LIMITS = { book: [15, 60], trade: [120, 600] }; // iguais aos padrões de wescalp.health
function ageClass(s, kind) {
  if (s === null || s === undefined) return 'bad';
  const [late, stale] = AGE_LIMITS[kind];
  return s > stale ? 'bad' : s > late ? 'warn' : '';
}
function setKV(id, text, cls = '') { const el = $(id); el.textContent = text ?? '—'; el.className = cls; }
async function loadHealth() {
  let h;
  try { h = await api('health'); } catch (e) { h = { state: 'parado', problems: ['painel sem resposta: ' + e.message], details: [] }; }
  const pill = $('st-state');
  pill.className = 'pill ' + h.state;
  pill.querySelector('b').textContent = h.state;
  const t = h.text || {};
  setKV('st-book', t.book_age ?? '—', ageClass(h.book_age_s, 'book'));
  setKV('st-trade', t.trade_age ?? '—', ageClass(h.trade_age_s, 'trade'));
  setKV('st-gap', h.gap_minutes_24h === undefined ? '—' : `${h.gap_minutes_24h} min`, h.gap_minutes_24h > 15 ? 'warn' : '');
  setKV('st-db', t.db ?? '—');
  setKV('st-disk', t.disk_free ?? '—', h.disk_free_bytes !== null && h.disk_free_bytes < 5 * 1024 ** 3 ? 'warn' : '');
  const box = $('st-details');
  box.replaceChildren();
  const add = (tag, cls, items) => {
    if (!items || !items.length) return;
    const ul = document.createElement(tag);
    ul.className = cls;
    for (const s of items) { const li = document.createElement('li'); li.textContent = s; ul.appendChild(li); }
    box.appendChild(ul);
  };
  const head = document.createElement('div');
  head.textContent = h.headline || '';
  box.appendChild(head);
  add('ul', 'problems ' + h.state, h.problems);
  add('ul', 'details', h.details);
}
$('st-state').addEventListener('click', () => {
  const box = $('st-details');
  box.hidden = !box.hidden;
  $('st-state').setAttribute('aria-expanded', String(!box.hidden));
});

// --- controles ---
document.querySelectorAll('[data-show]').forEach((el) => {
  el.checked = !!S.show[el.dataset.show];
  el.addEventListener('change', () => {
    S.show[el.dataset.show] = el.checked;
    store.set('show', S.show);
    applyVisibility();
  });
});
$('b-tp').value = S.barriers.tp;
$('b-sl').value = S.barriers.sl;
$('b-bars').value = S.barriers.bars;
$('b-holdout').checked = !!S.barriers.holdout;
async function applyBarriers() {
  const num = (id, d) => { const v = parseFloat($(id).value); return Number.isFinite(v) && v > 0 ? v : d; };
  S.barriers = { tp: num('b-tp', 1.5), sl: num('b-sl', 1), bars: Math.round(num('b-bars', 12)), holdout: $('b-holdout').checked };
  store.set('barriers', S.barriers);
  $('b-apply').closest('details').open = false;
  if (!S.items.length) return;
  loadingEl.hidden = false;
  try { await loadMarks(S.items[0].t, null, true); } catch (e) { console.warn('marcas:', e); }
  loadingEl.hidden = true;
}
$('b-apply').addEventListener('click', applyBarriers);
$('b-holdout').addEventListener('change', applyBarriers);
$('go-now').addEventListener('click', () => chart.timeScale().scrollToRealTime());

// --- abas e pesquisa ---
let reportsLoaded = false;
const TAB_HASH = { chart: '', strategy: '#estrategia', research: '#pesquisa' };
document.querySelectorAll('[data-tab]').forEach((btn) => btn.addEventListener('click', () => {
  const tab = btn.dataset.tab;
  document.querySelectorAll('[data-tab]').forEach((b) => b.setAttribute('aria-selected', String(b === btn)));
  $('tab-chart').hidden = tab !== 'chart';
  $('tab-strategy').hidden = tab !== 'strategy';
  $('tab-research').hidden = tab !== 'research';
  if (tab === 'research' && !reportsLoaded) loadReports();
  if (tab === 'strategy') loadStrategy();
  if (tab === 'chart') resizeFlow();
  try { history.replaceState(null, '', TAB_HASH[tab] || location.pathname); } catch { /* ignora */ }
}));
$('reports-reload').addEventListener('click', loadReports);

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else e.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined) e.append(k);
  return e;
}
function fmtWhen(v) {
  if (v === null || v === undefined) return '—';
  const d = new Date(typeof v === 'number' && v < 1e12 ? v * 1000 : v);
  return Number.isNaN(d.getTime()) ? String(v) : dtYear.format(d);
}
function fmtCell(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? fmt(v, 0) : fmt(v, Math.abs(v) >= 100 ? 1 : 2);
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function researchTable(rows) {
  const pct = (x) => (x === null || x === undefined ? '—' : fmt(x * 100, 1) + '%');
  const cols = [
    ['setup', 'setup', (r) => SETUP_NAME[r.setup] || r.setup, 'l'],
    ['lado', 'side', (r) => SIDE_NAME[r.side] || r.side, 'l'],
    ['entradas', 'n', (r) => fmt(r.n, 0)],
    ['alvo', 'tp', (r) => pct(r.tp)],
    ['ganha', 'win', (r) => pct(r.win)],
    ['bruto (bps)', 'gross_bps', (r) => fmt(r.gross_bps, 2)],
    ['líquido (bps)', 'net_bps', (r) => fmt(r.net_bps, 2)],
    ['IC 90% (bps)', 'lo_bps', (r) => `[${fmt(r.lo_bps, 1)}; ${fmt(r.hi_bps, 1)}]`],
    ['janelas +', 'windows_pos', (r) => `${fmt(r.windows_pos, 0)}/${fmt(r.windows, 0)}`],
  ];
  const head = el('tr', {}, ...cols.map(([name, , , cls]) => el('th', cls ? { class: cls } : {}, name)));
  const body = rows.map((r) => el('tr', {}, ...cols.map(([, key, f, cls]) => {
    let c = cls || '';
    if (key === 'gross_bps' || key === 'net_bps') c += r[key] > 0 ? ' pos' : r[key] < 0 ? ' neg' : '';
    return el('td', c ? { class: c.trim() } : {}, f(r));
  })));
  return el('div', { class: 'scroll' }, el('table', {}, el('thead', {}, head), el('tbody', {}, ...body)));
}

function genericBlock(v) {
  if (v === null || v === undefined) return el('span', { class: 'muted' }, '—');
  if (typeof v === 'string') return v.includes('\n') ? el('pre', {}, v) : el('span', {}, v);
  if (typeof v !== 'object') return el('span', {}, fmtCell(v));
  if (Array.isArray(v)) {
    if (v.length && v.every((x) => x && typeof x === 'object' && !Array.isArray(x))) {
      const keys = [...new Set(v.flatMap((x) => Object.keys(x)))];
      return el('div', { class: 'scroll' }, el('table', {},
        el('thead', {}, el('tr', {}, ...keys.map((k, i) => el('th', i ? {} : { class: 'l' }, k)))),
        el('tbody', {}, ...v.map((x) => el('tr', {}, ...keys.map((k) => el('td', {}, fmtCell(x[k]))))))));
    }
    if (v.every((x) => typeof x === 'string')) return el('pre', {}, v.join('\n'));
    return el('pre', {}, JSON.stringify(v, null, 2));
  }
  const dl = el('dl', { class: 'kvs' });
  const blocks = [];
  for (const [k, x] of Object.entries(v)) {
    if (x !== null && typeof x === 'object') blocks.push(el('h4', {}, k), genericBlock(x));
    else if (typeof x === 'string' && x.includes('\n')) blocks.push(el('h4', {}, k), el('pre', {}, x));
    else dl.append(el('dt', {}, k), el('dd', {}, fmtCell(x)));
  }
  return el('div', {}, dl.children.length ? dl : null, ...blocks);
}

function metaLine(b, extra) {
  const parts = [];
  if (b.symbol) parts.push(b.symbol);
  if (b.from || b.to) parts.push(`${fmtWhen(b.from)} a ${fmtWhen(b.to)}`);
  return parts.concat(extra.filter(Boolean)).join(' · ');
}

function renderReport(r) {
  const b = r.body || {};
  const card = el('article', { class: 'report' });
  const titles = { research: 'Setups contra a tripla barreira', sweep: 'Validação: grade, walk-forward, DSR e PBO' };
  card.append(el('h3', {}, titles[r.name] || r.name));
  if (r.name === 'research' && Array.isArray(b.rows)) {
    const bar = b.barriers && typeof b.barriers === 'object' ? JSON.stringify(b.barriers) : b.barriers;
    card.append(el('div', { class: 'meta' }, metaLine(b, [
      b.candles !== undefined ? `${fmt(b.candles, 0)} candles` : '', bar ? `barreiras ${bar}` : '',
      b.holdout_weeks !== undefined ? `sem as últimas ${b.holdout_weeks} semanas` : '', `gravado ${fmtWhen(r.created_at)}`,
    ])));
    card.append(researchTable(b.rows));
  } else if (r.name === 'sweep' && Array.isArray(b.sections)) {
    card.append(el('div', { class: 'meta' }, metaLine(b, [
      b.configs !== undefined ? `${fmt(b.configs, 0)} configurações` : '', `gravado ${fmtWhen(r.created_at)}`,
    ])));
    if (Array.isArray(b.header) && b.header.length) card.append(el('pre', {}, b.header.join('\n')));
    for (const s of b.sections) card.append(el('h4', {}, s.title || ''), el('pre', {}, s.text || ''));
  } else {
    card.append(el('div', { class: 'meta' }, `gravado ${fmtWhen(r.created_at)}`), genericBlock(b));
  }
  return card;
}

async function loadReports() {
  const box = $('reports');
  try {
    const { reports } = await api('reports');
    reportsLoaded = true;
    const order = { research: 0, sweep: 1 };
    reports.sort((a, b) => (order[a.name] ?? 9) - (order[b.name] ?? 9) || a.name.localeCompare(b.name));
    box.replaceChildren(...(reports.length ? reports.map(renderReport)
      : [el('p', { class: 'muted' }, 'Nenhum relatório salvo ainda. Os resultados de wescalp.research e wescalp.sweep aparecem aqui quando forem gravados na tabela reports.')]));
  } catch (e) {
    box.replaceChildren(el('p', { class: 'muted' }, 'Erro ao ler os relatórios: ' + e.message));
  }
}

// --- estratégia: carteiras simuladas de wescalp.paper (nenhuma ordem real; só o registro das regras no tempo) ---
const FROZEN_ID = 'media-50-100-150';
const SG = {
  data: null, last: 0, fitted: false, byDay: new Map(),
  strat: store.get('sgStrat', FROZEN_ID), range: store.get('sgRange', 'live'),
  sim: store.get('sgSim', true), sma: store.get('sgSma', true),
};
const SG_EVERY_MS = 60000;
const SMA_COLOR = ['#4fc3f7', '#f0b429', '#c792ea'];
const STRAT_COLOR = { [FROZEN_ID]: '#3fb950', 'aprendiz-50-170': '#d2a8ff' };
const STRAT_CLS = { [FROZEN_ID]: 'c-frozen', 'aprendiz-50-170': 'c-learner' };
const ymd = (t) => new Date(t * 1000).toISOString().slice(0, 10); // dia UTC: o gráfico diário fica em UTC
const dUTC = new Intl.DateTimeFormat('pt-BR', { timeZone: 'UTC', day: '2-digit', month: '2-digit', year: 'numeric' });
const fmtUTCDay = (t) => dUTC.format(new Date(t * 1000));
const pctS = (x, d = 1) => (x === null || x === undefined ? '—' : fmtSigned(x * 100, d) + '%');
const pctU = (x, d = 1) => (x === null || x === undefined ? '—' : fmt(x * 100, d) + '%');
const sign = (x) => (x > 0 ? 'pos' : x < 0 ? 'neg' : '');
const keyOf = (t) => (typeof t === 'string' ? t : typeof t === 'object' && t
  ? `${t.year}-${String(t.month).padStart(2, '0')}-${String(t.day).padStart(2, '0')}` : ymd(t));
const sgAll = () => (SG.data ? SG.data.strategies : []);
const sgSel = () => sgAll().find((x) => x.id === SG.strat) || sgAll()[0];

function dayChart(container, priceFormatter) {
  return LC.createChart(container, {
    autoSize: true,
    layout: { background: { type: 'solid', color: C.bg }, textColor: C.muted, fontSize: 11 },
    grid: { vertLines: { color: '#161d26' }, horzLines: { color: '#161d26' } },
    rightPriceScale: { borderColor: C.line, scaleMargins: { top: 0.14, bottom: 0.06 } },
    timeScale: { borderColor: C.line, timeVisible: false, rightOffset: 3, barSpacing: 5 },
    crosshair: { mode: LC.CrosshairMode.Normal },
    localization: { locale: 'pt-BR', dateFormat: 'dd/MM/yy', priceFormatter },
    kineticScroll: { touch: true, mouse: false },
  });
}

function sgCharts() {
  if (SG.chart) return;
  SG.chart = dayChart($('sg-chart'), (p) => NF[0].format(p));
  SG.candles = SG.chart.addCandlestickSeries({
    upColor: C.up, downColor: C.down, borderVisible: false, wickUpColor: C.up, wickDownColor: C.down,
  });
  SG.smas = SMA_COLOR.map((color) => SG.chart.addLineSeries({ ...lineOpts, color, lineWidth: 1 }));
  SG.eqChart = dayChart($('sg-equity'), (p) => fmtSigned(p, 1) + '%');
  SG.eqLines = {};
  for (const id of Object.keys(STRAT_COLOR)) {
    SG.eqLines[id] = SG.eqChart.addLineSeries({
      color: STRAT_COLOR[id], lineWidth: id === SG.strat ? 2 : 1, priceLineVisible: false, lastValueVisible: true,
    });
  }
  SG.hold = SG.eqChart.addLineSeries({ color: C.muted, lineWidth: 1, priceLineVisible: false, lastValueVisible: true });
  SG.chart.subscribeCrosshairMove((p) => sgLegend(p && p.time ? keyOf(p.time) : null));
  SG.eqChart.subscribeCrosshairMove((p) => sgEqLegend(p && p.time ? keyOf(p.time) : null));
}

// Dia-base da curva: o primeiro dia ao vivo, ou o primeiro dia em que as duas carteiras têm todas as médias.
function sgBaseT() {
  const list = sgAll().filter((x) => x.days.length);
  if (SG.range === 'live') {
    const t = list.map((x) => (x.days.find((d) => d.live) || {}).t);
    if (t.length && t.every(Boolean)) return Math.max(...t);
  }
  return Math.max(...list.map((x) => (x.history ? x.history.from : x.days[0].t)));
}

function sgCurve(x, baseT) {
  const base = x.days.find((d) => d.t >= baseT);
  if (!base) return { base: null, eq: [], hold: [] };
  const pts = x.days.filter((d) => d.t >= base.t);
  const tail = x.now && ymd(x.now.t - 300) > ymd(x.days[x.days.length - 1].t) ? ymd(x.now.t - 300) : null;
  const eq = pts.map((d) => ({ time: ymd(d.t), value: (d.eq / base.eq - 1) * 100 }));
  const hold = pts.map((d) => ({ time: ymd(d.t), value: (d.hold / base.hold - 1) * 100 }));
  if (tail) {
    eq.push({ time: tail, value: (x.now.equity / base.eq - 1) * 100 });
    hold.push({ time: tail, value: (x.now.hold / base.hold - 1) * 100 });
  }
  return { base, eq, hold };
}

function sgLegend(key) {
  const x = sgSel();
  const d = (key && SG.byDay.get(key)) || x.days[x.days.length - 1];
  const smas = x.chart_smas.map((n, i) => `<span style="color:${SMA_COLOR[i]}">M${n} ${d.sma[i] === null ? '—' : fmt(d.sma[i], 0)}</span>`);
  const o = x.orders.find((q) => q.t === d.t);
  $('sg-legend').innerHTML = `<div class="row"><span>${fmtUTCDay(d.t)}</span><span>fech. ${fmt(d.c, 0)}</span>${smas.join('')}</div>`
    + `<div class="row dim"><span>${x.name}: decisão no fechamento ${fmt(d.pos * 100, 0)}% comprado</span>`
    + `<span>${d.live ? 'ao vivo' : 'simulado'}</span></div>`
    + (o ? `<div class="row"><span class="${o.action === 'compra' ? 'up' : 'down'}">${o.action} de ${fmt(o.size * 100, 0)}%`
      + ` → ${fmt(o.position * 100, 0)}% comprado (${o.reason})</span></div>` : '');
}

function sgEqLegend(key) {
  const baseT = sgBaseT();
  const parts = [];
  let hold = null;
  for (const x of sgAll()) {
    const c = sgCurve(x, baseT);
    if (!c.base) continue;
    const at = key ? c.eq.find((p) => p.time === key) : c.eq[c.eq.length - 1];
    const h = key ? c.hold.find((p) => p.time === key) : c.hold[c.hold.length - 1];
    if (h && hold === null) hold = h.value;
    if (at) parts.push(`<span class="${STRAT_CLS[x.id]}">${x.name} ${fmtSigned(at.value, 1)}%</span>`);
  }
  $('sg-eq-legend').innerHTML = `<div class="row"><span>${key ? fmtUTCDay(Date.parse(key + 'T00:00:00Z') / 1000) : 'agora'}</span>`
    + parts.join('') + `<span class="dim">segurar BTC ${hold === null ? '—' : fmtSigned(hold, 1) + '%'}</span></div>`
    + `<div class="row dim"><span>desde ${fmtUTCDay(baseT)}</span></div>`;
}

function card(label, big, bigCls, ...subs) {
  return el('div', { class: 'card' }, el('small', {}, label), el('div', { class: 'big ' + (bigCls || '') }, big),
    ...subs.filter(Boolean).map((x) => (typeof x === 'string' ? el('div', { class: 'sub' }, x) : x)));
}

function sgCards() {
  const x = sgSel();
  const other = sgAll().find((y) => y.id !== x.id);
  const end = x.days[x.days.length - 1];
  const pos = x.now ? x.now.position : end.pos;
  const on = end.on.reduce((a, c) => a + c, 0);
  const sub = x.learn
    ? `${on} de ${end.on.length} médias abaixo do preço; elas pesam ${pctU(end.w.reduce((a, w, i) => a + w * end.on[i], 0), 0)}`
    : `preço acima de ${on} de ${end.on.length} médias`;
  const comps = x.learn ? null : el('div', { class: 'comps' }, ...end.on.map((c, i) => el('span', { class: c ? 'on' : '' },
    `${c ? '▲' : '▼'} M${x.lookbacks[i]}`)));
  const nowS = Date.now() / 1000;
  const next = (Math.floor(nowS / 86400) + 1) * 86400;
  const left = Math.max(0, next - nowS);
  const lastOrder = x.orders.length ? x.orders[x.orders.length - 1] : null;
  const cards = [card(`${x.name}: posição agora`, `${fmt(pos * 100, 0)}% comprado`, '', sub, comps)];
  if (x.live) {
    cards.push(
      card(`ao vivo desde ${fmtWhen(x.live.from)}`, pctS(x.live.ret, 2), sign(x.live.ret),
        `segurar BTC ${pctS(x.live.hold_ret, 2)}`, other && other.live ? `${other.name}: ${pctS(other.live.ret, 2)}` : '',
        `${fmt(x.live.days, 0)} dia(s) fechado(s) ao vivo`),
      card('maior queda ao vivo', pctU(x.live.dd), x.live.dd > 0 ? 'neg' : '', `segurar BTC ${pctU(x.live.hold_dd)}`),
      card('ordens ao vivo', fmt(x.live.orders, 0), '', lastOrder
        ? `última: ${lastOrder.action} de ${fmt(lastOrder.size * 100, 0)}% em ${fmtUTCDay(lastOrder.t)} (${lastOrder.reason})` : 'nenhuma ainda'),
    );
  } else {
    const start = x.live_start ? Date.parse(x.live_start + 'T00:00:00Z') / 1000 + 86400 : null;
    cards.push(card('período ao vivo', 'começa em breve', '', start ? `primeira decisão ao vivo: ${fmtWhen(start)}` : '—'));
  }
  cards.push(card('próxima decisão', fmtWhen(next), '', `em ${Math.floor(left / 3600)} h ${Math.floor((left % 3600) / 60)} min`,
    'fechamento das 00h UTC'));
  if (x.history) {
    cards.push(card(`histórico simulado ${fmtUTCDay(x.history.from)} a ${fmtUTCDay(x.history.to)}`, pctS(x.history.ret, 0),
      sign(x.history.ret), `segurar BTC ${pctS(x.history.hold_ret, 0)}`,
      `maior queda ${pctU(x.history.dd, 0)} (segurar ${pctU(x.history.hold_dd, 0)}) · ${fmt(x.history.orders, 0)} ordens`));
  }
  $('sg-cards').replaceChildren(...cards);
}

function sgOrders() {
  const x = sgSel();
  const rows = [...x.orders].reverse().map((o) => el('tr', {},
    el('td', {}, fmtUTCDay(o.t)),
    el('td', { class: 'l ' + (o.action === 'compra' ? 'pos' : 'neg') }, `${o.action} de ${fmt(o.size * 100, 0)}%`),
    el('td', {}, `${fmt(o.position * 100, 0)}%`), el('td', { class: 'l' }, o.reason), el('td', {}, fmt(o.price, 0)),
    el('td', { class: 'l ' + (o.live ? 'live' : 'muted') }, o.live ? 'ao vivo' : 'simulado')));
  const heads = [['dia da decisão'], ['ordem', 'l'], ['comprado depois'], ['motivo', 'l'], ['fechamento'], ['', 'l']];
  const head = el('tr', {}, ...heads.map(([h, c]) => el('th', c ? { class: c } : {}, h)));
  $('sg-signals').replaceChildren(rows.length
    ? el('div', { class: 'scroll' }, el('table', {}, el('thead', {}, head), el('tbody', {}, ...rows)))
    : el('p', { class: 'muted' }, 'Nenhuma ordem ainda.'));
}

function sgWeights() {
  const x = sgSel();
  $('sg-w-title').hidden = $('sg-weights').hidden = !x.learn;
  if (!x.learn) return;
  const top = Math.max(...x.components.map((c) => c.weight), 1e-9);
  const rows = x.components.map((c) => el('tr', {},
    el('td', { class: 'l' }, `média ${c.n}`),
    el('td', { class: 'l' }, el('span', { class: 'wbar', style: `width:${Math.round(80 * c.weight / top)}px` }), `${fmt(c.weight * 100, 1)}%`),
    el('td', {}, `${fmt(c.weight_30d * 100, 1)}%`),
    el('td', { class: sign(c.weight - c.weight_30d) }, `${fmtSigned((c.weight - c.weight_30d) * 100, 1)} p.p.`),
    el('td', { class: c.on ? 'pos' : 'muted' }, c.on ? 'preço acima' : 'preço abaixo')));
  const head = el('tr', {}, ...[['média', 'l'], ['peso agora', 'l'], ['há 30 dias'], ['mudança'], ['hoje']]
    .map(([h, c]) => el('th', c ? { class: c } : {}, h)));
  $('sg-weights').replaceChildren(el('div', { class: 'scroll' }, el('table', {}, el('thead', {}, head), el('tbody', {}, ...rows))));
}

function sgRender() {
  const list = sgAll();
  const x = sgSel();
  document.querySelectorAll('[data-strat]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.strat === x.id)));
  $('sg-foot').textContent = `Regra de "${x.name}": ${x.rule}. Nenhuma ordem é enviada: são carteiras simuladas.`
    + ' Antes do início ao vivo, os dias foram reconstruídos com a mesma regra (simulado, marcas claras); depois, cada dia'
    + ' é decidido quando fecha e gravado sem reescrita (ao vivo, marcas fortes). A congelada é o controle; o aprendiz'
    + ' muda os pesos sozinho conforme os sinais de cada média rendem. Datas em UTC; inclui 0,1% de taxa por lado.';
  if (!x || !x.days.length) {
    $('sg-cards').replaceChildren(el('p', { class: 'muted' },
      'As carteiras simuladas ainda não têm dias gravados: o serviço paper grava o histórico na primeira execução.'));
    return;
  }
  sgCharts();
  const days = x.days;
  SG.byDay = new Map(days.map((d) => [ymd(d.t), d]));
  const bars = days.map((d) => ({ time: ymd(d.t), open: d.o, high: d.h, low: d.l, close: d.c }));
  if (x.now) {
    const prev = days[days.length - 1];
    const t = ymd(x.now.t - 300);
    if (t > bars[bars.length - 1].time) {
      bars.push({ time: t, open: prev.c, high: Math.max(prev.c, x.now.price), low: Math.min(prev.c, x.now.price), close: x.now.price });
    }
  }
  SG.candles.setData(bars);
  x.chart_smas.forEach((n, i) => SG.smas[i].setData(days.filter((d) => d.sma[i] !== null)
    .map((d) => ({ time: ymd(d.t), value: d.sma[i] }))));
  SG.smas.forEach((s) => s.applyOptions({ visible: SG.sma }));
  const times = new Set(bars.map((b) => b.time));
  const ms = [];
  for (const o of x.orders) {
    if (!o.live && !SG.sim) continue;
    const buy = o.action === 'compra';
    ms.push({
      time: ymd(o.t), position: buy ? 'belowBar' : 'aboveBar', shape: buy ? 'arrowUp' : 'arrowDown',
      color: buy ? (o.live ? C.tp : 'rgba(63, 185, 80, .42)') : (o.live ? C.sl : 'rgba(248, 81, 73, .42)'),
      text: (buy ? '+' : '−') + fmt(o.size * 100, 0) + '%',
    });
  }
  if (x.live_start && times.has(x.live_start)) {
    ms.push({ time: x.live_start, position: 'aboveBar', shape: 'circle', color: C.accent, text: 'início ao vivo' });
  }
  ms.sort((a, c) => (a.time < c.time ? -1 : a.time > c.time ? 1 : 0));
  SG.candles.setMarkers(ms);

  const hasLive = days.some((d) => d.live);
  $('sg-notice').hidden = !(SG.range === 'live' && !hasLive);
  $('sg-notice').textContent = 'O período ao vivo ainda não fechou o primeiro dia: a curva mostra o histórico simulado até lá.';
  const baseT = sgBaseT();
  let holdSet = false;
  for (const y of list) {
    const c = sgCurve(y, baseT);
    SG.eqLines[y.id].setData(c.eq);
    SG.eqLines[y.id].applyOptions({ lineWidth: y.id === x.id ? 2 : 1 });
    if (!holdSet && c.base) { SG.hold.setData(c.hold); holdSet = true; }
  }
  SG.eqChart.timeScale().fitContent();
  if (!SG.fitted) {
    SG.chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, bars.length - 365), to: bars.length + 3 });
    SG.fitted = true;
  }
  sgCards();
  sgWeights();
  sgOrders();
  sgLegend(null);
  sgEqLegend(null);
  $('sg-updated').textContent = `atualizado ${fmtSec(Date.now() / 1000)} · último dia fechado ${fmtUTCDay(days[days.length - 1].t)}`;
}

async function loadStrategy() {
  SG.last = Date.now();
  try {
    SG.data = await api('paper');
    sgRender();
  } catch (e) {
    $('sg-cards').replaceChildren(el('p', { class: 'muted' }, 'Erro ao ler as carteiras simuladas: ' + e.message));
  }
}

$('sg-show-sma').checked = SG.sma;
$('sg-show-sim').checked = SG.sim;
$('sg-show-sma').addEventListener('change', (e) => { SG.sma = e.target.checked; store.set('sgSma', SG.sma); if (SG.data) sgRender(); });
$('sg-show-sim').addEventListener('change', (e) => { SG.sim = e.target.checked; store.set('sgSim', SG.sim); if (SG.data) sgRender(); });
document.querySelectorAll('[data-range]').forEach((btn) => {
  btn.setAttribute('aria-pressed', String(btn.dataset.range === SG.range));
  btn.addEventListener('click', () => {
    SG.range = btn.dataset.range;
    store.set('sgRange', SG.range);
    document.querySelectorAll('[data-range]').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
    if (SG.data) sgRender();
  });
});
document.querySelectorAll('[data-strat]').forEach((btn) => btn.addEventListener('click', () => {
  SG.strat = btn.dataset.strat;
  store.set('sgStrat', SG.strat);
  if (SG.data) sgRender();
}));

// --- laço ---
let lastGaps = 0;
async function tick() {
  if (document.hidden) return;
  loadHealth();
  if (!$('tab-strategy').hidden) {
    if (Date.now() - SG.last > SG_EVERY_MS) loadStrategy();
    return;
  }
  if ($('tab-chart').hidden) return;
  try { await refreshTail(); } catch (e) { console.warn('candles:', e); }
  if (Date.now() - lastGaps > GAPS_MS) { lastGaps = Date.now(); loadGaps(); }
  const r = chart.timeScale().getVisibleLogicalRange();
  if (S.show.flow && r && r.to >= S.items.length - 2) loadFlow(true);
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) tick(); });

(async function start() {
  if (location.hash === '#pesquisa') document.querySelector('[data-tab="research"]').click();
  if (location.hash === '#estrategia') document.querySelector('[data-tab="strategy"]').click();
  applyVisibility();
  loadHealth();
  await loadInitial();
  lastGaps = Date.now();
  legendEl.innerHTML = legendFor(S.items[S.items.length - 1]);
  resizeFlow();
  loadFlow(true);
  setInterval(tick, POLL_MS);
})();
