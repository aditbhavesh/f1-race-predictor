/* Race Monte Carlo, the same procedure as src/models.py::simulate.
   Each simulated race: every driver retires with probability pDnf, otherwise finishes with
   score + an error drawn from the pool of past model errors for his predicted band.
   Finishers are ranked by that value; retirements are placed behind every finisher. */
(function (root) {
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // number of edges strictly below v: same as numpy.searchsorted(edges, v) (side = left)
  function band(edges, v) {
    let lo = 0, hi = edges.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (edges[m] < v) lo = m + 1; else hi = m; }
    return lo;
  }
  function simulate(score, pDnf, edges, pools, nSims, seed) {
    const n = score.length, rnd = mulberry32(seed | 0);
    const bands = score.map((s) => band(edges, s));
    const win = new Float64Array(n), pod = new Float64Array(n), top = new Float64Array(n), dnf = new Float64Array(n);
    const posSum = new Float64Array(n), fin = new Float64Array(n);
    const counts = Array.from({ length: n }, () => new Float64Array(n));
    const val = new Float64Array(n), idx = new Array(n);
    for (let s = 0; s < nSims; s++) {
      let m = 0;
      for (let i = 0; i < n; i++) {
        if (rnd() < pDnf[i]) { dnf[i]++; continue; }
        const pool = pools[bands[i]];
        val[i] = score[i] + pool[Math.floor(rnd() * pool.length)];
        idx[m++] = i;
      }
      const f = idx.slice(0, m).sort((a, b) => val[a] - val[b] || a - b);
      for (let r = 0; r < m; r++) {
        const i = f[r], p = r + 1;
        counts[i][r]++; posSum[i] += p; fin[i]++;
        if (p === 1) win[i]++;
        if (p <= 3) pod[i]++;
        if (p <= 10) top[i]++;
      }
    }
    const k = 1 / nSims;
    return {
      p_win: Array.from(win, (x) => x * k), p_podium: Array.from(pod, (x) => x * k), p_top10: Array.from(top, (x) => x * k),
      p_dnf: Array.from(dnf, (x) => x * k), exp_pos: Array.from(posSum, (x, i) => (fin[i] ? x / fin[i] : null)),
      pos_counts: counts.map((c) => Array.from(c, (x) => x * k)),
    };
  }
  // order[slot] = driver index. slot_score[driver][slot], slot_dnf[driver][slot] come from the model.
  function whatIf(data, order, nSims, seed) {
    const n = order.length, score = new Array(n), pd = new Array(n);
    for (let slot = 0; slot < n; slot++) {
      const d = order[slot];
      score[d] = data.slot_score[d][slot]; pd[d] = data.slot_dnf[d][slot];
    }
    return simulate(score, pd, data.edges, data.pools, nSims, seed);
  }
  const api = { simulate, whatIf, mulberry32, band };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Engine = api;
})(typeof window !== "undefined" ? window : globalThis);
