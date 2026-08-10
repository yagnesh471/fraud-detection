import { useEffect, useState, useRef, useMemo } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:5001/api";

const PRIMARY_FEATURES = [
  { key: "Time", hint: "Seconds since first transaction in dataset" },
  { key: "Amount", hint: "Transaction value" },
];

const PCA_FEATURES = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const ALL_FEATURES = [...PRIMARY_FEATURES.map((f) => f.key), ...PCA_FEATURES];

const createEmptyTransaction = () =>
  Object.fromEntries(ALL_FEATURES.map((feature) => [feature, ""]));

const GAUGE_RADIUS = 68;
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;
const GAUGE_TICKS = 40; // one tick every 2.5% of dial travel


function App() {
  const [transaction, setTransaction] = useState(createEmptyTransaction());

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [animatedPct, setAnimatedPct] = useState(0);

  const resultRef = useRef(null);

  // =========================================================
  // INPUT CHANGE
  // =========================================================

  const handleChange = (e) => {
    const { name, value } = e.target;

    setTransaction((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // =========================================================
  // FETCH HISTORY
  // =========================================================

  const fetchHistory = async () => {
    try {
      setHistoryLoading(true);

      const response = await axios.get(`${API_URL}/predictions`);

      setHistory(Array.isArray(response.data) ? response.data : []);
    } catch (error) {
      console.error("Failed to load prediction history:", error);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // =========================================================
  // SUBMIT PREDICTION
  // =========================================================

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setResult(null);
    setAnimatedPct(0);

    try {
      const numericTransaction = Object.fromEntries(
        ALL_FEATURES.map((feature) => [feature, Number(transaction[feature])])
      );

      const response = await axios.post(`${API_URL}/predict`, numericTransaction);

      setResult(response.data);

      // Refresh history after prediction
      await fetchHistory();
    } catch (error) {
      console.error("Prediction error:", error);

      setResult({
        error: error.response?.data?.message || "Couldn't score this transaction.",
        detail: "Make sure the backend services are running, then try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  // =========================================================
  // CLEAR FORM
  // =========================================================

  const clearForm = () => {
    setTransaction(createEmptyTransaction());
    setResult(null);
    setAnimatedPct(0);
  };

  // =========================================================
  // RESULT
  // =========================================================

  const isFraud = result?.prediction === "Fraud";

  const pct = result?.fraudProbability != null ? result.fraudProbability * 100 : 0;

  useEffect(() => {
    if (result && !result.error) {
      const t = setTimeout(() => setAnimatedPct(pct), 60);

      resultRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });

      return () => clearTimeout(t);
    }
  }, [result, pct]);

  const dashOffset =
    GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * Math.min(animatedPct, 100)) / 100;

  // tick marks around the dial, evenly spaced on the same 270deg travel
  const gaugeTicks = useMemo(() => {
    const cx = 90;
    const cy = 90;
    const rOuter = GAUGE_RADIUS + 12;
    const rInner = GAUGE_RADIUS + 7;

    return Array.from({ length: GAUGE_TICKS }, (_, i) => {
      const angle = (i / GAUGE_TICKS) * 2 * Math.PI;
      const x1 = cx + rInner * Math.cos(angle);
      const y1 = cy + rInner * Math.sin(angle);
      const x2 = cx + rOuter * Math.cos(angle);
      const y2 = cy + rOuter * Math.sin(angle);
      return { id: i, x1, y1, x2, y2 };
    });
  }, []);

  // =========================================================
  // DASHBOARD CALCULATIONS
  // =========================================================

  const totalChecks = history.length;

  const fraudCount = history.filter((item) => item.prediction === "Fraud").length;

  const legitimateCount = history.filter(
    (item) => item.prediction === "Legitimate"
  ).length;

  const fraudRate = totalChecks > 0 ? (fraudCount / totalChecks) * 100 : 0;

  const averageFraudProbability =
    totalChecks > 0
      ? history.reduce((sum, item) => sum + Number(item.fraudProbability || 0), 0) /
        totalChecks
      : 0;

  const averageFraudPercentage = averageFraudProbability * 100;

  // =========================================================
  // RECENT PREDICTIONS
  // =========================================================

  const recentPredictions = useMemo(() => {
    return history.slice(0, 5);
  }, [history]);

  return (
    <div className="app">
      {/* =====================================================
          TOP BAR
      ====================================================== */}

      <div className="topbar">
        <div className="topbar-inner">
          <div className="wordmark">
            <span className="wordmark-mark" aria-hidden="true">
              ◆
            </span>

            <span className="wordmark-text">
              SIGNAL<span className="wordmark-dim">/</span>FRAUD
            </span>
          </div>

          <div className="status-pill">
            <span className="status-dot" aria-hidden="true"></span>
            Model online
          </div>
        </div>
      </div>

      {/* =====================================================
          MAIN LAYOUT
      ====================================================== */}

      <main className="layout">
        {/* ===================================================
            SIDEBAR
        ==================================================== */}

        <aside className="sidebar">
          <p className="sidebar-eyebrow">ML risk engine</p>

          <h1 className="sidebar-title">Credit card fraud detection</h1>

          <p className="sidebar-desc">
            Score a single transaction against a trained XGBoost classifier.
            Enter all 30 features to read a fraud probability, a verdict, and
            a running history of every check.
          </p>

          <dl className="meta-list">
            <div className="meta-item">
              <dt className="meta-label">Features</dt>
              <dd className="meta-value">30</dd>
            </div>

            <div className="meta-item">
              <dt className="meta-label">Model</dt>
              <dd className="meta-value">XGBoost</dd>
            </div>

            <div className="meta-item">
              <dt className="meta-label">Logged</dt>
              <dd className="meta-value">{totalChecks} checks</dd>
            </div>
          </dl>

          <p className="sidebar-note">
            V1–V28 come from a PCA transform, so the raw values are
            anonymized. Time and Amount are the only human-readable fields.
          </p>
        </aside>

        {/* ===================================================
            RIGHT SIDE
        ==================================================== */}

        <section className="panel-column">
          {/* =================================================
              DASHBOARD
          ================================================== */}

          <section className="dashboard-panel">
            <div className="dashboard-header">
              <div>
                <p className="panel-eyebrow">System overview</p>
                <h2 className="panel-title">Fraud detection dashboard</h2>
              </div>

              <button
                type="button"
                className="refresh-btn"
                onClick={fetchHistory}
                disabled={historyLoading}
              >
                {historyLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>

            {/* Dashboard cards */}

            <div className="dashboard-grid">
              {/* TOTAL CHECKS */}
              <div className="dashboard-card">
                <div className="dashboard-card-top">
                  <span className="dashboard-label">Total checks</span>
                  <span className="dashboard-icon">#</span>
                </div>

                <strong className="dashboard-value">{totalChecks}</strong>

                <span className="dashboard-description">Transactions analyzed</span>
              </div>

              {/* FRAUD */}
              <div className="dashboard-card dashboard-card-fraud">
                <div className="dashboard-card-top">
                  <span className="dashboard-label">Fraud detected</span>
                  <span className="dashboard-icon">!</span>
                </div>

                <strong className="dashboard-value">{fraudCount}</strong>

                <span className="dashboard-description">Transactions flagged</span>
              </div>

              {/* LEGITIMATE */}
              <div className="dashboard-card dashboard-card-safe">
                <div className="dashboard-card-top">
                  <span className="dashboard-label">Legitimate</span>
                  <span className="dashboard-icon">✓</span>
                </div>

                <strong className="dashboard-value">{legitimateCount}</strong>

                <span className="dashboard-description">Transactions cleared</span>
              </div>

              {/* FRAUD RATE */}
              <div className="dashboard-card">
                <div className="dashboard-card-top">
                  <span className="dashboard-label">Fraud rate</span>
                  <span className="dashboard-icon">%</span>
                </div>

                <strong className="dashboard-value">{fraudRate.toFixed(2)}%</strong>

                <span className="dashboard-description">
                  Of all analyzed transactions
                </span>
              </div>
            </div>

            {/* Dashboard summary */}

            <div className="dashboard-summary">
              <div>
                <span>Average fraud probability</span>
                <strong>{averageFraudPercentage.toFixed(4)}%</strong>
              </div>

              <div>
                <span>Model</span>
                <strong>XGBoost</strong>
              </div>

              <div>
                <span>Features analyzed</span>
                <strong>30</strong>
              </div>
            </div>
          </section>

          {/* =================================================
              ANALYTICS
          ================================================== */}

          <section className="analytics-section">
            {/* FRAUD VS LEGITIMATE */}

            <div className="analytics-card">
              <div className="analytics-card-header">
                <div>
                  <p className="panel-eyebrow">Classification overview</p>
                  <h3>Fraud vs Legitimate</h3>
                </div>

                <span className="analytics-count">{totalChecks} total</span>
              </div>

              <div className="classification-chart">
                {/* Legitimate */}
                <div className="chart-row">
                  <div className="chart-label">
                    <span className="chart-dot safe-dot"></span>
                    Legitimate
                  </div>

                  <div className="chart-track">
                    <div
                      className="chart-bar safe-bar"
                      style={{
                        width:
                          totalChecks > 0
                            ? `${(legitimateCount / totalChecks) * 100}%`
                            : "0%",
                      }}
                    ></div>
                  </div>

                  <strong className="chart-number">{legitimateCount}</strong>
                </div>

                {/* Fraud */}
                <div className="chart-row">
                  <div className="chart-label">
                    <span className="chart-dot fraud-dot"></span>
                    Fraud
                  </div>

                  <div className="chart-track">
                    <div
                      className="chart-bar fraud-bar"
                      style={{
                        width:
                          totalChecks > 0
                            ? `${(fraudCount / totalChecks) * 100}%`
                            : "0%",
                      }}
                    ></div>
                  </div>

                  <strong className="chart-number">{fraudCount}</strong>
                </div>
              </div>
            </div>

            {/* RECENT PREDICTIONS */}

            <div className="analytics-card">
              <div className="analytics-card-header">
                <div>
                  <p className="panel-eyebrow">Live activity</p>
                  <h3>Recent predictions</h3>
                </div>

                <span className="analytics-count">Latest 5</span>
              </div>

              <div className="recent-list">
                {recentPredictions.length === 0 ? (
                  <div className="recent-empty">No prediction activity yet.</div>
                ) : (
                  recentPredictions.map((item, index) => {
                    const probability = Number(item.fraudProbability || 0) * 100;
                    const fraud = item.prediction === "Fraud";

                    let riskLevel = "Low";

                    if (probability >= 50) {
                      riskLevel = "High";
                    } else if (probability >= 10) {
                      riskLevel = "Medium";
                    }

                    return (
                      <div className="recent-item" key={item._id || item.id || index}>
                        <div className="recent-status">
                          <span
                            className={`recent-status-dot ${
                              fraud ? "recent-dot-fraud" : "recent-dot-safe"
                            }`}
                          ></span>

                          <div>
                            <strong>
                              {fraud ? "Fraud detected" : "Legitimate transaction"}
                            </strong>
                            <span className="recent-date">
                              {item.createdAt
                                ? new Date(item.createdAt).toLocaleString()
                                : "Unknown time"}
                            </span>
                          </div>
                        </div>

                        <div className="recent-probability">
                          <strong>{probability.toFixed(4)}%</strong>
                          <span className={`risk-label risk-${riskLevel.toLowerCase()}`}>
                            {riskLevel} risk
                          </span>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </section>

          {/* =================================================
              TRANSACTION FORM
          ================================================== */}

          <form className="panel" onSubmit={handleSubmit}>
            <div className="panel-header">
              <div>
                <p className="panel-eyebrow">Transaction</p>
                <h2 className="panel-title">Enter feature values</h2>
              </div>

              <span className="panel-count">{ALL_FEATURES.length} required</span>
            </div>

            {/* Time + Amount */}

            <div className="primary-fields">
              {PRIMARY_FEATURES.map(({ key, hint }) => (
                <div className="field-primary" key={key}>
                  <label htmlFor={key}>
                    <span>{key}</span>
                    <span className="field-hint">{hint}</span>
                  </label>

                  <input
                    id={key}
                    name={key}
                    type="number"
                    step="any"
                    inputMode="decimal"
                    value={transaction[key]}
                    onChange={handleChange}
                    placeholder="0.00"
                    required
                  />
                </div>
              ))}
            </div>

            {/* PCA */}

            <div className="pca-header">
              <span>PCA components</span>
              <span className="pca-header-line" aria-hidden="true"></span>
            </div>

            <div className={`feature-grid ${loading ? "is-scanning" : ""}`}>
              {PCA_FEATURES.map((feature) => (
                <div className="field-mini" key={feature}>
                  <label htmlFor={feature}>{feature}</label>

                  <input
                    id={feature}
                    name={feature}
                    type="number"
                    step="any"
                    inputMode="decimal"
                    value={transaction[feature]}
                    onChange={handleChange}
                    placeholder="0"
                    required
                  />
                </div>
              ))}
            </div>

            {/* Buttons */}

            <div className="submit-row">
              <button
                type="button"
                className="clear-btn"
                onClick={clearForm}
                disabled={loading}
              >
                Clear form
              </button>

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner" aria-hidden="true"></span>
                    Scoring transaction...
                  </>
                ) : (
                  <>Run analysis</>
                )}
              </button>
            </div>
          </form>

          {/* =================================================
              RESULT
          ================================================== */}

          <div ref={resultRef}>
            {result && !result.error && (
              <section className={`result-panel ${isFraud ? "is-fraud" : "is-safe"}`}>
                <div className="result-head">
                  {/* Gauge — radar-style dial with tick marks */}

                  <div className="gauge-wrap">
                    <svg viewBox="0 0 180 180" className="gauge">
                      <g className="gauge-ticks" transform="rotate(90 90 90)">
                        {gaugeTicks.map((tick) => (
                          <line
                            key={tick.id}
                            className="gauge-tick"
                            x1={tick.x1}
                            y1={tick.y1}
                            x2={tick.x2}
                            y2={tick.y2}
                          />
                        ))}
                      </g>

                      <circle
                        className="gauge-track"
                        cx="90"
                        cy="90"
                        r={GAUGE_RADIUS}
                      />

                      <circle
                        className="gauge-value"
                        cx="90"
                        cy="90"
                        r={GAUGE_RADIUS}
                        style={{
                          strokeDasharray: GAUGE_CIRCUMFERENCE,
                          strokeDashoffset: dashOffset,
                        }}
                      />
                    </svg>

                    <div className="gauge-center">
                      <span className="gauge-pct">{pct.toFixed(4)}%</span>
                      <span className="gauge-pct-label">fraud probability</span>
                    </div>
                  </div>

                  {/* Result text */}

                  <div className="result-copy">
                    <p className="result-eyebrow">Model prediction</p>

                    <h2 className="result-verdict">
                      {isFraud
                        ? "Transaction flagged as fraud"
                        : "Transaction looks clean"}
                    </h2>

                    <p className="result-explain">
                      {isFraud
                        ? "This transaction scored above the fraud threshold. Review it before authorizing."
                        : "This transaction scored below the fraud threshold. No action needed."}
                    </p>
                  </div>
                </div>

                <div className="result-bar-track">
                  <div
                    className="result-bar-fill"
                    style={{ width: `${Math.min(pct, 100)}%` }}
                  ></div>
                </div>

                <div className="result-bar-scale">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </section>
            )}

            {/* Error */}

            {result?.error && (
              <section className="error-panel">
                <span className="error-icon" aria-hidden="true">
                  !
                </span>

                <div>
                  <strong>{result.error}</strong>
                  <p>{result.detail}</p>
                </div>
              </section>
            )}
          </div>

          {/* =================================================
              HISTORY
          ================================================== */}

          <section className="history-panel">
            <div className="history-head">
              <div>
                <p className="panel-eyebrow">Audit trail</p>
                <h2 className="panel-title">Prediction history</h2>
              </div>

              <button
                type="button"
                className="refresh-btn"
                onClick={fetchHistory}
                disabled={historyLoading}
              >
                {historyLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>

            {historyLoading && history.length === 0 && (
              <div className="history-state">Loading prediction history...</div>
            )}

            {!historyLoading && history.length === 0 && (
              <div className="history-state history-state-empty">
                <p>No predictions yet.</p>
                <span>Run your first transaction analysis to see it here.</span>
              </div>
            )}

            {history.length > 0 && (
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Amount</th>
                      <th>Fraud probability</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>

                  <tbody>
                    {history.map((item, index) => {
                      const probability = Number(item.fraudProbability);
                      const fraud = item.prediction === "Fraud";

                      return (
                        <tr key={item._id || item.id || index}>
                          <td className="cell-mono">
                            {item.createdAt
                              ? new Date(item.createdAt).toLocaleString()
                              : "—"}
                          </td>

                          <td className="cell-mono">
                            {item.amount !== undefined && item.amount !== null
                              ? Number(item.amount).toFixed(2)
                              : "—"}
                          </td>

                          <td className="cell-mono">{(probability * 100).toFixed(4)}%</td>

                          <td>
                            <span
                              className={`status-badge ${
                                fraud ? "status-fraud" : "status-safe"
                              }`}
                            >
                              {fraud ? "Fraud" : "Legitimate"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </section>
      </main>

      {/* =====================================================
          FOOTER
      ====================================================== */}

      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <span className="wordmark-mark" aria-hidden="true">
              ◆
            </span>

            <div>
              <p className="footer-title">Fraud detection system</p>
              <p className="footer-sub">
                Predictions are model estimates, not final decisions — route
                flagged transactions to manual review.
              </p>
            </div>
          </div>

          <span className="footer-stack">React · Express · Flask · XGBoost · MongoDB</span>
        </div>
      </footer>
    </div>
  );
}

export default App;