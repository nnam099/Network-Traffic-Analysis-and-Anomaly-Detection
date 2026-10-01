import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import Papa from "papaparse";
import { Activity, AlertTriangle, CheckCircle2, FileUp, ShieldCheck } from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8080";

function App() {
  const [health, setHealth] = useState(null);
  const [rows, setRows] = useState([]);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("dashboard");

  useEffect(() => { checkHealth(); }, []);

  const summary = useMemo(() => ({
    total: results.length,
    anomalies: results.filter((item) => item.is_anomaly).length,
    knownAttacks: results.filter((item) => !item.is_anomaly && ["Known Attack", "Known-Attack"].includes(item.label)).length,
    normal: results.filter((item) => !item.is_anomaly && ["Normal Traffic", "Normal"].includes(item.label)).length,
  }), [results]);

  async function checkHealth() {
    try {
      const response = await fetch(`${API}/health`);
      if (!response.ok) throw new Error("API unavailable");
      setHealth(await response.json()); setError("");
    } catch (err) { setHealth(null); setError(`Cannot connect to the Inference API at ${API}. Start FastAPI and try again.`); }
  }

  function loadCsv(file) {
    Papa.parse(file, { header: true, skipEmptyLines: true, complete: ({ data }) => {
      setRows(data.slice(0, 100)); setResults([]); setError("");
    }, error: (err) => setError(err.message) });
  }

  async function analyze() {
    setBusy(true); setError("");
    try {
      const output = [];
      for (const event of rows) {
        const response = await fetch(`${API}/predict/flow`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ event }),
        });
        if (!response.ok) throw new Error(await response.text());
        output.push(await response.json());
      }
      setResults(output);
    } catch (err) {
      const detail = String(err.message || "");
      setError(detail.includes("Failed to fetch")
        ? `Cannot connect to the Inference API at ${API}. Start FastAPI, then click Check API.`
        : `Analysis failed: ${detail}`);
    }
    finally { setBusy(false); }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><ShieldCheck size={19} /></div><div><strong>Network Anomaly IDS</strong><span>Traffic analysis and detection</span></div></div>
      <nav><button className={activeView === "dashboard" ? "active" : ""} onClick={() => setActiveView("dashboard")}><Activity size={17} />Dashboard</button><button className={activeView === "model" ? "active" : ""} onClick={() => setActiveView("model")}><ShieldCheck size={17} />Model architecture</button></nav>
      <div className="side-note"><span className={health ? "dot online" : "dot"}></span><div><b>Inference API</b><small>{health ? `Online · ${health.model_version}` : "Not checked"}</small></div></div>
    </aside>
    <main className="main">
      {activeView === "model" ? <ModelArchitecture onBack={() => setActiveView("dashboard")} /> : <>
      <header><div><p className="eyebrow">INTRUSION DETECTION SYSTEM</p><h1>Network Traffic Analysis</h1><p className="subtitle">Classify flows and identify potentially anomalous traffic with the v14 hybrid model.</p></div><button className="secondary" onClick={checkHealth}><span className={health ? "dot online" : "dot"}></span> Check API</button></header>
      <section className="metrics"><Metric label="Flows analyzed" value={summary.total} icon={<Activity />} /><Metric label="Known attacks" value={summary.knownAttacks} icon={<AlertTriangle />} tone="warning" /><Metric label="Anomalous traffic" value={summary.anomalies} icon={<AlertTriangle />} tone="danger" /><Metric label="Normal traffic" value={summary.normal} icon={<CheckCircle2 />} tone="success" /></section>
      <section className="workspace" id="analyze"><div className="section-heading"><div><p className="eyebrow">CORE WORKFLOW</p><h2>Analyze a CSV file</h2></div><span className="hint">Up to 100 rows per session</span></div>
        <label className="dropzone"><FileUp size={25} /><b>Choose a flow CSV</b><span>UNSW-NB15, CICFlowMeter, firewall or NetFlow-style data</span><input type="file" accept=".csv" onChange={(event) => event.target.files[0] && loadCsv(event.target.files[0])} /></label>
        {rows.length > 0 && <div className="file-row"><span>{rows.length} rows loaded</span><button className="primary" disabled={busy} onClick={analyze}>{busy ? "Analyzing…" : "Run analysis"}</button></div>}
      </section>
      {error && <div className="error"><AlertTriangle size={18} />{error}</div>}
      {results.length > 0 && <section className="workspace"><div className="section-heading"><div><p className="eyebrow">RESULTS</p><h2>Detection results</h2></div></div><div className="table-wrap"><table><thead><tr><th>#</th><th>Verdict</th><th>Classifier confidence</th><th>Anomaly score</th></tr></thead><tbody>{results.map((item, index) => { const verdict = item.is_anomaly ? "Anomalous Traffic" : ["Known Attack", "Known-Attack"].includes(item.label) ? "Known Attack" : "Normal Traffic"; const tone = verdict === "Normal Traffic" ? "safe" : "danger"; return <tr key={index}><td>{index + 1}</td><td><span className={`badge ${tone}`}>{verdict}</span></td><td>{(item.confidence * 100).toFixed(1)}%</td><td>{Number(item.hybrid_score).toFixed(4)}</td></tr>; })}</tbody></table></div></section>}
      </>}
    </main>
  </div>;
}

function ModelArchitecture({ onBack }) {
  return <>
    <header><div><p className="eyebrow">MODEL OVERVIEW</p><h1>Model architecture</h1><p className="subtitle">How the IDS combines supervised classification with autoencoder-based anomaly detection.</p></div><button className="secondary" onClick={onBack}>Back to dashboard</button></header>
    <section className="workspace"><div className="section-heading"><div><p className="eyebrow">INFERENCE PIPELINE</p><h2>From network flow to detection verdict</h2></div></div><div className="pipeline"><PipelineStep number="01" title="Raw network flow" text="CSV or flow event from UNSW-NB15, CICFlowMeter, firewall or NetFlow." /><span className="arrow">→</span><PipelineStep number="02" title="Normalization" text="Schema mapping, categorical encoding, feature scaling and validation." /><span className="arrow">→</span><PipelineStep number="03" title="Hybrid model" text="Classifier predicts known classes while the autoencoder measures reconstruction error." /><span className="arrow">→</span><PipelineStep number="04" title="Anomaly decision" text="The calibrated anomaly rule assigns Normal Traffic, Known Attack or Anomalous Traffic." /></div></section>
    <section className="architecture-grid"><div className="workspace"><p className="eyebrow">MODEL COMPONENTS</p><h2>Two complementary signals</h2><div className="component-card"><span className="component-tag classifier">CLASSIFIER</span><h3>Known attack classification</h3><p>Estimates the most likely traffic class and its confidence from learned labeled examples.</p></div><div className="component-card"><span className="component-tag autoencoder">AUTOENCODER</span><h3>Reconstruction error</h3><p>Measures how different a flow is from the learned traffic representation.</p></div></div><div className="workspace"><p className="eyebrow">OUTPUT</p><h2>Detection verdicts</h2><div className="verdict-row"><span className="badge safe">Normal Traffic</span><span>Traffic is not flagged and the classifier predicts Normal.</span></div><div className="verdict-row"><span className="badge danger">Known Attack</span><span>Classifier recognizes a known attack class.</span></div><div className="verdict-row"><span className="badge danger">Anomalous Traffic</span><span>The calibrated anomaly rule flags this flow for review.</span></div></div></section>
  </>;
}

function PipelineStep({ number, title, text }) { return <div className="pipeline-step"><span className="step-number">{number}</span><h3>{title}</h3><p>{text}</p></div>; }

function Metric({ label, value, icon, tone = "" }) { return <div className={`metric ${tone}`}><div className="metric-icon">{icon}</div><span>{label}</span><strong>{value}</strong></div>; }
createRoot(document.getElementById("root")).render(<App />);
