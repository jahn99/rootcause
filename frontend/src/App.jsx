import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const TABS = ["Investigate", "Report", "Evidence", "Review", "Ask", "SQL agent"];

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function text(value, fallback = "Not available") {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function list(value) {
  return Array.isArray(value) ? value : [];
}

function errorMessage(payload, status) {
  if (isRecord(payload) && typeof payload.detail === "string") return payload.detail;
  if (isRecord(payload) && Array.isArray(payload.detail)) {
    const messages = payload.detail
      .filter((item) => isRecord(item) && typeof item.msg === "string")
      .map((item) => item.msg);
    if (messages.length) return messages.join(" ");
  }
  if (isRecord(payload) && typeof payload.message === "string") return payload.message;
  return `Request failed (${status}). Check that the API is running and try again.`;
}

async function post(path, body) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(`Could not reach the API at ${API_BASE}. Check the API address and try again.`);
  }

  const raw = await response.text();
  let payload = {};
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    payload = { message: raw };
  }
  if (!response.ok) throw new Error(errorMessage(payload, response.status));
  return isRecord(payload) ? payload : {};
}

function Icon({ children }) {
  return <span className="nav-icon" aria-hidden="true">{children}</span>;
}

function Panel({ title, eyebrow, children, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      {title && <h2 className="panel-title">{title}</h2>}
      {children}
    </section>
  );
}

function EmptyState({ title, description }) {
  return (
    <div className="empty-state">
      <span className="empty-mark" aria-hidden="true">+</span>
      <strong>{title}</strong>
      <span>{description}</span>
    </div>
  );
}

function BulletList({ items, empty = "Not available" }) {
  const values = list(items).filter((item) => typeof item === "string" && item.trim());
  if (!values.length) return <p className="muted">{empty}</p>;
  return (
    <ul className="bullet-list">
      {values.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}
    </ul>
  );
}

function topPartsScope(report) {
  const evidence = list(report?.supporting_data).find(
    (item) => isRecord(item) && item.id === "sql:top_parts",
  );
  if (!evidence || typeof evidence.sql !== "string") return null;

  const parameters = isRecord(evidence.parameters) ? evidence.parameters : {};
  const startDate = typeof parameters.start_date === "string" ? parameters.start_date : null;
  const endDate = typeof parameters.end_date === "string" ? parameters.end_date : null;
  const hasDateFilter = /\bdate\s*(?:>=|<=|>|<|between\b)/i.test(evidence.sql);
  if (hasDateFilter && startDate && endDate) return `${startDate} to ${endDate}`;
  if (hasDateFilter && startDate) return `from ${startDate} onward`;
  if (hasDateFilter && endDate) return `through ${endDate}`;

  const hasNoFilters =
    !/\bwhere\b/i.test(evidence.sql) &&
    /\bfrom\s+production_defects\b/i.test(evidence.sql) &&
    Object.keys(parameters).length === 0;
  return hasNoFilters ? "all dates" : null;
}

function KeyFindings({ items, report }) {
  const values = list(items).filter((item) => typeof item === "string" && item.trim());
  const scope = topPartsScope(report);
  if (!values.length) return <p className="muted">Not available</p>;
  return (
    <ul className="bullet-list">
      {values.map((item, index) => (
        <li key={`${index}-${item}`}>
          {item}
          {scope && item.startsWith("Top part by defect count:") && (
            <span className="finding-scope">Scope: {scope}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

function ReportView({ report }) {
  if (!report) {
    return <EmptyState title="No report yet" description="Run an investigation to create an evidence-linked 8D draft." />;
  }

  const hypotheses = list(report.root_cause_hypotheses).filter(isRecord);
  const documents = list(report.supporting_documents).filter(isRecord);
  return (
    <div className="view-stack">
      <Panel title="Key findings" eyebrow="Evidence summary" className="accent-panel">
        <KeyFindings items={report.key_findings} report={report} />
      </Panel>

      <div className="report-grid">
        <Panel title="Problem" eyebrow="D2 · Describe">
          <p className="report-copy">{text(report.problem_description)}</p>
        </Panel>
        <Panel title="Containment actions" eyebrow="D3 · Contain">
          <BulletList items={report.containment_actions} />
        </Panel>
        <Panel title="Root cause hypotheses" eyebrow="D4 · Investigate" className="wide-panel">
          {hypotheses.length ? (
            <div className="hypothesis-list">
              {hypotheses.map((item, index) => (
                <article className="hypothesis" key={`${index}-${text(item.hypothesis, "hypothesis")}`}>
                  <div className="hypothesis-head">
                    <strong>{text(item.hypothesis)}</strong>
                    <span className={`confidence confidence-${String(item.confidence_level || "unknown").toLowerCase()}`}>
                      {text(item.confidence_level)} confidence
                    </span>
                  </div>
                  <div className="evidence-ids">
                    {list(item.evidence).length
                      ? list(item.evidence).map((id, idIndex) => <code key={`${id}-${idIndex}`}>{text(id)}</code>)
                      : <span className="muted">Not available</span>}
                  </div>
                </article>
              ))}
            </div>
          ) : <p className="muted">Not available</p>}
        </Panel>
        <Panel title="Corrective actions" eyebrow="D5 · Correct">
          <BulletList items={report.corrective_actions} />
        </Panel>
        <Panel title="Preventive actions" eyebrow="D7 · Prevent">
          <BulletList items={report.preventive_actions} />
        </Panel>
        <Panel title="Limitations" eyebrow="Evidence boundaries">
          <BulletList items={report.limitations} />
        </Panel>
        <Panel title="Human review note" eyebrow="Draft status">
          <p className="report-copy">{text(report.human_review_note)}</p>
        </Panel>
      </div>

      <Panel title="Supporting documents" eyebrow="Public evidence">
        {documents.length ? (
          <div className="document-list">
            {documents.map((document, index) => (
              <article className="document-card" key={`${text(document.id)}-${index}`}>
                <div className="document-meta">
                  <code>{text(document.id)}</code>
                  <span>{text(document.source_type)}</span>
                  <span>{text(document.date)}</span>
                </div>
                <p>{text(document.snippet)}</p>
              </article>
            ))}
          </div>
        ) : <p className="muted">Not available</p>}
      </Panel>
    </div>
  );
}

function trendRows(report) {
  const records = list(report?.supporting_data).filter(isRecord);
  const chosen = records.find((item) => item.id === "sql:weekly_defects") || records.find((item) =>
    list(item.rows).some((row) => isRecord(row) && row.week !== undefined && row.defect_count !== undefined),
  );
  const points = list(chosen?.rows)
    .filter((row) => isRecord(row) && row.week !== undefined && Number.isFinite(Number(row.defect_count)))
    .map((row) => ({
      week: String(row.week).slice(0, 10),
      defects: Number(row.defect_count),
    }))
    .sort((a, b) => a.week.localeCompare(b.week));
  if (!points.length) return [];

  const baselineValues = points.map((point) => point.defects).sort((a, b) => a - b).slice(0, -1);
  const baseline = baselineValues.length
    ? baselineValues[Math.floor((baselineValues.length - 1) / 2)]
    : 0;
  return points.map((point) => ({
    ...point,
    spike: points.length > 1 && point.defects > Math.max(baseline * 1.5, baseline + 1),
  }));
}

function EvidenceChart({ report }) {
  const points = useMemo(() => trendRows(report), [report]);
  if (!points.length) {
    return <EmptyState title="No weekly defect trend" description="The report does not contain weekly defect rows to chart." />;
  }
  const spikeCount = points.filter((point) => point.spike).length;
  return (
    <div className="view-stack">
      <Panel title="Weekly defect trend" eyebrow="Manufacturing evidence" className="chart-panel">
        <div className="chart-legend">
          <span><i className="legend-dot normal-dot" /> Weekly defects</span>
          <span><i className="legend-dot spike-dot" /> Spike week</span>
        </div>
        <div className="chart-wrap" role="img" aria-label="Weekly defect count chart; spike weeks are highlighted in red">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={points} margin={{ top: 12, right: 12, left: -12, bottom: 4 }}>
              <CartesianGrid vertical={false} stroke="#e8edf0" />
              <XAxis dataKey="week" tickLine={false} axisLine={false} tick={{ fill: "#76858b", fontSize: 11 }} />
              <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: "#76858b", fontSize: 11 }} />
              <Tooltip
                cursor={{ fill: "#f0f4f4" }}
                contentStyle={{ borderRadius: 12, border: "1px solid #e1e8e9", boxShadow: "0 8px 24px #142d3810" }}
                formatter={(value) => [`${value ?? "Not available"} defects`, "Weekly count"]}
                labelFormatter={(label) => `Week of ${label}`}
              />
              <Bar dataKey="defects" name="Defects" radius={[5, 5, 0, 0]} maxBarSize={58}>
                {points.map((point) => <Cell key={point.week} fill={point.spike ? "#df6756" : "#4b8f82"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="chart-note">
          {spikeCount ? `${spikeCount} week${spikeCount > 1 ? "s" : ""} highlighted from the returned defect counts.` : "No spike week was distinguishable in the returned counts."}
          {" "}The chart uses only weekly rows returned by the API.
        </p>
      </Panel>
      <Panel title="Source rows" eyebrow="Supporting data">
        <pre className="data-block">{JSON.stringify(points, null, 2)}</pre>
      </Panel>
    </div>
  );
}

function AskView() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true); setError(""); setResult(null);
    try { setResult(await post("/ask", { question: question.trim() })); }
    catch (err) { setError(err.message || "The request failed. Please try again."); }
    finally { setLoading(false); }
  }

  return (
    <div className="view-stack">
      <Panel title="Ask the knowledge base" eyebrow="Public NHTSA documents">
        <form onSubmit={submit} className="form-stack">
          <label htmlFor="ask-question">Question</label>
          <textarea id="ask-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What reported failures relate to water leaks?" rows={4} />
          {error && <div className="error-box" role="alert">{error}</div>}
          <button className="button button-primary" type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? <><span className="spinner" /> Searching documents…</> : "Ask question"}
          </button>
        </form>
      </Panel>
      {result && <Panel title="Answer" eyebrow="API response">
        <p className="report-copy">{text(result.answer)}</p>
        <h3 className="subheading">Sources</h3>
        {list(result.sources).length ? <div className="document-list">
          {list(result.sources).map((source, index) => <article className="document-card" key={`${text(source?.id)}-${index}`}>
            <div className="document-meta"><code>{text(source?.id)}</code><span>{text(source?.component)}</span><span>{text(source?.date)}</span></div>
            <p>{text(source?.snippet)}</p>
          </article>)}
        </div> : <p className="muted">Not available</p>}
      </Panel>}
    </div>
  );
}

function SQLView() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true); setError(""); setResult(null);
    try { setResult(await post("/sql-agent", { question: question.trim() })); }
    catch (err) { setError(err.message || "The request failed. Please try again."); }
    finally { setLoading(false); }
  }

  return (
    <div className="view-stack">
      <Panel title="Ask manufacturing data" eyebrow="Guarded, read-only SQL">
        <form onSubmit={submit} className="form-stack">
          <label htmlFor="sql-question">Question</label>
          <textarea id="sql-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="How many door seal defects were recorded last month?" rows={4} />
          {error && <div className="error-box" role="alert">{error}</div>}
          <button className="button button-primary" type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? <><span className="spinner" /> Querying data…</> : "Run read-only query"}
          </button>
        </form>
      </Panel>
      {result && <div className="view-stack">
        {result.refused === true && <div className="refused-banner" role="status"><span aria-hidden="true">!</span><div><strong>Request refused</strong><p>This request was not run by the read-only data agent.</p></div></div>}
        <Panel title="Answer" eyebrow="SQL agent response"><p className="report-copy">{text(result.answer)}</p></Panel>
        <Panel title="SQL statement" eyebrow="Validated query"><pre className="data-block">{text(result.sql)}</pre></Panel>
        <Panel title="Returned rows" eyebrow="Query result"><pre className="data-block">{list(result.rows).length ? JSON.stringify(result.rows, null, 2) : "Not available"}</pre></Panel>
      </div>}
    </div>
  );
}

function ReviewView({ report, decision, setDecision, comment, setComment }) {
  return (
    <Panel title="Human review" eyebrow="Demo · Human-in-the-loop step" className="review-panel">
      <p className="report-copy">This decision is stored only in this browser session. It does not approve a production action or change the report in the backend.</p>
      {decision && <div className={`decision-banner decision-${decision}`} role="status">
        <span className="decision-icon">{decision === "approved" ? "✓" : "×"}</span>
        <div><strong>{decision} draft</strong><span>Local demo status</span></div>
      </div>}
      <label htmlFor="review-comment">Optional review comment</label>
      <textarea id="review-comment" value={comment} onChange={(event) => setComment(event.target.value)} placeholder="Add a note for this review…" rows={4} />
      <div className="button-row">
        <button className="button button-approve" type="button" disabled={!report} onClick={() => setDecision("approved")}>Approve draft</button>
        <button className="button button-reject" type="button" disabled={!report} onClick={() => setDecision("rejected")}>Reject draft</button>
      </div>
      {!report && <p className="muted review-hint">Run an investigation before recording a review decision.</p>}
    </Panel>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("Investigate");
  const [apiStatus, setApiStatus] = useState("checking");
  const [issue, setIssue] = useState("");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [decision, setDecision] = useState("");
  const [comment, setComment] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function checkApi() {
      try {
        const response = await fetch(`${API_BASE}/health`, {
          method: "GET",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) throw new Error("Health request failed");
        const payload = await response.json();
        if (payload?.status !== "ok") throw new Error("Unexpected health response");
        if (!cancelled) setApiStatus("connected");
      } catch {
        if (!cancelled) setApiStatus("failed");
      }
    }

    checkApi();
    return () => { cancelled = true; };
  }, []);

  async function investigate(event) {
    event.preventDefault();
    if (!issue.trim() || loading) return;
    setLoading(true); setError(""); setReport(null); setDecision(""); setComment("");
    try {
      setReport(await post("/investigate", { issue: issue.trim() }));
      setActiveTab("Report");
    } catch (err) {
      setError(err.message || "Investigation failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell min-h-screen bg-[#f5f7f8] text-slate-900">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="RootCause dashboard home">
          <span className="brand-mark"><span /></span>
          <span><strong>RootCause</strong><small>Quality engineering</small></span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Dashboard sections">
          {TABS.map((tab, index) => (
            <button key={tab} className={`nav-item ${activeTab === tab ? "nav-active" : ""}`} onClick={() => setActiveTab(tab)} aria-current={activeTab === tab ? "page" : undefined}>
              <Icon>{["⌕", "▤", "⌁", "✓", "?", "⌘"][index]}</Icon><span>{tab}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-connection" role="status" aria-live="polite">
          <span className={`connection-dot connection-dot-${apiStatus}`} />
          <div>
            <strong className={`connection-text-${apiStatus}`}>
              {apiStatus === "connected" ? "Connected" : apiStatus === "failed" ? "Not connected" : "Checking…"}
            </strong>
            <small>{API_BASE}</small>
          </div>
        </div>
        <div className="sidebar-foot">DEMO WORKSPACE <span>·</span> v1.0</div>
      </aside>

      <main id="top" className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>{activeTab}</strong></div>
          <div className="topbar-right"><span className="public-badge"><i /> Public + synthetic data</span><div className="avatar">RC</div></div>
        </header>
        <div className="content-wrap">
          <div className="page-heading">
            <div><div className="eyebrow">AUTOMOTIVE QUALITY INVESTIGATION</div><h1>{activeTab === "Investigate" ? "Investigate an issue" : activeTab}</h1><p>Gather evidence, surface patterns, and prepare an 8D draft for review.</p></div>
            {report && <span className="draft-pill"><i /> Draft available</span>}
          </div>

          {activeTab === "Investigate" && <div className="investigate-layout">
            <Panel title="Start an investigation" eyebrow="Issue intake" className="intake-panel">
              <form onSubmit={investigate} className="form-stack">
                <label htmlFor="issue">Describe the quality issue</label>
                <textarea id="issue" value={issue} onChange={(event) => setIssue(event.target.value)} placeholder="Example: Door-seal defects are spiking on Line 3 at Cedar Falls in May 2025." rows={7} />
                <div className="field-hint">Include the affected part, location, or date range when known. The investigation uses only the details you provide.</div>
                {error && <div className="error-box" role="alert"><strong>Investigation could not run</strong><span>{error}</span></div>}
                <div className="form-footer"><span className="muted">Results are a draft for human review.</span><button className="button button-primary" type="submit" disabled={loading || issue.trim().length < 5}>
                  {loading ? <><span className="spinner" /> Running investigation…</> : <>Run investigation <span aria-hidden="true">→</span></>}
                </button></div>
              </form>
            </Panel>
            <div className="intake-aside">
              <Panel title="What happens next" eyebrow="Investigation flow">
                <ol className="flow-list"><li><span>01</span><div><strong>Triage the issue</strong><small>Identify part, plant, line, and time range.</small></div></li><li><span>02</span><div><strong>Gather evidence</strong><small>Query synthetic quality data and public documents.</small></div></li><li><span>03</span><div><strong>Draft an 8D report</strong><small>Build a cited report for a person to review.</small></div></li></ol>
              </Panel>
              <div className="notice-card"><span className="notice-mark">i</span><p>RootCause is a demonstration. Manufacturing records are synthetic and public documents may not match the issue.</p></div>
            </div>
          </div>}

          {activeTab === "Report" && <ReportView report={report} />}
          {activeTab === "Evidence" && <EvidenceChart report={report} />}
          {activeTab === "Review" && <ReviewView report={report} decision={decision} setDecision={setDecision} comment={comment} setComment={setComment} />}
          {activeTab === "Ask" && <AskView key={activeTab} />}
          {activeTab === "SQL agent" && <SQLView key={activeTab} />}
        </div>
        <footer className="page-footer"><span>RootCause <span>·</span> Quality investigation workspace</span><span>Human review required for all report drafts</span></footer>
      </main>
    </div>
  );
}
