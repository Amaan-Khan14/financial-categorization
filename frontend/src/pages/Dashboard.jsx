import { useState, useEffect } from "react";
import { getMetrics } from "../services/api";
import "./Dashboard.css";

function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const metricsData = await getMetrics();
      setMetrics(metricsData);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  const stats = [
    {
      label: "F1 Score",
      value: metrics?.overall_metrics?.f1_macro
        ? `${(metrics.overall_metrics.f1_macro * 100).toFixed(2)}%`
        : "N/A",
      color: "var(--success)",
      description: "Macro F1 Score",
    },
    {
      label: "Accuracy",
      value: metrics?.overall_metrics?.accuracy
        ? `${(metrics.overall_metrics.accuracy * 100).toFixed(2)}%`
        : "N/A",
      color: "var(--info)",
      description: "Overall Accuracy",
    },
    {
      label: "Avg Latency",
      value: metrics?.usage_stats?.avg_latency_ms
        ? `${metrics.usage_stats.avg_latency_ms.toFixed(1)}ms`
        : "N/A",
      color: "var(--primary-purple)",
      description: "Response Time",
    },
  ];

  const performanceMetrics = [
    {
      label: "Balanced Accuracy",
      value: metrics?.overall_metrics?.balanced_accuracy
        ? `${(metrics.overall_metrics.balanced_accuracy * 100).toFixed(2)}%`
        : "N/A",
    },
    {
      label: "F1 Micro",
      value: metrics?.overall_metrics?.f1_micro
        ? `${(metrics.overall_metrics.f1_micro * 100).toFixed(2)}%`
        : "N/A",
    },
    {
      label: "F1 Weighted",
      value: metrics?.overall_metrics?.f1_weighted
        ? `${(metrics.overall_metrics.f1_weighted * 100).toFixed(2)}%`
        : "N/A",
    },
    {
      label: "Categories Supported",
      value: metrics?.overall_metrics?.num_categories || "N/A",
    },
  ];

  const getPerformanceColor = (value, isPercentage = true) => {
    if (!isPercentage) return "var(--primary-purple)";
    const numValue = parseFloat(value);
    if (numValue >= 95) return "var(--success)";
    if (numValue >= 85) return "var(--info)";
    if (numValue >= 75) return "var(--warning)";
    return "var(--error)";
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Dashboard Overview</h2>
        <p className="dashboard-subtitle">
          Real-time system performance and model metrics
        </p>
      </div>

      <div className="stats-grid">
        {stats.map((stat, index) => (
          <div key={index} className="stat-card card">
            <div className="stat-label">{stat.label}</div>
            <div className="stat-value" style={{ color: stat.color }}>
              {stat.value}
            </div>
            <div className="stat-description">{stat.description}</div>
          </div>
        ))}
      </div>

      <div className="dashboard-row">
        <div className="dashboard-section performance-section">
          <h3>Model Performance Metrics</h3>
          <div className="performance-grid">
            {performanceMetrics.map((metric, index) => {
              const isPercentage =
                typeof metric.value === "string" && metric.value.includes("%");
              const isCategory = metric.label === "Categories Supported";

              return (
                <div key={index} className="performance-item card">
                  <div className="performance-label">{metric.label}</div>
                  <div
                    className="performance-value"
                    style={{
                      color: getPerformanceColor(metric.value, !isCategory),
                    }}
                  >
                    {metric.value}
                  </div>
                  {!isCategory && (
                    <div className="performance-bar">
                      <div
                        className="performance-bar-fill"
                        style={{
                          width: isPercentage ? metric.value : "100%",
                          backgroundColor: getPerformanceColor(
                            metric.value,
                            !isCategory,
                          ),
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="dashboard-section system-info-section">
          <h3>System Information</h3>
          <div className="system-info-card card">
            <div className="info-item">
              <div className="info-content">
                <div className="info-label">Model Version</div>
                <div className="info-value">v1.0.0</div>
              </div>
            </div>
            <div className="info-item">
              <div className="info-content">
                <div className="info-label">Last Updated</div>
                <div className="info-value">
                  {new Date().toLocaleDateString()}
                </div>
              </div>
            </div>
            <div className="info-item">
              <div className="info-content">
                <div className="info-label">Status</div>
                <div className="info-value status-active">
                  <span className="status-dot"></span>
                  Active
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="dashboard-section">
        <h3>Key Features</h3>
        <div className="features-grid">
          <div className="feature-card card">
            <div className="feature-badge">AI POWERED</div>
            <h4>Multi-stage Ensemble</h4>
            <p>
              Advanced classification with <strong>96.15% F1-score</strong>{" "}
              across all categories
            </p>
          </div>
          <div className="feature-card card">
            <div className="feature-badge">EXTENSIVE</div>
            <h4>150+ Brands</h4>
            <p>
              Comprehensive coverage of Indian marketplace and fintech brands
            </p>
          </div>
          <div className="feature-card card">
            <div className="feature-badge">SELF-CONTAINED</div>
            <h4>Zero External APIs</h4>
            <p>100% offline inference with no external dependencies</p>
          </div>
          <div className="feature-card card">
            <div className="feature-badge">TRANSPARENT</div>
            <h4>Full Explainability</h4>
            <p>
              LIME-based feature attribution with complete prediction insights
            </p>
          </div>
          <div className="feature-card card">
            <div className="feature-badge">OCR ENABLED</div>
            <h4>Image Classification</h4>
            <p>Receipt & invoice scanning with automatic text extraction</p>
          </div>
          <div className="feature-card card">
            <div className="feature-badge">INSIGHTS</div>
            <h4>Smart Analytics</h4>
            <p>AI-driven insights, anomaly detection, and spending forecasts</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
