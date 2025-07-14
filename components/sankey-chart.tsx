"use client"
import dynamic from "next/dynamic"

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false })

interface SankeyChartProps {
  data: {
    labels: string[]
    source: number[]
    target: number[]
    value: number[]
  }
}

export function SankeyChart({ data }: SankeyChartProps) {
  // Validate data structure
  if (!data || !data.labels || !data.source || !data.target || !data.value) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-50 rounded-lg border-2 border-dashed border-slate-300">
        <div className="text-center">
          <p className="text-slate-500">Invalid data structure for Sankey chart</p>
        </div>
      </div>
    )
  }

  // Color scheme matching the Python version
  const COLORS = {
    ECO: "#1f77b4", // muted blue
    Part: "#ff7f0e", // safety orange
    Customer: "#2ca02c", // cooked asparagus green
    PM: "#d62728", // brick red
  }

  // Assign colors to nodes (simplified - you may need to adjust based on your node levels)
  const node_colors = data.labels.map((label, index) => {
    if (index === 0) return COLORS.ECO // First node is typically ECO
    if (index < data.labels.length * 0.3) return COLORS.Part
    if (index < data.labels.length * 0.7) return COLORS.Customer
    return COLORS.PM
  })

  // Link colors with transparency
  const max_value = Math.max(...data.value)
  const link_colors = data.value.map((v) => `rgba(31,119,180,${0.2 + 0.6 * (v / max_value)})`)

  const plotData = [
    {
      type: "sankey",
      arrangement: "snap",
      node: {
        label: data.labels,
        color: node_colors,
        pad: 15,
        thickness: 25,
        line: { color: "black", width: 0.5 },
        hovertemplate: "%{label}<br>Connections: %{value}<extra></extra>",
      },
      link: {
        source: data.source,
        target: data.target,
        value: data.value,
        color: link_colors,
        hovertemplate: "%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>",
      },
    },
  ]

  const layout = {
    font: { size: 12, family: "Inter, system-ui, sans-serif" },
    margin: { l: 20, r: 20, t: 20, b: 20 },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
  }

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
    responsive: true,
  }

  return (
    <div className="h-full w-full">
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler={true}
      />
    </div>
  )
}
