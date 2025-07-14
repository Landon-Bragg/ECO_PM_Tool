"use client"

import { useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { BarChart3, FileSpreadsheet, Users, Settings } from "lucide-react"
import { FileUpload } from "./components/file-upload"
import { SankeyChart } from "./components/sankey-chart"
import { DataPreview } from "./components/data-preview"

// Import necessary libraries for Excel parsing (install if not already)
// For example, using 'xlsx': npm install xlsx
import * as XLSX from "xlsx"

export default function ECOSankeyDashboard() {
  const [file, setFile] = useState<File | null>(null)
  const [ecoNumber, setEcoNumber] = useState("")
  const [data, setData] = useState<any[]>([]) // This will store the parsed Excel data
  const [chartData, setChartData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = useCallback((uploadedFile: File) => {
    setFile(uploadedFile)
    setError(null) // Clear previous errors

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const binaryString = e.target?.result
        const workbook = XLSX.read(binaryString, { type: "binary" })

        // Assuming your data is on the first sheet or a sheet named "Combined"
        const sheetName = workbook.SheetNames[0]
        const worksheet = workbook.Sheets[sheetName]

        // Convert sheet to JSON array
        const jsonData = XLSX.utils.sheet_to_json(worksheet)

        // Perform any initial data cleaning/renaming similar to your Python parse_excel_data
        const processedData = jsonData.map((row: any) => ({
          Change_Order: row["ECO #"] ? String(row["ECO #"]).trim() : "",
          Affected_PN: row["Affected Item"] ? String(row["Affected Item"]).trim() : "",
          Customers: row["Customers"] ? String(row["Customers"]) : "",
          PMs: row["PMs"] ? String(row["PMs"]) : "",
        }))

        setData(processedData)
      } catch (err: any) {
        console.error("Error parsing Excel file:", err)
        setError(
          `Error parsing Excel file: ${err.message}. Please ensure it's a valid Excel file with expected columns.`,
        )
        setData([])
      }
    }
    reader.readAsBinaryString(uploadedFile)
  }, [])

  const generateSankey = useCallback(async () => {
    if (!file || !ecoNumber.trim()) return

    setIsLoading(true)
    setError(null) // Clear previous errors

    try {
      // Filter data for the specific ECO number
      const filteredData = data.filter((record) => record.Change_Order === ecoNumber.trim())

      if (filteredData.length === 0) {
        setError(`No data found for ECO number: ${ecoNumber.trim()}`)
        setChartData(null)
        setIsLoading(false)
        return
      }

      // --- Node Levels ---
      const ecoNodes = [ecoNumber.trim()]
      const partNodes = Array.from(new Set(filteredData.map((record) => record.Affected_PN).filter(Boolean)))

      const customerNodes = Array.from(
        new Set(
          filteredData.flatMap((record) =>
            record.Customers
              ? String(record.Customers)
                  .split(",")
                  .map((s: string) => s.trim())
                  .filter(Boolean)
              : [],
          ),
        ),
      )

      const pmNodes = Array.from(
        new Set(
          filteredData
            .flatMap((record) =>
              record.PMs
                ? String(record.PMs)
                    .split(",")
                    .map((s: string) => s.trim())
                    .filter(Boolean)
                : [],
            )
            .filter((pm) => pm !== "#N/A"), // Filter out '#N/A' as in your Python script
        ),
      )

      const allNodes = [...ecoNodes, ...partNodes, ...customerNodes, ...pmNodes]

      const nodeMap = new Map(allNodes.map((node, i) => [node, i]))

      // --- Links ---
      const links: { source: number; target: number; value: number }[] = []

      // ECO -> Parts
      partNodes.forEach((part) => {
        const count = filteredData.filter((record) => record.Affected_PN === part).length
        links.push({ source: nodeMap.get(ecoNumber.trim())!, target: nodeMap.get(part)!, value: count })
      })

      // Parts -> Customers & Customers -> PMs (more complex relationships, simplified for brevity)
      // This part requires more sophisticated logic to replicate your Python's link generation
      // For a true Sankey, you'll need to iterate through your data to create these links
      // based on relationships, counting occurrences.
      // This example is a placeholder and may not exactly match your Python logic's counts/flows
      filteredData.forEach((record) => {
        const part = record.Affected_PN
        const customers = record.Customers
          ? String(record.Customers)
              .split(",")
              .map((s: string) => s.trim())
              .filter(Boolean)
          : []
        const pms = record.PMs
          ? String(record.PMs)
              .split(",")
              .map((s: string) => s.trim())
              .filter(Boolean)
              .filter((pm) => pm !== "#N/A")
          : []

        customers.forEach((customer: string) => {
          if (nodeMap.has(part) && nodeMap.has(customer)) {
            links.push({ source: nodeMap.get(part)!, target: nodeMap.get(customer)!, value: 1 }) // Value can be adjusted
          }
          pms.forEach((pm: string) => {
            if (nodeMap.has(customer) && nodeMap.has(pm)) {
              links.push({ source: nodeMap.get(customer)!, target: nodeMap.get(pm)!, value: 1 }) // Value can be adjusted
            }
          })
        })
      })

      const sankeyChartData = {
        labels: allNodes,
        source: links.map((link) => link.source),
        target: links.map((link) => link.target),
        value: links.map((link) => link.value),
      }

      setChartData(sankeyChartData)
    } catch (err: any) {
      console.error("Error generating Sankey chart data:", err)
      setError(`Failed to generate chart: ${err.message}`)
      setChartData(null)
    } finally {
      setIsLoading(false)
    }
  }, [file, ecoNumber, data])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">ECO Flow Analyzer</h1>
                <p className="text-sm text-slate-500">Engineering Change Order Visualization</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-slate-600">
              <Settings className="h-4 w-4" />
              <span>Dashboard</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* File Upload Card */}
            <Card className="shadow-lg border-0 bg-white">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center space-x-2">
                  <FileSpreadsheet className="h-5 w-5 text-blue-600" />
                  <span>Data Source</span>
                </CardTitle>
                <CardDescription>Upload your Excel file containing ECO data</CardDescription>
              </CardHeader>
              <CardContent>
                <FileUpload onFileUpload={handleFileUpload} />
                {file && (
                  <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-sm text-green-800 font-medium">✓ {file.name} uploaded successfully</p>
                    <p className="text-xs text-green-600 mt-1">{data.length} records loaded</p>
                  </div>
                )}
                {error && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-sm text-red-800 font-medium">❌ Error: {error}</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* ECO Input Card */}
            <Card className="shadow-lg border-0 bg-white">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center space-x-2">
                  <Users className="h-5 w-5 text-green-600" />
                  <span>Analysis Parameters</span>
                </CardTitle>
                <CardDescription>Enter the ECO number to analyze</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="eco-number" className="text-sm font-medium text-slate-700">
                    ECO Number
                  </Label>
                  <Input
                    id="eco-number"
                    placeholder="e.g., C01798"
                    value={ecoNumber}
                    onChange={(e) => setEcoNumber(e.target.value)}
                    className="mt-1"
                  />
                </div>
                <Button
                  onClick={generateSankey}
                  disabled={!file || !ecoNumber.trim() || isLoading}
                  className="w-full bg-blue-600 hover:bg-blue-700"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="h-4 w-4 mr-2" />
                      Generate Sankey Chart
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Data Preview */}
            {data.length > 0 && <DataPreview data={data} />}
          </div>

          {/* Right Panel - Chart */}
          <div className="lg:col-span-2">
            <Card className="shadow-lg border-0 bg-white h-full">
              <CardHeader>
                <CardTitle className="text-xl">
                  {chartData ? `ECO ${ecoNumber} Flow Diagram` : "Sankey Flow Visualization"}
                </CardTitle>
                <CardDescription>
                  {chartData
                    ? "Interactive flow from ECO → Affected Items → Customers → Project Managers"
                    : "Upload data and enter an ECO number to generate the flow diagram"}
                </CardDescription>
              </CardHeader>
              <CardContent className="h-[600px]">
                {chartData ? (
                  <SankeyChart data={chartData} />
                ) : (
                  <div className="h-full flex items-center justify-center bg-slate-50 rounded-lg border-2 border-dashed border-slate-300">
                    <div className="text-center">
                      <BarChart3 className="h-16 w-16 text-slate-400 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-slate-900 mb-2">Ready to Generate Chart</h3>
                      <p className="text-slate-500 max-w-sm">
                        Upload your Excel file and enter an ECO number to create an interactive Sankey diagram
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}
