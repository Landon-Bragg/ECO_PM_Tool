"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Database } from "lucide-react"

interface DataPreviewProps {
  data: Array<{
    Change_Order: string
    Affected_PN: string
    Customers: string
    PMs: string
  }>
}

export function DataPreview({ data }: DataPreviewProps) {
  const previewData = data.slice(0, 5) // Show first 5 rows

  return (
    <Card className="shadow-lg border-0 bg-white">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center space-x-2">
          <Database className="h-5 w-5 text-purple-600" />
          <span>Data Preview</span>
        </CardTitle>
        <CardDescription>
          First {previewData.length} of {data.length} records
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">ECO #</TableHead>
                <TableHead className="text-xs">Part</TableHead>
                <TableHead className="text-xs">Customers</TableHead>
                <TableHead className="text-xs">PMs</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {previewData.map((row, index) => (
                <TableRow key={index}>
                  <TableCell className="text-xs font-mono">{row.Change_Order}</TableCell>
                  <TableCell className="text-xs font-mono">{row.Affected_PN}</TableCell>
                  <TableCell className="text-xs max-w-32 truncate" title={row.Customers}>
                    {row.Customers}
                  </TableCell>
                  <TableCell className="text-xs">{row.PMs}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}
