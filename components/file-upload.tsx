"use client"

import { useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, FileSpreadsheet } from "lucide-react"

interface FileUploadProps {
  onFileUpload: (file: File) => void
}

export function FileUpload({ onFileUpload }: FileUploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileUpload(acceptedFiles[0])
      }
    },
    [onFileUpload],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
    },
    multiple: false,
  })

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
        isDragActive ? "border-blue-400 bg-blue-50" : "border-slate-300 hover:border-slate-400 hover:bg-slate-50"
      }`}
    >
      <input {...getInputProps()} />
      <div className="space-y-3">
        <div className="mx-auto w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center">
          {isDragActive ? (
            <Upload className="h-6 w-6 text-blue-600" />
          ) : (
            <FileSpreadsheet className="h-6 w-6 text-slate-600" />
          )}
        </div>
        <div>
          <p className="text-sm font-medium text-slate-900">
            {isDragActive ? "Drop your Excel file here" : "Upload Excel File"}
          </p>
          <p className="text-xs text-slate-500 mt-1">Drag and drop or click to select (.xlsx, .xls)</p>
        </div>
      </div>
    </div>
  )
}
