import { NextRequest, NextResponse } from "next/server"
import { analyzeCsv, AnalyzeError } from "./lib"

export const runtime = "nodejs"
export const maxDuration = 60

export async function POST(request: NextRequest) {
  let formData: FormData
  try {
    formData = await request.formData()
  } catch {
    return NextResponse.json({ error: "Request must be multipart/form-data." }, { status: 400 })
  }

  const file = formData.get("file")
  if (!file || !(file instanceof Blob)) {
    return NextResponse.json({ error: "No file uploaded. Attach a CSV under the 'file' field." }, { status: 400 })
  }

  const csvText = await file.text()
  if (!csvText.trim()) {
    return NextResponse.json({ error: "Uploaded file is empty." }, { status: 400 })
  }

  try {
    const result = analyzeCsv(csvText)
    return NextResponse.json(result)
  } catch (err) {
    if (err instanceof AnalyzeError) {
      return NextResponse.json({ error: err.message }, { status: 400 })
    }
    console.error("analyze route failed:", err)
    return NextResponse.json({ error: "Unexpected error while analyzing the file." }, { status: 500 })
  }
}
