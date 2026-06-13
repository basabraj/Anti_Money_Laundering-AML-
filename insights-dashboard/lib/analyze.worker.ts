// Web Worker entry point: runs analyzeCsvStream() off the main thread so
// large CSVs (up to ~1GB) can be parsed and analyzed entirely in the
// browser, with no upload and no server body-size limit.
import { analyzeCsvStream, AnalyzeError, type AnalyzeResponse } from "@/app/api/analyze/lib"

export interface AnalyzeWorkerRequest {
  file: File
}

export type AnalyzeWorkerResponse =
  | { type: "progress"; fraction: number }
  | { type: "done"; result: AnalyzeResponse }
  | { type: "error"; message: string }

interface WorkerGlobal {
  onmessage: ((event: MessageEvent<AnalyzeWorkerRequest>) => void) | null
  postMessage: (message: AnalyzeWorkerResponse) => void
}

const ctx = self as unknown as WorkerGlobal

ctx.onmessage = (event) => {
  const { file } = event.data
  analyzeCsvStream(file, (fraction) => {
    ctx.postMessage({ type: "progress", fraction })
  })
    .then((result) => {
      ctx.postMessage({ type: "done", result })
    })
    .catch((err: unknown) => {
      const message = err instanceof AnalyzeError || err instanceof Error
        ? err.message
        : "Failed to analyze file."
      ctx.postMessage({ type: "error", message })
    })
}
