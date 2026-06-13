// Regression check: analyzeCsvStream() must produce (near-)identical
// results to the original analyzeCsv() for the same CSV input.
//
// Run with: node scripts/test-analyze-stream.mts
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import path from "node:path"

import { analyzeCsv, analyzeCsvStream, type AnalyzeResponse } from "../app/api/analyze/lib.ts"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const datasetsDir = path.resolve(__dirname, "..", "..", "Datasets")

interface Diff {
  path: string
  a: unknown
  b: unknown
}

function compare(a: unknown, b: unknown, path: string, skip: Set<string>, diffs: Diff[], maxDiffs: number) {
  if (diffs.length >= maxDiffs) return
  if (skip.has(path)) return

  if (typeof a === "number" && typeof b === "number") {
    if (Number.isNaN(a) && Number.isNaN(b)) return
    const tol = 1e-9 * Math.max(1, Math.abs(a), Math.abs(b))
    if (Math.abs(a - b) > tol) diffs.push({ path, a, b })
    return
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      diffs.push({ path: `${path}.length`, a: a.length, b: b.length })
      return
    }
    for (let i = 0; i < a.length; i++) compare(a[i], b[i], `${path}[${i}]`, skip, diffs, maxDiffs)
    return
  }

  if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)])
    for (const k of keys) {
      compare((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], path ? `${path}.${k}` : k, skip, diffs, maxDiffs)
    }
    return
  }

  if (a !== b) diffs.push({ path, a, b })
}

async function runCase(name: string, file: string, skip: string[]) {
  const csvText = readFileSync(file, "utf-8")
  const a = analyzeCsv(csvText)
  const b = await analyzeCsvStream(csvText)

  const diffs: Diff[] = []
  compare(a as unknown, b as unknown, "", new Set(skip), diffs, 25)

  console.log(`\n=== ${name} (${path.basename(file)}) ===`)
  console.log(`rows: ${a.summary.totalRows}, accounts: ${a.summary.uniqueAccounts}`)
  console.log(`rocAuc  a=${a.metrics.rocAuc}  b=${b.metrics.rocAuc}`)
  console.log(`volumeSeries entries  a=${a.volumeSeries.length}  b=${b.volumeSeries.length}`)
  console.log(`graph links  a=${a.graph.links.length}  b=${b.graph.links.length}`)

  if (diffs.length === 0) {
    console.log("PASS — no differences (outside skipped paths)")
  } else {
    console.log(`FAIL — ${diffs.length} difference(s) found:`)
    for (const d of diffs) {
      console.log(`  ${d.path}: a=${JSON.stringify(d.a)} b=${JSON.stringify(d.b)}`)
    }
  }
  return diffs.length === 0
}

function sumVolume(r: AnalyzeResponse) {
  return r.volumeSeries.reduce((acc, v) => acc + v.totalAmount, 0)
}

async function main() {
  let allPassed = true

  // Small file, no date column -> exercises bucketed (non-date) volumeSeries
  // and exact graph-link parity (well under the 4000-row trim threshold).
  {
    const file = path.join(datasetsDir, "Aml_800x800.csv")
    const csvText = readFileSync(file, "utf-8")
    const a = analyzeCsv(csvText)
    const b = await analyzeCsvStream(csvText)

    // volumeSeries bucketing differs intentionally (fixed-size streaming
    // buckets vs. row-count-based buckets); check totals instead.
    const aTotal = sumVolume(a)
    const bTotal = sumVolume(b)
    console.log(`\n=== Aml_800x800.csv volumeSeries totals ===`)
    console.log(`a=${aTotal} b=${bTotal} (summary.totalVolume=${a.summary.totalVolume})`)
    if (Math.abs(aTotal - bTotal) > 1e-6 || Math.abs(aTotal - a.summary.totalVolume) > 1e-6) {
      console.log("FAIL — volumeSeries totals don't match")
      allPassed = false
    } else {
      console.log("PASS — volumeSeries totals match")
    }

    const diffs: Diff[] = []
    compare(a as unknown, b as unknown, "", new Set(["volumeSeries"]), diffs, 25)
    console.log(`\n=== Aml_800x800.csv (full comparison, volumeSeries excluded) ===`)
    if (diffs.length === 0) {
      console.log("PASS — no differences")
    } else {
      console.log(`FAIL — ${diffs.length} difference(s):`)
      for (const d of diffs) console.log(`  ${d.path}: a=${JSON.stringify(d.a)} b=${JSON.stringify(d.b)}`)
      allPassed = false
    }
  }

  // Larger file with a Date column -> exercises date-grouped volumeSeries
  // (must match exactly) and the candidate-link trimming approximation for
  // graph.links (excluded from strict comparison, reported informationally).
  {
    const ok = await runCase("sample_30k_full.csv", path.join(datasetsDir, "sample_30k_full.csv"), [
      "graph.links",
    ])
    if (!ok) allPassed = false
  }

  console.log(`\n${allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED"}`)
  process.exit(allPassed ? 0 : 1)
}

main()
