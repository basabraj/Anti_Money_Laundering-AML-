import Papa from "papaparse"

// ─────────────────────────────────────────────────────────────────
// Types — the API contract shared with the frontend
// ─────────────────────────────────────────────────────────────────

export interface AnalyzeSummary {
  totalTransactions: number
  totalVolume: number
  uniqueAccounts: number
  flaggedAccountsCount: number
  flaggedThreshold: number
  totalRows: number
  hasDateColumn: boolean
  hasOptionalColumns: {
    paymentType: boolean
    paymentCurrency: boolean
    receivedCurrency: boolean
    senderLocation: boolean
    receiverLocation: boolean
    isLaundering: boolean
  }
  detectedColumns: {
    sender: string
    receiver: string
    amount: string
    date: string | null
    time: string | null
    paymentType: string | null
    paymentCurrency: string | null
    receivedCurrency: string | null
    senderLocation: string | null
    receiverLocation: string | null
    isLaundering: string | null
  }
  featureNames: string[]
}

export interface AccountResult {
  id: string
  riskScore: number
  flagged: boolean
  avgSent: number
  avgReceived: number
  features: Record<string, number>
}

export interface TransactionRow {
  sender: string
  receiver: string
  amount: number
  date?: string
  time?: string
  paymentType?: string
  paymentCurrency?: string
  receivedCurrency?: string
  senderLocation?: string
  receiverLocation?: string
  isLaundering?: number
}

export interface AnalyzeResponse {
  summary: AnalyzeSummary
  metrics: {
    labelsSource: "real" | "synthetic"
    rocAuc: number | null
    precisionAt: { k: number; value: number }[]
  }
  accounts: AccountResult[]
  riskScoreDistribution: { binStart: number; binEnd: number; count: number }[]
  graph: {
    nodes: { id: string; riskScore: number; flagged: boolean; totalSent: number; totalReceived: number }[]
    links: { source: string; target: string; amount: number }[]
  }
  volumeSeries: { label: string; count: number; totalAmount: number }[]
  distribution: { name: string; value: number }[] | null
  distributionLabel: string | null
  transactionsSample: TransactionRow[]
}

export class AnalyzeError extends Error {}

// ─────────────────────────────────────────────────────────────────
// Column auto-detection (port of find_column / find_optional_column)
// ─────────────────────────────────────────────────────────────────

function findColumn(headers: string[], keywords: string[]): string | null {
  for (const col of headers) {
    const lower = col.toLowerCase()
    for (const key of keywords) {
      if (lower.includes(key)) return col
    }
  }
  return null
}

function normalizeColName(s: string): string {
  return s.toLowerCase().replace(/[\s_]/g, "")
}

function findOptionalColumn(headers: string[], target: string): string | null {
  const targetNorm = normalizeColName(target)
  for (const col of headers) {
    if (normalizeColName(col) === targetNorm) return col
  }
  return null
}

// ─────────────────────────────────────────────────────────────────
// Aggregation helpers (vectorized pandas groupby equivalents)
// ─────────────────────────────────────────────────────────────────

interface ParsedRow {
  sender: string
  receiver: string
  amount: number
  paymentType?: string
  paymentCurrency?: string
  receivedCurrency?: string
  senderLoc?: string
  receiverLoc?: string
  isLaundering?: number
  date?: string
  time?: string
}

function groupByMean(
  rows: ParsedRow[],
  keyFn: (r: ParsedRow) => string,
  valFn: (r: ParsedRow) => number,
): Map<string, number> {
  const sums = new Map<string, number>()
  const counts = new Map<string, number>()
  for (const r of rows) {
    const k = keyFn(r)
    sums.set(k, (sums.get(k) ?? 0) + valFn(r))
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  const out = new Map<string, number>()
  for (const [k, sum] of sums) out.set(k, sum / (counts.get(k) ?? 1))
  return out
}

function groupByNUnique(
  rows: ParsedRow[],
  keyFn: (r: ParsedRow) => string,
  valFn: (r: ParsedRow) => string | undefined,
): Map<string, number> {
  const sets = new Map<string, Set<string>>()
  for (const r of rows) {
    const v = valFn(r)
    if (v === undefined) continue
    const k = keyFn(r)
    if (!sets.has(k)) sets.set(k, new Set())
    sets.get(k)!.add(v)
  }
  const out = new Map<string, number>()
  for (const [k, s] of sets) out.set(k, s.size)
  return out
}

// pandas .add(other, fill_value=0) equivalent
function addMaps(a: Map<string, number>, b: Map<string, number>): Map<string, number> {
  const out = new Map(a)
  for (const [k, v] of b) out.set(k, (out.get(k) ?? 0) + v)
  return out
}

// ─────────────────────────────────────────────────────────────────
// Z-score normalization (sample std, matches pandas .std() default)
// ─────────────────────────────────────────────────────────────────

function zScoreNormalize(features: number[][], n: number, f: number): number[][] {
  const means = new Array(f).fill(0)
  const stds = new Array(f).fill(0)
  for (let j = 0; j < f; j++) {
    let sum = 0
    for (let i = 0; i < n; i++) sum += features[i][j]
    means[j] = sum / n
  }
  for (let j = 0; j < f; j++) {
    let sumSq = 0
    for (let i = 0; i < n; i++) sumSq += (features[i][j] - means[j]) ** 2
    stds[j] = n > 1 ? Math.sqrt(sumSq / (n - 1)) : 0
  }
  const normalized: number[][] = Array.from({ length: n }, () => new Array(f).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < f; j++) {
      normalized[i][j] = (features[i][j] - means[j]) / (stds[j] + 1e-6)
    }
  }
  return normalized
}

// ─────────────────────────────────────────────────────────────────
// Heuristic risk score
//
// app.py's AML_TGNN model is an UNTRAINED nn.Module (random weights on
// every run), so its "scores" are not meaningful to replicate bit-for-bit.
// Instead we compute a deterministic, explainable risk score in [0, 1]:
// a weighted mean of the per-account z-scored features, passed through a
// sigmoid (so the population mean maps to ~0.5, and accounts that are
// further from the population norm — larger amounts, more diverse
// counterparties/locations/currencies, more cross-border activity — score
// closer to 1). This is a heuristic ranking signal, NOT a trained model.
// ─────────────────────────────────────────────────────────────────

const FEATURE_WEIGHTS: Record<string, number> = {
  "Avg Sent Amount": 1.0,
  "Avg Received Amount": 1.0,
  "Payment Type Diversity": 0.5,
  "Sender Currency Diversity": 0.5,
  "Receiver Currency Diversity": 0.5,
  "Sender Location Diversity": 0.5,
  "Receiver Location Diversity": 0.5,
  "Cross-Border Ratio": 0.75,
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

function computeRiskScores(normalized: number[][], featureNames: string[]): number[] {
  const weights = featureNames.map((name) => FEATURE_WEIGHTS[name] ?? 1.0)
  const weightSum = weights.reduce((a, b) => a + b, 0)
  return normalized.map((row) => {
    const weightedMean = row.reduce((acc, v, j) => acc + v * weights[j], 0) / weightSum
    return sigmoid(weightedMean)
  })
}

// ─────────────────────────────────────────────────────────────────
// ROC-AUC (rank-based Mann-Whitney U) and Precision@K
// ─────────────────────────────────────────────────────────────────

function rocAuc(yTrue: number[], yScore: number[]): number | null {
  const n = yTrue.length
  const positives = yTrue.filter((y) => y === 1).length
  const negatives = n - positives
  if (positives === 0 || negatives === 0) return null

  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => yScore[a] - yScore[b])
  const ranks = new Array(n)
  let i = 0
  while (i < n) {
    let j = i
    while (j + 1 < n && yScore[indices[j + 1]] === yScore[indices[i]]) j++
    const avgRank = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) ranks[indices[k]] = avgRank
    i = j + 1
  }

  let sumRanksPos = 0
  for (let idx = 0; idx < n; idx++) if (yTrue[idx] === 1) sumRanksPos += ranks[idx]
  return (sumRanksPos - (positives * (positives + 1)) / 2) / (positives * negatives)
}

function precisionAtK(yTrue: number[], yScore: number[], k: number): number | null {
  if (yTrue.length === 0) return null
  const kClamped = Math.min(k, yTrue.length)
  const indices = Array.from({ length: yTrue.length }, (_, i) => i)
    .sort((a, b) => yScore[b] - yScore[a])
    .slice(0, kClamped)
  const hits = indices.reduce((acc, idx) => acc + yTrue[idx], 0)
  return hits / kClamped
}

// ─────────────────────────────────────────────────────────────────
// Capping constants
// ─────────────────────────────────────────────────────────────────

const TOP_ACCOUNTS_LIMIT = 150
const MAX_GRAPH_LINKS = 500
const TRANSACTIONS_SAMPLE_SIZE = 500
const RISK_HISTOGRAM_BINS = 20
const DEFAULT_THRESHOLD = 0.5
const VOLUME_BUCKET_COUNT = 20

// ─────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────

export function analyzeCsv(csvText: string): AnalyzeResponse {
  const parsed = Papa.parse<Record<string, string>>(csvText, {
    header: true,
    dynamicTyping: false,
    skipEmptyLines: true,
  })

  if (!parsed.data || parsed.data.length === 0) {
    throw new AnalyzeError("CSV is empty or could not be parsed.")
  }

  const headers = parsed.meta.fields ?? []

  const senderCol = findColumn(headers, ["sender", "from", "src", "origin"])
  const receiverCol = findColumn(headers, ["receiver", "to", "dst", "target"])
  const amountCol = findColumn(headers, ["amount", "value", "money", "amt"])

  if (!senderCol || !receiverCol || !amountCol) {
    throw new AnalyzeError(
      "CSV must contain sender, receiver, and amount columns (any naming allowed).",
    )
  }

  const paymentTypeCol = findOptionalColumn(headers, "payment_type")
  const paymentCurrencyCol = findOptionalColumn(headers, "payment_currency")
  const receivedCurrencyCol = findOptionalColumn(headers, "received_currency")
  const senderLocCol = findOptionalColumn(headers, "sender_bank_location")
  const receiverLocCol = findOptionalColumn(headers, "receiver_bank_location")
  const isLaunderingCol = findOptionalColumn(headers, "is_laundering")
  const dateCol = findOptionalColumn(headers, "date")
  const timeCol = findOptionalColumn(headers, "time")

  // ── Row parsing ──────────────────────────────────────────────
  const rows: ParsedRow[] = []
  for (const raw of parsed.data) {
    const sender = raw[senderCol]?.trim()
    const receiver = raw[receiverCol]?.trim()
    if (!sender || !receiver) continue

    const amountRaw = Number(raw[amountCol])
    const amount = Number.isNaN(amountRaw) ? 0 : amountRaw

    const row: ParsedRow = { sender, receiver, amount }
    if (paymentTypeCol) row.paymentType = raw[paymentTypeCol]
    if (paymentCurrencyCol) row.paymentCurrency = raw[paymentCurrencyCol]
    if (receivedCurrencyCol) row.receivedCurrency = raw[receivedCurrencyCol]
    if (senderLocCol) row.senderLoc = raw[senderLocCol]
    if (receiverLocCol) row.receiverLoc = raw[receiverLocCol]
    if (isLaunderingCol) {
      const v = Number(raw[isLaunderingCol])
      row.isLaundering = v === 1 ? 1 : 0
    }
    if (dateCol) row.date = raw[dateCol]
    if (timeCol) row.time = raw[timeCol]
    rows.push(row)
  }

  if (rows.length === 0) {
    throw new AnalyzeError("CSV has no valid transaction rows (sender/receiver missing).")
  }

  // ── Graph construction ───────────────────────────────────────
  const accountSet = new Set<string>()
  for (const r of rows) {
    accountSet.add(r.sender)
    accountSet.add(r.receiver)
  }
  const accounts = Array.from(accountSet)
  const N = accounts.length

  // ── Per-account feature engineering ──────────────────────────
  const extraColumns: { name: string; col: string | null }[] = [
    { name: "Payment Type Diversity", col: paymentTypeCol },
    { name: "Sender Currency Diversity", col: paymentCurrencyCol },
    { name: "Receiver Currency Diversity", col: receivedCurrencyCol },
    { name: "Sender Location Diversity", col: senderLocCol },
    { name: "Receiver Location Diversity", col: receiverLocCol },
  ]

  const crossBorder = Boolean(senderLocCol && receiverLocCol)

  const featureNames: string[] = ["Avg Sent Amount", "Avg Received Amount"]
  for (const { name, col } of extraColumns) if (col) featureNames.push(name)
  if (crossBorder) featureNames.push("Cross-Border Ratio")

  const featureSeries = new Map<string, Map<string, number>>()
  featureSeries.set(
    "Avg Sent Amount",
    groupByMean(rows, (r) => r.sender, (r) => r.amount),
  )
  featureSeries.set(
    "Avg Received Amount",
    groupByMean(rows, (r) => r.receiver, (r) => r.amount),
  )

  const extraValueGetters: Record<string, (r: ParsedRow) => string | undefined> = {
    "Payment Type Diversity": (r) => r.paymentType,
    "Sender Currency Diversity": (r) => r.paymentCurrency,
    "Receiver Currency Diversity": (r) => r.receivedCurrency,
    "Sender Location Diversity": (r) => r.senderLoc,
    "Receiver Location Diversity": (r) => r.receiverLoc,
  }

  for (const { name, col } of extraColumns) {
    if (!col) continue
    const valFn = extraValueGetters[name]
    const senderSide = groupByNUnique(rows, (r) => r.sender, valFn)
    const receiverSide = groupByNUnique(rows, (r) => r.receiver, valFn)
    featureSeries.set(name, addMaps(senderSide, receiverSide))
  }

  if (crossBorder) {
    const crossBorderOf = (r: ParsedRow) => (r.senderLoc !== r.receiverLoc ? 1 : 0)
    const senderAvg = groupByMean(rows, (r) => r.sender, crossBorderOf)
    const receiverAvg = groupByMean(rows, (r) => r.receiver, crossBorderOf)
    const combined = addMaps(senderAvg, receiverAvg)
    for (const [k, v] of combined) combined.set(k, v / 2)
    featureSeries.set("Cross-Border Ratio", combined)
  }

  const F = featureNames.length
  const features: number[][] = Array.from({ length: N }, () => new Array(F).fill(0))
  for (let i = 0; i < N; i++) {
    const acc = accounts[i]
    for (let j = 0; j < F; j++) {
      const series = featureSeries.get(featureNames[j])!
      const val = series.get(acc) ?? 0
      features[i][j] = Number.isNaN(val) ? 0 : val
    }
  }

  const normalized = zScoreNormalize(features, N, F)
  const riskScores = computeRiskScores(normalized, featureNames)

  const flaggedMask = riskScores.map((s) => s > DEFAULT_THRESHOLD)
  const flaggedAccountsCount = flaggedMask.filter(Boolean).length

  // ── Evaluation labels and metrics ────────────────────────────
  let evalLabels: number[]
  let labelsSource: "real" | "synthetic"

  if (isLaunderingCol) {
    const launderingAccounts = new Set<string>()
    for (const r of rows) {
      if (r.isLaundering === 1) {
        launderingAccounts.add(r.sender)
        launderingAccounts.add(r.receiver)
      }
    }
    evalLabels = accounts.map((a) => (launderingAccounts.has(a) ? 1 : 0))
    labelsSource = "real"
  } else {
    const sentIdx = featureNames.indexOf("Avg Sent Amount")
    const recvIdx = featureNames.indexOf("Avg Received Amount")
    evalLabels = normalized.map((row) => (row[sentIdx] + row[recvIdx] > 0 ? 1 : 0))
    labelsSource = "synthetic"
  }

  const rocAucValue = rocAuc(evalLabels, riskScores)
  const precisionAt: { k: number; value: number }[] = []
  if (rocAucValue !== null) {
    for (const k of [5, 10, 20]) {
      const p = precisionAtK(evalLabels, riskScores, k)
      if (p !== null) precisionAt.push({ k, value: p })
    }
  }

  // ── Account totals (for graph node sizing) ───────────────────
  const totalSentMap = new Map<string, number>()
  const totalReceivedMap = new Map<string, number>()
  for (const r of rows) {
    totalSentMap.set(r.sender, (totalSentMap.get(r.sender) ?? 0) + r.amount)
    totalReceivedMap.set(r.receiver, (totalReceivedMap.get(r.receiver) ?? 0) + r.amount)
  }

  // ── Top accounts (capped) ────────────────────────────────────
  const accountOrder = Array.from({ length: N }, (_, i) => i).sort(
    (a, b) => riskScores[b] - riskScores[a],
  )
  const topIndices = accountOrder.slice(0, TOP_ACCOUNTS_LIMIT)

  const accountResults: AccountResult[] = topIndices.map((i) => {
    const acc = accounts[i]
    const accFeatures: Record<string, number> = {}
    for (let j = 0; j < F; j++) accFeatures[featureNames[j]] = features[i][j]
    return {
      id: acc,
      riskScore: riskScores[i],
      flagged: riskScores[i] > DEFAULT_THRESHOLD,
      avgSent: featureSeries.get("Avg Sent Amount")!.get(acc) ?? 0,
      avgReceived: featureSeries.get("Avg Received Amount")!.get(acc) ?? 0,
      features: accFeatures,
    }
  })

  // ── Risk score distribution (full dataset) ───────────────────
  const riskScoreDistribution: { binStart: number; binEnd: number; count: number }[] = []
  for (let b = 0; b < RISK_HISTOGRAM_BINS; b++) {
    riskScoreDistribution.push({ binStart: b / RISK_HISTOGRAM_BINS, binEnd: (b + 1) / RISK_HISTOGRAM_BINS, count: 0 })
  }
  for (const score of riskScores) {
    let binIdx = Math.floor(score * RISK_HISTOGRAM_BINS)
    if (binIdx >= RISK_HISTOGRAM_BINS) binIdx = RISK_HISTOGRAM_BINS - 1
    if (binIdx < 0) binIdx = 0
    riskScoreDistribution[binIdx].count++
  }

  // ── Transaction graph (capped) ────────────────────────────────
  const topAccountSet = new Set(topIndices.map((i) => accounts[i]))
  const graphNodes = topIndices.map((i) => {
    const acc = accounts[i]
    return {
      id: acc,
      riskScore: riskScores[i],
      flagged: riskScores[i] > DEFAULT_THRESHOLD,
      totalSent: totalSentMap.get(acc) ?? 0,
      totalReceived: totalReceivedMap.get(acc) ?? 0,
    }
  })

  const candidateLinks = rows
    .filter((r) => topAccountSet.has(r.sender) && topAccountSet.has(r.receiver))
    .map((r) => ({ source: r.sender, target: r.receiver, amount: r.amount }))
  candidateLinks.sort((a, b) => b.amount - a.amount)
  const graphLinks = candidateLinks.slice(0, MAX_GRAPH_LINKS)

  // ── Volume series (full dataset) ─────────────────────────────
  let volumeSeries: { label: string; count: number; totalAmount: number }[] = []
  if (dateCol) {
    const byDate = new Map<string, { count: number; totalAmount: number }>()
    for (const r of rows) {
      const key = r.date ?? "unknown"
      const entry = byDate.get(key) ?? { count: 0, totalAmount: 0 }
      entry.count++
      entry.totalAmount += r.amount
      byDate.set(key, entry)
    }
    volumeSeries = Array.from(byDate.entries())
      .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .map(([label, v]) => ({ label, count: v.count, totalAmount: v.totalAmount }))
  } else {
    const bucketSize = Math.ceil(rows.length / VOLUME_BUCKET_COUNT)
    for (let start = 0; start < rows.length; start += bucketSize) {
      const end = Math.min(start + bucketSize, rows.length)
      let count = 0
      let totalAmount = 0
      for (let i = start; i < end; i++) {
        count++
        totalAmount += rows[i].amount
      }
      volumeSeries.push({ label: `${start + 1}-${end}`, count, totalAmount })
    }
  }

  // ── Distribution chart (first available categorical column) ──
  let distribution: { name: string; value: number }[] | null = null
  let distributionLabel: string | null = null
  const distributionCandidates: { label: string; col: string | null; getter: (r: ParsedRow) => string | undefined }[] = [
    { label: "Payment Type", col: paymentTypeCol, getter: (r) => r.paymentType },
    { label: "Payment Currency", col: paymentCurrencyCol, getter: (r) => r.paymentCurrency },
    { label: "Received Currency", col: receivedCurrencyCol, getter: (r) => r.receivedCurrency },
  ]
  for (const candidate of distributionCandidates) {
    if (!candidate.col) continue
    const counts = new Map<string, number>()
    for (const r of rows) {
      const v = candidate.getter(r)
      if (v === undefined || v === "") continue
      counts.set(v, (counts.get(v) ?? 0) + 1)
    }
    if (counts.size === 0) continue
    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])
    const top = sorted.slice(0, 10)
    const rest = sorted.slice(10)
    const result = top.map(([name, value]) => ({ name, value }))
    if (rest.length > 0) {
      const otherTotal = rest.reduce((acc, [, v]) => acc + v, 0)
      result.push({ name: "Other", value: otherTotal })
    }
    distribution = result
    distributionLabel = candidate.label
    break
  }

  // ── Transactions sample ───────────────────────────────────────
  const transactionsSample: TransactionRow[] = rows.slice(0, TRANSACTIONS_SAMPLE_SIZE).map((r) => ({
    sender: r.sender,
    receiver: r.receiver,
    amount: r.amount,
    ...(r.date !== undefined ? { date: r.date } : {}),
    ...(r.time !== undefined ? { time: r.time } : {}),
    ...(r.paymentType !== undefined ? { paymentType: r.paymentType } : {}),
    ...(r.paymentCurrency !== undefined ? { paymentCurrency: r.paymentCurrency } : {}),
    ...(r.receivedCurrency !== undefined ? { receivedCurrency: r.receivedCurrency } : {}),
    ...(r.senderLoc !== undefined ? { senderLocation: r.senderLoc } : {}),
    ...(r.receiverLoc !== undefined ? { receiverLocation: r.receiverLoc } : {}),
    ...(r.isLaundering !== undefined ? { isLaundering: r.isLaundering } : {}),
  }))

  // ── Summary ────────────────────────────────────────────────────
  const totalVolume = rows.reduce((acc, r) => acc + r.amount, 0)

  const summary: AnalyzeSummary = {
    totalTransactions: rows.length,
    totalVolume,
    uniqueAccounts: N,
    flaggedAccountsCount,
    flaggedThreshold: DEFAULT_THRESHOLD,
    totalRows: rows.length,
    hasDateColumn: Boolean(dateCol),
    hasOptionalColumns: {
      paymentType: Boolean(paymentTypeCol),
      paymentCurrency: Boolean(paymentCurrencyCol),
      receivedCurrency: Boolean(receivedCurrencyCol),
      senderLocation: Boolean(senderLocCol),
      receiverLocation: Boolean(receiverLocCol),
      isLaundering: Boolean(isLaunderingCol),
    },
    detectedColumns: {
      sender: senderCol,
      receiver: receiverCol,
      amount: amountCol,
      date: dateCol,
      time: timeCol,
      paymentType: paymentTypeCol,
      paymentCurrency: paymentCurrencyCol,
      receivedCurrency: receivedCurrencyCol,
      senderLocation: senderLocCol,
      receiverLocation: receiverLocCol,
      isLaundering: isLaunderingCol,
    },
    featureNames,
  }

  return {
    summary,
    metrics: {
      labelsSource,
      rocAuc: rocAucValue,
      precisionAt,
    },
    accounts: accountResults,
    riskScoreDistribution,
    graph: {
      nodes: graphNodes,
      links: graphLinks,
    },
    volumeSeries,
    distribution,
    distributionLabel,
    transactionsSample,
  }
}

// ─────────────────────────────────────────────────────────────────
// Streaming entry point (client-side, large files)
//
// Single-pass, account-level aggregation so memory stays O(unique
// accounts) instead of O(rows) — lets the browser analyze CSVs far
// larger than Vercel's 4.5MB serverless request body limit, since the
// file never needs to be uploaded at all.
// ─────────────────────────────────────────────────────────────────

const VOLUME_FINE_BUCKET_SIZE = 1000
const CANDIDATE_LINK_LIMIT = 2000
const CANDIDATE_LINK_TRIM_AT = 4000

export function analyzeCsvStream(
  input: File | string,
  onProgress?: (fraction: number) => void,
): Promise<AnalyzeResponse> {
  return new Promise((resolve, reject) => {
    let settled = false
    const fail = (err: unknown) => {
      if (settled) return
      settled = true
      reject(err)
    }
    const succeed = (response: AnalyzeResponse) => {
      if (settled) return
      settled = true
      if (onProgress) onProgress(1)
      resolve(response)
    }

    const totalSize = typeof input === "string" ? input.length : input.size

    // ── Aggregation state (all O(unique accounts), not O(rows)) ────
    const accountSet = new Set<string>()
    const sentSum = new Map<string, number>()
    const sentCount = new Map<string, number>()
    const recvSum = new Map<string, number>()
    const recvCount = new Map<string, number>()
    const totalSentMap = new Map<string, number>()
    const totalReceivedMap = new Map<string, number>()
    const senderDiversitySets = new Map<string, Map<string, Set<string>>>()
    const receiverDiversitySets = new Map<string, Map<string, Set<string>>>()
    const senderCBSum = new Map<string, number>()
    const senderCBCount = new Map<string, number>()
    const receiverCBSum = new Map<string, number>()
    const receiverCBCount = new Map<string, number>()
    const launderingAccounts = new Set<string>()
    const byDate = new Map<string, { count: number; totalAmount: number }>()
    const fineBuckets: { count: number; totalAmount: number }[] = []
    const distributionCounts = new Map<string, Map<string, number>>()
    const transactionsSample: TransactionRow[] = []
    let candidateLinks: { source: string; target: string; amount: number }[] = []
    let rowCount = 0
    let totalAmountSum = 0

    // Resolved once the header row has been seen.
    let featureNames: string[] = []
    let dateCol: string | null = null
    let timeCol: string | null = null
    let paymentTypeCol: string | null = null
    let paymentCurrencyCol: string | null = null
    let receivedCurrencyCol: string | null = null
    let senderLocCol: string | null = null
    let receiverLocCol: string | null = null
    let isLaunderingCol: string | null = null
    let senderCol: string | null = null
    let receiverCol: string | null = null
    let amountCol: string | null = null

    const trimCandidateLinks = () => {
      if (candidateLinks.length > CANDIDATE_LINK_TRIM_AT) {
        candidateLinks.sort((a, b) => b.amount - a.amount)
        candidateLinks.length = CANDIDATE_LINK_LIMIT
      }
    }

    const addToSet = (
      store: Map<string, Map<string, Set<string>>>,
      feature: string,
      account: string,
      value: string,
    ) => {
      let perAccount = store.get(feature)
      if (!perAccount) {
        perAccount = new Map()
        store.set(feature, perAccount)
      }
      let set = perAccount.get(account)
      if (!set) {
        set = new Set()
        perAccount.set(account, set)
      }
      set.add(value)
    }

    // Replaced with the real row processor once the header is resolved.
    let processRow: (raw: Record<string, string>) => void = () => {}

    const resolveColumns = (headers: string[]): boolean => {
      senderCol = findColumn(headers, ["sender", "from", "src", "origin"])
      receiverCol = findColumn(headers, ["receiver", "to", "dst", "target"])
      amountCol = findColumn(headers, ["amount", "value", "money", "amt"])
      if (!senderCol || !receiverCol || !amountCol) {
        fail(new AnalyzeError("CSV must contain sender, receiver, and amount columns (any naming allowed)."))
        return false
      }
      const sCol = senderCol
      const rCol = receiverCol
      const aCol = amountCol

      paymentTypeCol = findOptionalColumn(headers, "payment_type")
      paymentCurrencyCol = findOptionalColumn(headers, "payment_currency")
      receivedCurrencyCol = findOptionalColumn(headers, "received_currency")
      senderLocCol = findOptionalColumn(headers, "sender_bank_location")
      receiverLocCol = findOptionalColumn(headers, "receiver_bank_location")
      isLaunderingCol = findOptionalColumn(headers, "is_laundering")
      dateCol = findOptionalColumn(headers, "date")
      timeCol = findOptionalColumn(headers, "time")

      // ── Diversity feature getters (Payment Type / currencies / locations) ──
      const diversityFeatures: { name: string; getter: (r: Record<string, string>) => string }[] = []
      if (paymentTypeCol) {
        const c = paymentTypeCol
        diversityFeatures.push({ name: "Payment Type Diversity", getter: (r) => r[c] })
      }
      if (paymentCurrencyCol) {
        const c = paymentCurrencyCol
        diversityFeatures.push({ name: "Sender Currency Diversity", getter: (r) => r[c] })
      }
      if (receivedCurrencyCol) {
        const c = receivedCurrencyCol
        diversityFeatures.push({ name: "Receiver Currency Diversity", getter: (r) => r[c] })
      }
      if (senderLocCol) {
        const c = senderLocCol
        diversityFeatures.push({ name: "Sender Location Diversity", getter: (r) => r[c] })
      }
      if (receiverLocCol) {
        const c = receiverLocCol
        diversityFeatures.push({ name: "Receiver Location Diversity", getter: (r) => r[c] })
      }

      // ── Cross-border ratio getter ────────────────────────────────
      let crossBorderGetter: ((r: Record<string, string>) => number) | null = null
      if (senderLocCol && receiverLocCol) {
        const sc = senderLocCol
        const rc = receiverLocCol
        crossBorderGetter = (r) => (r[sc] !== r[rc] ? 1 : 0)
      }

      // ── is_laundering getter ─────────────────────────────────────
      let isLaunderingGetter: ((r: Record<string, string>) => number) | null = null
      if (isLaunderingCol) {
        const c = isLaunderingCol
        isLaunderingGetter = (r) => (Number(r[c]) === 1 ? 1 : 0)
      }

      // ── Date getter (drives volumeSeries grouping) ────────────────
      let dateGetter: ((r: Record<string, string>) => string) | null = null
      if (dateCol) {
        const c = dateCol
        dateGetter = (r) => r[c] ?? "unknown"
      }

      // ── Distribution candidates (Payment Type / currencies) ────────
      const distributionConfig: { label: string; getter: (r: Record<string, string>) => string | undefined }[] = []
      if (paymentTypeCol) {
        const c = paymentTypeCol
        distributionConfig.push({ label: "Payment Type", getter: (r) => r[c] })
      }
      if (paymentCurrencyCol) {
        const c = paymentCurrencyCol
        distributionConfig.push({ label: "Payment Currency", getter: (r) => r[c] })
      }
      if (receivedCurrencyCol) {
        const c = receivedCurrencyCol
        distributionConfig.push({ label: "Received Currency", getter: (r) => r[c] })
      }

      // ── Transactions sample field extractors ───────────────────────
      const sampleExtractors: ((r: Record<string, string>) => Partial<TransactionRow>)[] = []
      if (dateCol) {
        const c = dateCol
        sampleExtractors.push((r) => ({ date: r[c] }))
      }
      if (timeCol) {
        const c = timeCol
        sampleExtractors.push((r) => ({ time: r[c] }))
      }
      if (paymentTypeCol) {
        const c = paymentTypeCol
        sampleExtractors.push((r) => ({ paymentType: r[c] }))
      }
      if (paymentCurrencyCol) {
        const c = paymentCurrencyCol
        sampleExtractors.push((r) => ({ paymentCurrency: r[c] }))
      }
      if (receivedCurrencyCol) {
        const c = receivedCurrencyCol
        sampleExtractors.push((r) => ({ receivedCurrency: r[c] }))
      }
      if (senderLocCol) {
        const c = senderLocCol
        sampleExtractors.push((r) => ({ senderLocation: r[c] }))
      }
      if (receiverLocCol) {
        const c = receiverLocCol
        sampleExtractors.push((r) => ({ receiverLocation: r[c] }))
      }
      if (isLaunderingCol) {
        const c = isLaunderingCol
        sampleExtractors.push((r) => ({ isLaundering: Number(r[c]) === 1 ? 1 : 0 }))
      }

      featureNames = ["Avg Sent Amount", "Avg Received Amount"]
      for (const f of diversityFeatures) featureNames.push(f.name)
      if (crossBorderGetter) featureNames.push("Cross-Border Ratio")

      // ── Per-row processor ───────────────────────────────────────
      processRow = (raw: Record<string, string>) => {
        const sender = raw[sCol]?.trim()
        const receiver = raw[rCol]?.trim()
        if (!sender || !receiver) return

        const amountRaw = Number(raw[aCol])
        const amount = Number.isNaN(amountRaw) ? 0 : amountRaw

        accountSet.add(sender)
        accountSet.add(receiver)

        sentSum.set(sender, (sentSum.get(sender) ?? 0) + amount)
        sentCount.set(sender, (sentCount.get(sender) ?? 0) + 1)
        recvSum.set(receiver, (recvSum.get(receiver) ?? 0) + amount)
        recvCount.set(receiver, (recvCount.get(receiver) ?? 0) + 1)
        totalSentMap.set(sender, (totalSentMap.get(sender) ?? 0) + amount)
        totalReceivedMap.set(receiver, (totalReceivedMap.get(receiver) ?? 0) + amount)

        for (const feature of diversityFeatures) {
          const value = feature.getter(raw)
          addToSet(senderDiversitySets, feature.name, sender, value)
          addToSet(receiverDiversitySets, feature.name, receiver, value)
        }

        if (crossBorderGetter) {
          const cb = crossBorderGetter(raw)
          senderCBSum.set(sender, (senderCBSum.get(sender) ?? 0) + cb)
          senderCBCount.set(sender, (senderCBCount.get(sender) ?? 0) + 1)
          receiverCBSum.set(receiver, (receiverCBSum.get(receiver) ?? 0) + cb)
          receiverCBCount.set(receiver, (receiverCBCount.get(receiver) ?? 0) + 1)
        }

        if (isLaunderingGetter && isLaunderingGetter(raw) === 1) {
          launderingAccounts.add(sender)
          launderingAccounts.add(receiver)
        }

        if (dateGetter) {
          const key = dateGetter(raw)
          const entry = byDate.get(key) ?? { count: 0, totalAmount: 0 }
          entry.count++
          entry.totalAmount += amount
          byDate.set(key, entry)
        } else {
          const bucketIdx = Math.floor(rowCount / VOLUME_FINE_BUCKET_SIZE)
          const entry = fineBuckets[bucketIdx] ?? { count: 0, totalAmount: 0 }
          entry.count++
          entry.totalAmount += amount
          fineBuckets[bucketIdx] = entry
        }

        for (const candidate of distributionConfig) {
          const v = candidate.getter(raw)
          if (v === undefined || v === "") continue
          let counts = distributionCounts.get(candidate.label)
          if (!counts) {
            counts = new Map()
            distributionCounts.set(candidate.label, counts)
          }
          counts.set(v, (counts.get(v) ?? 0) + 1)
        }

        if (transactionsSample.length < TRANSACTIONS_SAMPLE_SIZE) {
          let extra: Partial<TransactionRow> = {}
          for (const ext of sampleExtractors) extra = { ...extra, ...ext(raw) }
          transactionsSample.push({ sender, receiver, amount, ...extra })
        }

        candidateLinks.push({ source: sender, target: receiver, amount })
        trimCandidateLinks()

        rowCount++
        totalAmountSum += amount
      }

      return true
    }

    const finalize = (): AnalyzeResponse => {
      const accounts = Array.from(accountSet)
      const N = accounts.length
      const F = featureNames.length
      const sentIdx = featureNames.indexOf("Avg Sent Amount")
      const recvIdx = featureNames.indexOf("Avg Received Amount")

      const features: number[][] = Array.from({ length: N }, () => new Array(F).fill(0))
      for (let i = 0; i < N; i++) {
        const acc = accounts[i]
        for (let j = 0; j < F; j++) {
          const name = featureNames[j]
          let val: number
          if (name === "Avg Sent Amount") {
            const c = sentCount.get(acc)
            val = c ? (sentSum.get(acc) ?? 0) / c : 0
          } else if (name === "Avg Received Amount") {
            const c = recvCount.get(acc)
            val = c ? (recvSum.get(acc) ?? 0) / c : 0
          } else if (name === "Cross-Border Ratio") {
            const sc = senderCBCount.get(acc)
            const rc = receiverCBCount.get(acc)
            const senderAvg = sc ? (senderCBSum.get(acc) ?? 0) / sc : 0
            const receiverAvg = rc ? (receiverCBSum.get(acc) ?? 0) / rc : 0
            val = (senderAvg + receiverAvg) / 2
          } else {
            const senderSize = senderDiversitySets.get(name)?.get(acc)?.size ?? 0
            const receiverSize = receiverDiversitySets.get(name)?.get(acc)?.size ?? 0
            val = senderSize + receiverSize
          }
          features[i][j] = Number.isNaN(val) ? 0 : val
        }
      }

      const normalized = zScoreNormalize(features, N, F)
      const riskScores = computeRiskScores(normalized, featureNames)

      const flaggedMask = riskScores.map((s) => s > DEFAULT_THRESHOLD)
      const flaggedAccountsCount = flaggedMask.filter(Boolean).length

      let evalLabels: number[]
      let labelsSource: "real" | "synthetic"
      if (isLaunderingCol) {
        evalLabels = accounts.map((a) => (launderingAccounts.has(a) ? 1 : 0))
        labelsSource = "real"
      } else {
        evalLabels = normalized.map((row) => (row[sentIdx] + row[recvIdx] > 0 ? 1 : 0))
        labelsSource = "synthetic"
      }

      const rocAucValue = rocAuc(evalLabels, riskScores)
      const precisionAt: { k: number; value: number }[] = []
      if (rocAucValue !== null) {
        for (const k of [5, 10, 20]) {
          const p = precisionAtK(evalLabels, riskScores, k)
          if (p !== null) precisionAt.push({ k, value: p })
        }
      }

      const accountOrder = Array.from({ length: N }, (_, i) => i).sort((a, b) => riskScores[b] - riskScores[a])
      const topIndices = accountOrder.slice(0, TOP_ACCOUNTS_LIMIT)

      const accountResults: AccountResult[] = topIndices.map((i) => {
        const acc = accounts[i]
        const accFeatures: Record<string, number> = {}
        for (let j = 0; j < F; j++) accFeatures[featureNames[j]] = features[i][j]
        return {
          id: acc,
          riskScore: riskScores[i],
          flagged: riskScores[i] > DEFAULT_THRESHOLD,
          avgSent: features[i][sentIdx],
          avgReceived: features[i][recvIdx],
          features: accFeatures,
        }
      })

      const riskScoreDistribution: { binStart: number; binEnd: number; count: number }[] = []
      for (let b = 0; b < RISK_HISTOGRAM_BINS; b++) {
        riskScoreDistribution.push({ binStart: b / RISK_HISTOGRAM_BINS, binEnd: (b + 1) / RISK_HISTOGRAM_BINS, count: 0 })
      }
      for (const score of riskScores) {
        let binIdx = Math.floor(score * RISK_HISTOGRAM_BINS)
        if (binIdx >= RISK_HISTOGRAM_BINS) binIdx = RISK_HISTOGRAM_BINS - 1
        if (binIdx < 0) binIdx = 0
        riskScoreDistribution[binIdx].count++
      }

      const topAccountSet = new Set(topIndices.map((i) => accounts[i]))
      const graphNodes = topIndices.map((i) => {
        const acc = accounts[i]
        return {
          id: acc,
          riskScore: riskScores[i],
          flagged: riskScores[i] > DEFAULT_THRESHOLD,
          totalSent: totalSentMap.get(acc) ?? 0,
          totalReceived: totalReceivedMap.get(acc) ?? 0,
        }
      })

      trimCandidateLinks()
      const graphLinks = candidateLinks
        .filter((l) => topAccountSet.has(l.source) && topAccountSet.has(l.target))
        .sort((a, b) => b.amount - a.amount)
        .slice(0, MAX_GRAPH_LINKS)

      let volumeSeries: { label: string; count: number; totalAmount: number }[] = []
      if (dateCol) {
        volumeSeries = Array.from(byDate.entries())
          .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
          .map(([label, v]) => ({ label, count: v.count, totalAmount: v.totalAmount }))
      } else {
        const numFine = fineBuckets.length
        const mergeFactor = Math.max(1, Math.ceil(numFine / VOLUME_BUCKET_COUNT))
        for (let i = 0; i < numFine; i += mergeFactor) {
          let count = 0
          let totalAmount = 0
          for (let j = i; j < Math.min(i + mergeFactor, numFine); j++) {
            const bucket = fineBuckets[j]
            if (!bucket) continue
            count += bucket.count
            totalAmount += bucket.totalAmount
          }
          const startRow = i * VOLUME_FINE_BUCKET_SIZE + 1
          const endRow = Math.min((i + mergeFactor) * VOLUME_FINE_BUCKET_SIZE, rowCount)
          volumeSeries.push({ label: `${startRow}-${endRow}`, count, totalAmount })
        }
      }

      let distribution: { name: string; value: number }[] | null = null
      let distributionLabel: string | null = null
      for (const [label, counts] of distributionCounts) {
        if (counts.size === 0) continue
        const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])
        const top = sorted.slice(0, 10)
        const rest = sorted.slice(10)
        const result = top.map(([name, value]) => ({ name, value }))
        if (rest.length > 0) {
          const otherTotal = rest.reduce((acc, [, v]) => acc + v, 0)
          result.push({ name: "Other", value: otherTotal })
        }
        distribution = result
        distributionLabel = label
        break
      }

      const summary: AnalyzeSummary = {
        totalTransactions: rowCount,
        totalVolume: totalAmountSum,
        uniqueAccounts: N,
        flaggedAccountsCount,
        flaggedThreshold: DEFAULT_THRESHOLD,
        totalRows: rowCount,
        hasDateColumn: Boolean(dateCol),
        hasOptionalColumns: {
          paymentType: Boolean(paymentTypeCol),
          paymentCurrency: Boolean(paymentCurrencyCol),
          receivedCurrency: Boolean(receivedCurrencyCol),
          senderLocation: Boolean(senderLocCol),
          receiverLocation: Boolean(receiverLocCol),
          isLaundering: Boolean(isLaunderingCol),
        },
        detectedColumns: {
          sender: senderCol as string,
          receiver: receiverCol as string,
          amount: amountCol as string,
          date: dateCol,
          time: timeCol,
          paymentType: paymentTypeCol,
          paymentCurrency: paymentCurrencyCol,
          receivedCurrency: receivedCurrencyCol,
          senderLocation: senderLocCol,
          receiverLocation: receiverLocCol,
          isLaundering: isLaunderingCol,
        },
        featureNames,
      }

      return {
        summary,
        metrics: { labelsSource, rocAuc: rocAucValue, precisionAt },
        accounts: accountResults,
        riskScoreDistribution,
        graph: { nodes: graphNodes, links: graphLinks },
        volumeSeries,
        distribution,
        distributionLabel,
        transactionsSample,
      }
    }

    let columnsResolved = false
    const step = (results: Papa.ParseStepResult<Record<string, string>>, parser: Papa.Parser) => {
      if (settled) return
      if (!columnsResolved) {
        columnsResolved = true
        const headers = results.meta.fields ?? []
        if (!resolveColumns(headers)) {
          parser.abort()
          return
        }
      }
      processRow(results.data)
      if (onProgress && rowCount % 5000 === 0 && totalSize > 0) {
        const cursor = results.meta.cursor ?? 0
        onProgress(Math.min(cursor / totalSize, 1))
      }
    }
    const complete = () => {
      if (settled) return
      if (rowCount === 0) {
        fail(new AnalyzeError("CSV has no valid transaction rows (sender/receiver missing)."))
        return
      }
      try {
        succeed(finalize())
      } catch (err) {
        fail(err)
      }
    }
    const handleError = (err: Error) => fail(err)

    const config = {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true,
      step,
      complete,
      error: handleError,
    }

    // Papa.parse's overloads can't unify a single `step`+`complete`+`error`
    // config across both the `string` and `File` source types; both
    // branches behave identically at runtime.
    Papa.parse<Record<string, string>>(input as any, config as any)
  })
}
