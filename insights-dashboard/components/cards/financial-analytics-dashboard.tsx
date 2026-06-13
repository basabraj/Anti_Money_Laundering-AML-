"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { motion, AnimatePresence } from "motion/react"
import {
  BarChart3, Share2, ShieldAlert, ArrowLeftRight, ChevronRight,
  Upload, AlertTriangle, DollarSign, Users, Target,
} from "lucide-react"
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Spinner } from "@/components/ui/spinner"
import { Progress } from "@/components/ui/progress"
import {
  Table, TableHeader, TableBody, TableRow, TableHead, TableCell,
} from "@/components/ui/table"
import type { AnalyzeResponse } from "@/app/api/analyze/lib"
import type { AnalyzeWorkerRequest, AnalyzeWorkerResponse } from "@/lib/analyze.worker"

const TransactionGraph = dynamic(
  () => import("./transaction-graph").then((m) => m.default),
  {
    ssr: false,
    loading: () => (
      <div className="h-[500px] flex items-center justify-center text-sm text-muted-foreground">
        Loading graph...
      </div>
    ),
  },
)

// ─── Design tokens ─────────────────────────────────────────────

const CARD_SHADOW =
  "rgba(14, 63, 126, 0.04) 0px 0px 0px 1px, rgba(42, 51, 69, 0.04) 0px 1px 1px -0.5px, rgba(42, 51, 70, 0.04) 0px 3px 3px -1.5px, rgba(42, 51, 70, 0.04) 0px 6px 6px -3px, rgba(14, 63, 126, 0.04) 0px 12px 12px -6px, rgba(14, 63, 126, 0.04) 0px 24px 24px -12px"

const SECTION_MIN_H = "min-h-[calc(100vh-10.5rem)]"

const C = {
  teal: "oklch(0.78 0.16 182)",
  azure: "oklch(0.68 0.14 245)",
  amber: "oklch(0.76 0.14 75)",
  gold: "oklch(0.70 0.16 48)",
  rose: "oklch(0.62 0.22 18)",
  slate: "oklch(0.50 0.02 260)",
  grid: "oklch(0.24 0.01 260)",
  tick: "oklch(0.50 0.015 260)",
}

const DISTRIBUTION_COLORS = [C.teal, C.azure, C.amber, C.rose, C.slate]

const SPRING = { type: "spring" as const, stiffness: 400, damping: 32 }
const EASE_OUT = [0.16, 1, 0.3, 1] as const

// ─── Nav ───────────────────────────────────────────────────────

type SectionId = "overview" | "graph" | "fraud" | "transactions"

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "graph", label: "Transaction Graph", icon: Share2 },
  { id: "fraud", label: "Fraud Accounts", icon: ShieldAlert },
  { id: "transactions", label: "Transactions", icon: ArrowLeftRight },
] as const satisfies readonly { id: SectionId; label: string; icon: React.ElementType }[]

// ─── Sub-Components ─────────────────────────────────────────────

function GlowOrb({ className }: { className?: string }) {
  return <div className={`absolute rounded-full blur-3xl pointer-events-none ${className}`} />
}

function KpiCard({
  label, value, prefix = "", suffix = "", delay = 0, icon: Icon, caption,
}: {
  label: string; value: string; prefix?: string; suffix?: string; delay?: number; icon?: React.ElementType; caption?: string
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay, ease: EASE_OUT }}
      className="relative overflow-hidden rounded-2xl surface-card p-4 lg:p-5 group hover:scale-[1.01] transition-transform duration-300"
      style={{ boxShadow: CARD_SHADOW }}
    >
      <div className="absolute top-0 right-0 w-24 h-24 opacity-[0.03] pointer-events-none">
        {Icon && <Icon className="size-24 -translate-y-4 translate-x-4" />}
      </div>
      <p className="text-[11px] font-semibold tracking-[0.08em] uppercase text-muted-foreground mb-2.5 font-sans">
        {label}
      </p>
      <p className="text-2xl lg:text-3xl font-bold text-foreground font-mono tracking-tighter leading-none">
        {prefix}{value}{suffix}
      </p>
      {caption && (
        <p className="text-[10px] text-muted-foreground/70 mt-3 font-sans">{caption}</p>
      )}
    </motion.div>
  )
}

function ChartTooltipContent({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name: string; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-xl surface-elevated p-3 text-xs backdrop-blur-md" style={{ boxShadow: CARD_SHADOW }}>
      {label && <p className="text-muted-foreground mb-2 font-semibold text-[11px] uppercase tracking-wider font-sans">{label}</p>}
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2 py-0.5">
          <div className="size-2 rounded-full" style={{ backgroundColor: entry.color }} />
          <span className="text-muted-foreground capitalize font-sans">{entry.name}:</span>
          <span className="font-mono font-bold text-foreground">{typeof entry.value === "number" ? entry.value.toLocaleString() : entry.value}</span>
        </div>
      ))}
    </div>
  )
}

function SectionPanel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.15, ease: EASE_OUT }}
      className={`rounded-2xl surface-card p-5 lg:p-6 ${className}`}
      style={{ boxShadow: CARD_SHADOW }}
    >
      {children}
    </motion.div>
  )
}

function SectionHeader({ title, subtitle, children }: { title: string; subtitle: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-5">
      <div>
        <h3 className="text-sm font-bold text-foreground tracking-tight font-display">{title}</h3>
        <p className="text-[11px] text-muted-foreground mt-0.5 font-sans">{subtitle}</p>
      </div>
      {children}
    </div>
  )
}

// ─── Upload prompt ───────────────────────────────────────────────

function UploadPrompt({
  isLoading, progress, error, onFileSelect,
}: {
  isLoading: boolean; progress: number; error: string | null; onFileSelect: (file: File) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0]
    if (file) onFileSelect(file)
  }

  return (
    <div className={`relative flex items-center justify-center ${SECTION_MIN_H}`}>
      {/* Warm frosted-glass atmosphere */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: `radial-gradient(ellipse 1200px 900px at 15% 10%, ${C.amber} 0%, transparent 55%), radial-gradient(ellipse 1100px 1000px at 88% 95%, ${C.gold} 0%, transparent 55%)`,
          opacity: 0.45,
        }}
      />
      <div className="fixed top-[2%] left-[5%] w-[620px] h-[620px] rounded-full opacity-[0.28] blur-[110px] animate-float pointer-events-none" style={{ background: C.amber }} />
      <div className="fixed bottom-[-5%] right-[2%] w-[580px] h-[580px] rounded-full opacity-[0.25] blur-[110px] animate-float pointer-events-none" style={{ background: C.gold, animationDelay: "3s" }} />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.15, ease: EASE_OUT }}
        className="w-full max-w-xl relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 backdrop-blur-2xl p-5 lg:p-6"
        style={{ boxShadow: CARD_SHADOW }}
      >
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setIsDragging(false)
            handleFiles(e.dataTransfer.files)
          }}
          className={`rounded-2xl border-2 border-dashed transition-colors duration-200 ${
            isDragging ? "border-chart-3 bg-chart-3/5" : "border-white/10"
          }`}
        >
          {isLoading ? (
            <div className="flex flex-col items-center gap-3 py-10 text-center px-10 w-full">
              <Spinner className="size-8 text-chart-3" />
              <p className="text-sm font-semibold text-foreground font-display">Analyzing transactions...</p>
              <p className="text-xs text-muted-foreground font-sans">
                Processing in your browser — no upload required, even for very large files.
              </p>
              <div className="w-full max-w-xs mt-2">
                <Progress value={Math.round(progress * 100)} />
                <p className="text-[11px] text-muted-foreground font-mono mt-1.5">{Math.round(progress * 100)}%</p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 py-10 text-center px-6">
              <div className="size-14 rounded-2xl bg-chart-3/12 border border-chart-3/20 flex items-center justify-center">
                <Upload className="size-6 text-chart-3" />
              </div>
              <div>
                <h3 className="text-base font-bold text-foreground font-display">Upload a transaction CSV</h3>
                <p className="text-xs text-muted-foreground mt-1 font-sans max-w-sm">
                  Drop a CSV file here or click to browse. The file is analyzed entirely in your
                  browser — nothing is uploaded, so even multi-gigabyte files work.
                </p>
              </div>
              <button
                onClick={() => inputRef.current?.click()}
                className="mt-2 text-xs font-bold text-chart-3 hover:text-chart-3/80 transition-colors px-4 py-2 rounded-xl bg-chart-3/8 hover:bg-chart-3/14 border border-chart-3/20 font-sans"
              >
                Choose File
              </button>
              <p className="text-[11px] font-semibold font-sans" style={{ color: C.rose }}>
                Maximum file size: 1 GB. Files larger than this cannot be uploaded.
              </p>
              <input
                ref={inputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => { handleFiles(e.target.files); e.target.value = "" }}
              />
            </div>
          )}
        </div>

        {error && (
          <Alert variant="destructive" className="mt-5">
            <AlertTriangle />
            <AlertTitle>Could not analyze file</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="mt-6">
          <h4 className="text-[11px] font-semibold tracking-[0.08em] uppercase text-muted-foreground mb-2 font-sans">
            Expected columns
          </h4>
          <table className="w-full text-xs">
            <tbody>
              <tr className="border-b border-white/10">
                <td className="py-1.5 pr-3 font-semibold text-foreground font-sans whitespace-nowrap">Sender</td>
                <td className="py-1.5 text-muted-foreground font-mono">sender, from, src, origin</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-1.5 pr-3 font-semibold text-foreground font-sans whitespace-nowrap">Receiver</td>
                <td className="py-1.5 text-muted-foreground font-mono">receiver, to, dst, target</td>
              </tr>
              <tr>
                <td className="py-1.5 pr-3 font-semibold text-foreground font-sans whitespace-nowrap">Amount</td>
                <td className="py-1.5 text-muted-foreground font-mono">amount, value, money, amt</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  )
}

// ─── Section: Overview ──────────────────────────────────────────

function OverviewSection({ analysis }: { analysis: AnalyzeResponse }) {
  const { summary, volumeSeries, distribution, distributionLabel } = analysis

  return (
    <div className={`flex flex-col gap-5 ${SECTION_MIN_H}`}>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
        <KpiCard label="Total Transactions" value={summary.totalTransactions.toLocaleString()} delay={0} icon={ArrowLeftRight} />
        <KpiCard label="Total Volume" value={summary.totalVolume.toLocaleString(undefined, { maximumFractionDigits: 0 })} prefix="$" delay={0.06} icon={DollarSign} />
        <KpiCard label="Unique Accounts" value={summary.uniqueAccounts.toLocaleString()} delay={0.12} icon={Users} />
        <KpiCard
          label="Flagged Accounts"
          value={summary.flaggedAccountsCount.toLocaleString()}
          delay={0.18}
          icon={ShieldAlert}
          caption={`at default ${summary.flaggedThreshold.toFixed(2)} threshold`}
        />
      </div>

      <SectionPanel className="relative overflow-hidden">
        <GlowOrb className="w-64 h-64 -top-32 -right-32 bg-primary/10" />
        <SectionHeader title="Transaction Volume" subtitle={summary.hasDateColumn ? "Over time" : "By sequence"} />
        <div className="h-56 lg:h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={volumeSeries} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="volumeGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.teal} stopOpacity={0.25} />
                  <stop offset="50%" stopColor={C.teal} stopOpacity={0.08} />
                  <stop offset="100%" stopColor={C.teal} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: C.tick }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: C.tick }} axisLine={false} tickLine={false} tickFormatter={(v: number) => (v >= 1000 ? `$${(v / 1000).toFixed(0)}k` : `$${v}`)} />
              <Tooltip content={<ChartTooltipContent />} />
              <Area type="monotone" dataKey="totalAmount" stroke={C.teal} strokeWidth={2.5} fill="url(#volumeGrad)" name="volume" animationDuration={1400} animationEasing="ease-out" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </SectionPanel>

      {distribution ? (
        <SectionPanel>
          <SectionHeader title={distributionLabel ?? "Distribution"} subtitle="Breakdown of transactions" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-center">
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={distribution} cx="50%" cy="50%" innerRadius="55%" outerRadius="85%" paddingAngle={3} dataKey="value" nameKey="name" stroke="none" animationDuration={1200} animationEasing="ease-out">
                    {distribution.map((_, i) => <Cell key={i} fill={DISTRIBUTION_COLORS[i % DISTRIBUTION_COLORS.length]} />)}
                  </Pie>
                  <Tooltip content={<ChartTooltipContent />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-col gap-2.5">
              {distribution.map((item, i) => (
                <div key={item.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div className="size-2.5 rounded-full shrink-0" style={{ backgroundColor: DISTRIBUTION_COLORS[i % DISTRIBUTION_COLORS.length] }} />
                    <span className="text-muted-foreground font-sans truncate">{item.name}</span>
                  </div>
                  <span className="font-mono font-bold text-foreground shrink-0 ml-2">{item.value.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </SectionPanel>
      ) : (
        <SectionPanel>
          <SectionHeader title="Distribution" subtitle="Categorical breakdown" />
          <p className="text-sm text-muted-foreground font-sans">
            No categorical columns (Payment Type or Currency) were detected in this dataset.
          </p>
        </SectionPanel>
      )}
    </div>
  )
}

// ─── Section: Transaction Graph ─────────────────────────────────

function TransactionGraphSection({ analysis, riskThreshold }: { analysis: AnalyzeResponse; riskThreshold: number }) {
  return (
    <div className={`flex flex-col gap-5 ${SECTION_MIN_H}`}>
      <SectionPanel>
        <SectionHeader title="Transaction Graph" subtitle="Sender to receiver flows, top 150 accounts by risk score">
          <div className="flex items-center gap-5 text-[11px]">
            <div className="flex items-center gap-2"><div className="size-2.5 rounded-full" style={{ background: C.rose }} /><span className="text-muted-foreground font-sans">Flagged</span></div>
            <div className="flex items-center gap-2"><div className="size-2.5 rounded-full" style={{ background: C.azure }} /><span className="text-muted-foreground font-sans">Clean</span></div>
          </div>
        </SectionHeader>
        <TransactionGraph nodes={analysis.graph.nodes} links={analysis.graph.links} riskThreshold={riskThreshold} />
        <p className="text-[11px] text-muted-foreground mt-4 font-sans">
          Showing the top {analysis.graph.nodes.length} accounts by risk score and up to {analysis.graph.links.length} transactions
          between them. Node size reflects total sent plus received volume.
        </p>
      </SectionPanel>
    </div>
  )
}

// ─── Section: Fraud Accounts ────────────────────────────────────

function FraudAccountsSection({ analysis, riskThreshold, onThresholdChange }: {
  analysis: AnalyzeResponse; riskThreshold: number; onThresholdChange: (value: number) => void
}) {
  const { metrics, accounts, riskScoreDistribution } = analysis

  const flaggedAccounts = useMemo(
    () => accounts.filter((a) => a.riskScore > riskThreshold).sort((a, b) => b.riskScore - a.riskScore),
    [accounts, riskThreshold],
  )

  const histogramData = useMemo(
    () => riskScoreDistribution.map((bin) => ({
      label: `${bin.binStart.toFixed(2)}-${bin.binEnd.toFixed(2)}`,
      count: bin.count,
    })),
    [riskScoreDistribution],
  )

  return (
    <div className={`flex flex-col gap-5 ${SECTION_MIN_H}`}>
      {metrics.rocAuc !== null ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
          <KpiCard label="ROC-AUC" value={metrics.rocAuc.toFixed(3)} delay={0} icon={Target} />
          {metrics.precisionAt.map((p, i) => (
            <KpiCard key={p.k} label={`Precision@${p.k}`} value={p.value.toFixed(3)} delay={0.06 * (i + 1)} icon={Target} />
          ))}
        </div>
      ) : (
        <SectionPanel>
          <p className="text-sm text-muted-foreground font-sans">
            ROC-AUC and Precision@K cannot be computed because evaluation labels contain only one class.
          </p>
        </SectionPanel>
      )}
      <p className="text-[11px] text-muted-foreground -mt-2 font-sans">
        Evaluation labels: {metrics.labelsSource === "real" ? "real Is_laundering ground truth" : "synthetic demo labels"}
      </p>

      <SectionPanel>
        <SectionHeader
          title="Risk Threshold"
          subtitle={`${flaggedAccounts.length} of ${accounts.length} top accounts flagged at ${riskThreshold.toFixed(2)}`}
        />
        <Slider value={[riskThreshold]} min={0} max={1} step={0.01} onValueChange={([v]) => onThresholdChange(v)} />
      </SectionPanel>

      <SectionPanel className="relative overflow-hidden">
        <GlowOrb className="w-48 h-48 -top-24 -left-24 bg-chart-2/8" />
        <SectionHeader title="Risk Score Distribution" subtitle="All accounts, by risk score bucket" />
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: C.tick }} axisLine={false} tickLine={false} interval={1} angle={-45} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11, fill: C.tick }} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip content={<ChartTooltipContent />} />
              <Bar dataKey="count" name="accounts" fill={C.azure} radius={[4, 4, 0, 0]} animationDuration={900} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </SectionPanel>

      <SectionPanel className="!p-0 overflow-hidden">
        <div className="p-5 lg:p-6 border-b border-border/50">
          <SectionHeader title="Flagged Accounts" subtitle="From the top 150 accounts by risk score" />
        </div>
        {flaggedAccounts.length === 0 ? (
          <div className="p-6 text-sm text-muted-foreground font-sans">No accounts exceed the current risk threshold.</div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Account</TableHead>
                <TableHead className="text-right">Risk Score</TableHead>
                <TableHead className="hidden sm:table-cell">Score</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {flaggedAccounts.map((account) => (
                <TableRow key={account.id}>
                  <TableCell className="font-mono font-semibold text-foreground">{account.id}</TableCell>
                  <TableCell className="text-right font-mono font-bold" style={{ color: C.rose }}>{account.riskScore.toFixed(3)}</TableCell>
                  <TableCell className="hidden sm:table-cell">
                    <div className="h-2 rounded-full bg-muted/60 overflow-hidden w-full max-w-40">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(account.riskScore, 1) * 100}%` }}
                        transition={{ duration: 0.8, ease: EASE_OUT }}
                        className="h-full rounded-full"
                        style={{ background: C.rose }}
                      />
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </SectionPanel>
    </div>
  )
}

// ─── Section: Transactions ──────────────────────────────────────

function TransactionsSection({ analysis }: { analysis: AnalyzeResponse }) {
  const { summary, transactionsSample } = analysis
  const avgSize = summary.totalTransactions > 0 ? summary.totalVolume / summary.totalTransactions : 0

  return (
    <div className={`flex flex-col gap-5 ${SECTION_MIN_H}`}>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4">
        <KpiCard label="Total Volume" value={summary.totalVolume.toLocaleString(undefined, { maximumFractionDigits: 0 })} prefix="$" delay={0} icon={DollarSign} />
        <KpiCard label="Transactions" value={summary.totalRows.toLocaleString()} delay={0.06} icon={ArrowLeftRight} />
        <KpiCard label="Avg. Size" value={avgSize.toLocaleString(undefined, { maximumFractionDigits: 2 })} prefix="$" delay={0.12} icon={BarChart3} />
        <KpiCard label="Unique Accounts" value={summary.uniqueAccounts.toLocaleString()} delay={0.18} icon={Users} />
      </div>

      <SectionPanel className="!p-0 overflow-hidden">
        <div className="p-5 lg:p-6 border-b border-border/50">
          <SectionHeader title="Transactions" subtitle={`Showing ${transactionsSample.length.toLocaleString()} of ${summary.totalTransactions.toLocaleString()} transactions`} />
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Sender</TableHead>
              <TableHead>Receiver</TableHead>
              <TableHead className="text-right">Amount</TableHead>
              {summary.hasDateColumn && <TableHead>Date</TableHead>}
              {summary.detectedColumns.time && <TableHead>Time</TableHead>}
              {summary.hasOptionalColumns.paymentType && <TableHead>Payment Type</TableHead>}
              {summary.hasOptionalColumns.isLaundering && <TableHead>Flag</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {transactionsSample.map((tx, i) => (
              <TableRow key={i}>
                <TableCell className="font-mono">{tx.sender}</TableCell>
                <TableCell className="font-mono">{tx.receiver}</TableCell>
                <TableCell className="text-right font-mono font-bold">{tx.amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</TableCell>
                {summary.hasDateColumn && <TableCell className="font-mono text-muted-foreground">{tx.date ?? "—"}</TableCell>}
                {summary.detectedColumns.time && <TableCell className="font-mono text-muted-foreground">{tx.time ?? "—"}</TableCell>}
                {summary.hasOptionalColumns.paymentType && <TableCell className="text-muted-foreground">{tx.paymentType ?? "—"}</TableCell>}
                {summary.hasOptionalColumns.isLaundering && (
                  <TableCell>
                    {tx.isLaundering === 1 ? <Badge variant="destructive">Laundering</Badge> : null}
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </SectionPanel>
    </div>
  )
}

// ─── Main Dashboard ─────────────────────────────────────────────

function renderSection(
  activeSection: SectionId,
  analysis: AnalyzeResponse,
  riskThreshold: number,
  onThresholdChange: (value: number) => void,
) {
  switch (activeSection) {
    case "overview":
      return <OverviewSection analysis={analysis} />
    case "graph":
      return <TransactionGraphSection analysis={analysis} riskThreshold={riskThreshold} />
    case "fraud":
      return <FraudAccountsSection analysis={analysis} riskThreshold={riskThreshold} onThresholdChange={onThresholdChange} />
    case "transactions":
      return <TransactionsSection analysis={analysis} />
  }
}

export default function FinancialAnalyticsDashboard() {
  const [activeSection, setActiveSection] = useState<SectionId>("overview")
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [riskThreshold, setRiskThreshold] = useState(0.5)
  const [fileName, setFileName] = useState<string | null>(null)
  const reuploadInputRef = useRef<HTMLInputElement>(null)
  const workerRef = useRef<Worker | null>(null)

  const handleNavigation = useCallback((sectionId: SectionId) => {
    setActiveSection((prev) => {
      if (sectionId === prev) return prev
      setIsTransitioning(true)
      setTimeout(() => setIsTransitioning(false), 180)
      return sectionId
    })
  }, [])

  const handleFileUpload = useCallback((file: File) => {
    setIsLoading(true)
    setError(null)
    setProgress(0)

    workerRef.current?.terminate()
    const worker = new Worker("/analyze-worker.js")
    workerRef.current = worker

    worker.onmessage = (event: MessageEvent<AnalyzeWorkerResponse>) => {
      const msg = event.data
      switch (msg.type) {
        case "progress":
          setProgress(msg.fraction)
          break
        case "done":
          setAnalysis(msg.result)
          setFileName(file.name)
          setRiskThreshold(msg.result.summary.flaggedThreshold)
          setActiveSection("overview")
          setIsLoading(false)
          worker.terminate()
          workerRef.current = null
          break
        case "error":
          setError(msg.message)
          setIsLoading(false)
          worker.terminate()
          workerRef.current = null
          break
      }
    }
    worker.onerror = () => {
      setError("Failed to analyze file.")
      setIsLoading(false)
      worker.terminate()
      workerRef.current = null
    }

    const request: AnalyzeWorkerRequest = { file }
    worker.postMessage(request)
  }, [])

  useEffect(() => {
    return () => workerRef.current?.terminate()
  }, [])

  const activeNav = useMemo(() => NAV_ITEMS.find((n) => n.id === activeSection), [activeSection])

  return (
    <div className="w-full min-h-screen bg-background text-foreground flex flex-col relative" style={{ boxShadow: CARD_SHADOW }}>
      {/* Atmospheric mesh gradient background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] rounded-full opacity-[0.03] blur-[120px] animate-float" style={{ background: C.teal }} />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] rounded-full opacity-[0.02] blur-[100px] animate-float" style={{ background: C.azure, animationDelay: "3s" }} />
      </div>

      {/* Header */}
      <header className="border-b border-border/60 bg-card/60 backdrop-blur-xl sticky top-0 z-30 relative">
        <div className="w-full px-5 lg:px-10 xl:px-14">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2.5">
                <div className="size-8 rounded-xl bg-primary/12 flex items-center justify-center glow-teal-sm">
                  <ShieldAlert className="size-4 text-primary" />
                </div>
              </div>
              {analysis && (
                <div className="hidden md:flex items-center gap-1 ml-3 text-xs text-muted-foreground font-sans">
                  <span>AML Insights</span>
                  <ChevronRight className="size-3 text-muted-foreground/50" />
                  <span className="text-foreground font-semibold">{activeNav?.label}</span>
                </div>
              )}
            </div>
            {analysis && (
              <div className="flex items-center gap-3">
                {fileName && <span className="hidden sm:inline text-[11px] text-muted-foreground font-mono truncate max-w-48">{fileName}</span>}
                <button
                  onClick={() => reuploadInputRef.current?.click()}
                  disabled={isLoading}
                  className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors px-3 py-2 rounded-xl surface-card hover:bg-accent/50 font-sans disabled:opacity-50"
                >
                  {isLoading ? <Spinner className="size-3.5" /> : <Upload className="size-3.5" />}
                  Upload new file
                </button>
                <input
                  ref={reuploadInputRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleFileUpload(file)
                    e.target.value = ""
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Navigation */}
      {analysis && (
        <nav className="border-b border-border/40 bg-card/40 backdrop-blur-xl sticky top-16 z-20 relative">
          <div className="w-full px-5 lg:px-10 xl:px-14">
            <div className="flex items-center gap-0.5 overflow-x-auto py-1.5 -mb-px scrollbar-none">
              {NAV_ITEMS.map((item) => {
                const isActive = item.id === activeSection
                const Icon = item.icon
                return (
                  <button key={item.id} onClick={() => handleNavigation(item.id)}
                    className={`relative flex items-center gap-2 px-4 py-2.5 text-[13px] font-semibold rounded-xl transition-all duration-250 whitespace-nowrap shrink-0 font-sans ${
                      isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground hover:bg-accent/30"
                    }`}
                    aria-current={isActive ? "page" : undefined}
                  >
                    <Icon className="size-4" />
                    <span>{item.label}</span>
                    {isActive && (
                      <motion.div layoutId="nav-indicator" className="absolute bottom-0 left-4 right-4 h-0.5 rounded-full bg-primary" style={{ boxShadow: `0 0 8px 2px oklch(0.78 0.16 182 / 0.3)` }} transition={SPRING} />
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        </nav>
      )}

      {/* Content */}
      <main className="w-full px-5 lg:px-10 xl:px-14 py-6 lg:py-8 flex-1 relative z-10">
        {!analysis ? (
          <UploadPrompt isLoading={isLoading} progress={progress} error={error} onFileSelect={handleFileUpload} />
        ) : (
          <AnimatePresence mode="wait">
            <motion.div key={activeSection}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: isTransitioning ? 0.3 : 1, y: isTransitioning ? 6 : 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.35, ease: EASE_OUT }}
            >
              {error && (
                <div className="mb-5">
                  <Alert variant="destructive">
                    <AlertTriangle />
                    <AlertTitle>Could not analyze new file</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                </div>
              )}
              {renderSection(activeSection, analysis, riskThreshold, setRiskThreshold)}
            </motion.div>
          </AnimatePresence>
        )}
      </main>
    </div>
  )
}
