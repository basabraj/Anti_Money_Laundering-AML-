"use client"

import { useEffect, useRef, useState } from "react"
import ForceGraph2D from "react-force-graph-2d"
import type { AnalyzeResponse } from "@/app/api/analyze/lib"

const C = {
  rose: "oklch(0.62 0.22 18)",
  azure: "oklch(0.68 0.14 245)",
}

interface GraphNode {
  id: string
  val: number
  color: string
  riskScore: number
}

interface GraphLink {
  source: string
  target: string
  value: number
}

interface TransactionGraphProps {
  nodes: AnalyzeResponse["graph"]["nodes"]
  links: AnalyzeResponse["graph"]["links"]
  riskThreshold: number
}

export default function TransactionGraph({ nodes, links, riskThreshold }: TransactionGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width)
    })
    observer.observe(el)
    setWidth(el.clientWidth)
    return () => observer.disconnect()
  }, [])

  const maxAmount = links.reduce((max, l) => Math.max(max, l.amount), 0) || 1

  const volumes = nodes.map((n) => n.totalSent + n.totalReceived)
  const minVolume = Math.min(...volumes, 0)
  const maxVolume = Math.max(...volumes, 1)
  const volumeRange = maxVolume - minVolume || 1
  // Map raw transaction volume to a small range so node radius (sqrt(val) * nodeRelSize)
  // stays a few px to ~15px instead of blanketing the canvas for high-volume accounts.
  const normalizeVal = (v: number) => 1 + ((v - minVolume) / volumeRange) * 12

  const graphData = {
    nodes: nodes.map(
      (n): GraphNode => ({
        id: n.id,
        val: normalizeVal(n.totalSent + n.totalReceived),
        color: n.riskScore > riskThreshold ? C.rose : C.azure,
        riskScore: n.riskScore,
      }),
    ),
    links: links.map(
      (l): GraphLink => ({
        source: l.source,
        target: l.target,
        value: l.amount,
      }),
    ),
  }

  return (
    <div ref={containerRef} className="h-[500px] w-full">
      {width > 0 && (
        <ForceGraph2D
          graphData={graphData}
          width={width}
          height={500}
          nodeLabel={(node) => {
            const n = node as unknown as GraphNode
            return `${n.id} (risk ${n.riskScore.toFixed(2)})`
          }}
          nodeColor="color"
          nodeVal="val"
          linkColor={() => "rgba(255,255,255,0.08)"}
          linkWidth={(link) => {
            const l = link as unknown as GraphLink
            return Math.max(0.5, (l.value / maxAmount) * 3)
          }}
          backgroundColor="transparent"
          cooldownTicks={100}
        />
      )}
    </div>
  )
}
