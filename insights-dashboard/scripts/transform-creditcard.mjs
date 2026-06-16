import { createReadStream, writeFileSync, mkdirSync } from "fs"
import { createInterface } from "readline"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))
const SRC = "F:/Credit Card Fraud Detection/creditcard.csv"
const OUT_DIR = join(__dirname, "../public/demo")
mkdirSync(OUT_DIR, { recursive: true })

const BASE_DATE = new Date("2023-01-01T00:00:00Z").getTime()

function makeSender(v1, v2) {
  const bucket = ((Math.abs(parseFloat(v1) * 13 + parseFloat(v2) * 7) | 0) % 250) + 1
  return `CARD-${String(bucket).padStart(4, "0")}`
}
function makeReceiver(v3, v4, v5) {
  const bucket = ((Math.abs(parseFloat(v3) * 11 + parseFloat(v4) * 5 + parseFloat(v5) * 3) | 0) % 80) + 1
  return `MER-${String(bucket).padStart(3, "0")}`
}
function fmtDate(ms) {
  const d = new Date(ms)
  const pad = n => String(n).padStart(2, "0")
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`
}

// Read all rows, separate fraud vs normal
const fraudRows = []
const normalRows = []
let header = null

const rl = createInterface({ input: createReadStream(SRC), crlfDelay: Infinity })

rl.on("line", (line) => {
  if (!header) { header = line; return }
  // Quick Class check: last field
  const lastComma = line.lastIndexOf(",")
  const cls = line.slice(lastComma + 1).replace(/"/g, "").trim()
  if (cls === "1") fraudRows.push(line)
  else normalRows.push(line)
})

rl.on("close", () => {
  // Stratified sample: all fraud + random normal to reach ~5000 total
  const TARGET = 5000
  const normalTarget = TARGET - fraudRows.length
  // shuffle normal reservoir sample
  const sample = []
  for (let i = 0; i < normalRows.length; i++) {
    if (sample.length < normalTarget) {
      sample.push(normalRows[i])
    } else {
      const j = Math.floor(Math.random() * (i + 1))
      if (j < normalTarget) sample[j] = normalRows[i]
    }
  }

  const rows = ["sender,receiver,amount,timestamp,is_laundering"]

  for (const line of [...fraudRows, ...sample]) {
    const fields = line.split(",")
    // columns: Time,V1..V28,Amount,Class  (31 total, 0-indexed)
    const time    = parseFloat(fields[0]) * 1000  // seconds → ms
    const v1      = fields[1]
    const v2      = fields[2]
    const v3      = fields[3]
    const v4      = fields[4]
    const v5      = fields[5]
    const amount  = parseFloat(fields[29]).toFixed(2)
    const cls     = fields[30].replace(/"/g, "").trim()

    const sender   = makeSender(v1, v2)
    const receiver = makeReceiver(v3, v4, v5)
    const ts       = fmtDate(BASE_DATE + time)

    rows.push(`${sender},${receiver},${amount},${ts},${cls}`)
  }

  writeFileSync(join(OUT_DIR, "creditcard-fraud-5k.csv"), rows.join("\n"))
  console.log(`✓ creditcard-fraud-5k.csv — ${rows.length - 1} rows (${fraudRows.length} fraud + ${sample.length} normal)`)
})
