import { writeFileSync, mkdirSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))
const outDir = join(__dirname, "../public/demo")
mkdirSync(outDir, { recursive: true })

function rnd(min, max) { return Math.random() * (max - min) + min }
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)] }
function pad2(n) { return String(n).padStart(2, "0") }
function fmtDate(d) {
  return `${d.getFullYear()}-${pad2(d.getMonth()+1)}-${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
}

// ── Dataset 1: simple AML (1 000 rows, basic columns) ───────────────
{
  const accounts = Array.from({ length: 80 }, (_, i) => `ACC-${String(i+1).padStart(4,"0")}`)
  // Suspicious hubs: 5 accounts that send to many others
  const hubs = accounts.slice(0, 5)
  // Layering chains: A→B→C→D
  const chains = [accounts.slice(5,9), accounts.slice(9,13), accounts.slice(13,17)]

  const rows = ["sender,receiver,amount,timestamp"]
  const start = new Date("2023-01-01").getTime()
  const end   = new Date("2024-01-01").getTime()

  for (let i = 0; i < 1000; i++) {
    const t = new Date(rnd(start, end))
    let sender, receiver, amount

    const roll = Math.random()
    if (roll < 0.15) {
      // Hub smurfing: hub sends many small amounts to random accounts
      sender = pick(hubs)
      receiver = pick(accounts.filter(a => !hubs.includes(a)))
      amount = rnd(500, 4999).toFixed(2)
    } else if (roll < 0.25) {
      // Layering chain
      const chain = pick(chains)
      const idx = Math.floor(rnd(0, chain.length - 1))
      sender = chain[idx]; receiver = chain[idx + 1]
      amount = rnd(10000, 80000).toFixed(2)
    } else {
      // Normal transactions
      sender = pick(accounts)
      do { receiver = pick(accounts) } while (receiver === sender)
      amount = rnd(50, 15000).toFixed(2)
    }
    rows.push(`${sender},${receiver},${amount},${fmtDate(t)}`)
  }

  writeFileSync(join(outDir, "demo-simple.csv"), rows.join("\n"))
  console.log("✓ demo-simple.csv", rows.length - 1, "rows")
}

// ── Dataset 2: rich bank fraud (2 500 rows, SAML-D columns) ─────────
{
  const banks = {
    "US": ["Bank of America","Chase","Wells Fargo"],
    "GB": ["Barclays","HSBC","Lloyds"],
    "DE": ["Deutsche Bank","Commerzbank","DZ Bank"],
    "SG": ["DBS","OCBC","UOB"],
    "AE": ["Emirates NBD","ADIB","FAB"],
    "HK": ["HSBC HK","Hang Seng","Bank of East Asia"],
    "CH": ["UBS","Credit Suisse","Raiffeisen"],
  }
  const countries = Object.keys(banks)
  const currencies = ["USD","EUR","GBP","SGD","AED","HKD","CHF"]
  const payTypes = ["WIRE","ACH","SWIFT","SEPA","CRYPTO","CASH","CHEQUE"]

  const accounts = Array.from({ length: 150 }, (_, i) => `BNK-${String(i+1).padStart(5,"0")}`)
  const mules    = accounts.slice(0, 12)  // mule accounts
  const shells   = accounts.slice(12, 20) // shell companies

  const rows = [
    "sender,receiver,amount,payment_type,payment_currency,received_currency,sender_bank_location,receiver_bank_location,is_laundering"
  ]
  const start = new Date("2022-06-01").getTime()
  const end   = new Date("2024-01-01").getTime()

  for (let i = 0; i < 2500; i++) {
    const roll = Math.random()
    let sender, receiver, amount, isLaundering = 0

    if (roll < 0.08) {
      // Structuring: mule receives many sub-10k amounts
      sender = pick(accounts.filter(a => !mules.includes(a)))
      receiver = pick(mules)
      amount = rnd(3000, 9800).toFixed(2)
      isLaundering = 1
    } else if (roll < 0.15) {
      // Shell layering: shell→shell cross-border
      sender = pick(shells); receiver = pick(shells.filter(a => a !== sender))
      amount = rnd(50000, 500000).toFixed(2)
      isLaundering = 1
    } else if (roll < 0.20) {
      // Mule integration: mule sends large amount out
      sender = pick(mules)
      receiver = pick(accounts.filter(a => !mules.includes(a) && !shells.includes(a)))
      amount = rnd(20000, 120000).toFixed(2)
      isLaundering = 1
    } else {
      // Legitimate traffic
      sender = pick(accounts)
      do { receiver = pick(accounts) } while (receiver === sender)
      amount = rnd(100, 25000).toFixed(2)
    }

    const senderCountry   = pick(countries)
    const receiverCountry = pick(countries)
    const payType         = pick(payTypes)
    const currency        = pick(currencies)
    const receivedCcy     = isLaundering && Math.random() < 0.5 ? pick(currencies.filter(c => c !== currency)) : currency

    rows.push(
      `${sender},${receiver},${amount},${payType},${currency},${receivedCcy},${senderCountry},${receiverCountry},${isLaundering}`
    )
  }

  writeFileSync(join(outDir, "demo-rich.csv"), rows.join("\n"))
  console.log("✓ demo-rich.csv", rows.length - 1, "rows")
}
