import { chromium } from "playwright"
import path from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const datasetsDir = path.resolve(__dirname, "..", "..", "Datasets")

const file = process.argv[2] ?? "Aml_800x800.csv"
const filePath = path.isAbsolute(file) ? file : path.join(datasetsDir, file)
const shotPrefix = process.argv[3] ?? "small"
const baseUrl = process.argv[4] ?? "http://localhost:3000"

const browser = await chromium.launch()
const page = await browser.newPage()

page.on("console", (msg) => console.log(`[console:${msg.type()}]`, msg.text()))
page.on("pageerror", (err) => console.log("[pageerror]", err.message))

await page.goto(baseUrl)

const fileInput = page.locator('input[type="file"]')
await fileInput.setInputFiles(filePath)

console.log("Uploaded:", filePath)

// Capture the in-progress state (progress bar) for large files.
await page.waitForTimeout(1500)
await page.screenshot({ path: `scripts/${shotPrefix}-progress.png` })
const progressText = await page.locator("text=/%/").first().textContent().catch(() => null)
console.log("Progress at 1.5s:", progressText)

// Wait for either the analysis to complete (nav appears) or an error alert.
const navOrError = await Promise.race([
  page.locator('nav button:has-text("Overview")').waitFor({ state: "visible", timeout: 600000 }).then(() => "nav"),
  page.locator('text=Could not analyze file').waitFor({ state: "visible", timeout: 600000 }).then(() => "error"),
])

console.log("Result:", navOrError)

if (navOrError === "error") {
  const errText = await page.locator('[role="alert"], .text-destructive, text=Could not analyze file').first().textContent().catch(() => null)
  console.log("Error text:", errText)
  await page.screenshot({ path: `scripts/${shotPrefix}-error.png`, fullPage: true })
} else {
  await page.screenshot({ path: `scripts/${shotPrefix}-overview.png`, fullPage: true })

  for (const tab of ["Transaction Graph", "Fraud Accounts", "Transactions"]) {
    const btn = page.locator(`nav button:has-text("${tab}")`)
    if (await btn.count() > 0) {
      await btn.first().click()
      await page.waitForTimeout(800)
      const slug = tab.toLowerCase().replace(/\s+/g, "-")
      await page.screenshot({ path: `scripts/${shotPrefix}-${slug}.png`, fullPage: true })
      console.log("Captured tab:", tab)
    } else {
      console.log("Tab not found:", tab)
    }
  }

  // Print summary KPIs from the overview for sanity-checking.
  await page.locator('nav button:has-text("Overview")').click()
  await page.waitForTimeout(500)
  const kpis = await page.locator("main").innerText()
  console.log("--- Overview text (first 1200 chars) ---")
  console.log(kpis.slice(0, 1200))
}

await browser.close()
