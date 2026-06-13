// Bundles lib/analyze.worker.ts (TS + papaparse + analyze lib) into a single
// self-contained JS file in public/, so the browser can load it as a plain
// Worker script (new Worker("/analyze-worker.js")) without relying on
// Next.js/Turbopack's `new Worker(new URL(...))` bundling support, which
// emits raw .ts files in production builds.
import esbuild from "esbuild"
import path from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, "..")

await esbuild.build({
  entryPoints: [path.join(root, "lib/analyze.worker.ts")],
  outfile: path.join(root, "public/analyze-worker.js"),
  bundle: true,
  format: "iife",
  platform: "browser",
  target: "es2020",
  minify: true,
  alias: {
    "@": root,
  },
  logLevel: "info",
})
