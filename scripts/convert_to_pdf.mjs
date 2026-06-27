import { readFileSync, writeFileSync, unlinkSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { marked } from "marked";
import puppeteer from "puppeteer";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = resolve(__dirname, "../docs");
const REPORT_PATH = resolve(DOCS_DIR, "end_to_end_report.md");
const PDF_PATH = resolve(DOCS_DIR, "end_to_end_report.pdf");
const TEMP_HTML = resolve(DOCS_DIR, "_report_temp.html");

const markdown = readFileSync(REPORT_PATH, "utf-8");

// Render markdown → HTML
const bodyHtml = await marked(markdown, { async: true });

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @page { margin: 2.5cm 2cm; }
  body { font: 12pt/1.5 'Segoe UI', 'DejaVu Sans', sans-serif; color: #222; max-width: 900px; margin: auto; padding: 1em; }
  h1 { font-size: 2em; border-bottom: 2px solid #ddd; padding-bottom: 0.3em; }
  h2 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.2em; margin-top: 1.5em; }
  h3 { font-size: 1.2em; margin-top: 1.2em; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #f0f0f0; }
  code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 10pt; }
  pre { background: #f8f8f8; padding: 1em; border: 1px solid #ddd; border-radius: 4px; overflow-x: auto; font-size: 9pt; }
  pre code { background: none; padding: 0; }
  img { max-width: 100%; height: auto; display: block; margin: 1em auto; }
  blockquote { border-left: 4px solid #ddd; margin: 1em 0; padding: 0.5em 1em; color: #555; }
  hr { border: none; border-top: 1px solid #ddd; margin: 2em 0; }
  .table-of-contents { background: #f9f9f9; padding: 1em; border: 1px solid #ddd; border-radius: 4px; }
  .table-of-contents ul { margin: 0; padding-left: 2em; }
  .table-of-contents li { margin: 0.2em 0; }
</style>
</head>
<body>
${bodyHtml}
</body>
</html>`;

// Write HTML to a temp file in docs/ so relative image paths (figures/...) resolve
writeFileSync(TEMP_HTML, html, "utf-8");

const browser = await puppeteer.launch({
  headless: true,
  executablePath: "/usr/bin/google-chrome-stable",
  args: ["--no-sandbox", "--disable-gpu"],
});

try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(TEMP_HTML).href, { waitUntil: "networkidle0" });

  await page.pdf({
    path: PDF_PATH,
    format: "A4",
    printBackground: true,
    margin: { top: "2.5cm", bottom: "2.5cm", left: "2cm", right: "2cm" },
  });
} finally {
  await browser.close();
  try { unlinkSync(TEMP_HTML); } catch {}
}

console.log(`PDF saved to ${PDF_PATH}`);

