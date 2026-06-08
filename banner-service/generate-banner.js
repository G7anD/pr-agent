#!/usr/bin/env node
/**
 * Aurora+ — update banner generator (CLI / CI).
 *
 * Renders the release-notification banner to a PNG using headless Chromium.
 * Each run picks a RANDOM motivational copy variant (unless --variant is given).
 *
 * Usage:
 *   node generate-banner.js --old v2.4 --new v2.5
 *   node generate-banner.js --new 2026.06.10 --lang ru --out dist/banner.png
 *
 * Via env vars (handy in GitLab CI):
 *   OLD_VERSION   previous version  (optional — omit to show only the new release tag)
 *   NEW_VERSION   new version       (required; falls back to CI_COMMIT_TAG)
 *   BANNER_LANG   ru | uz | both    (default: ru)
 *   BANNER_OUT    output path       (default: aurora-update.png)
 *
 * CLI flags override env vars. Flags: --old --new --lang --out --variant --layout
 */

const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");
const { PRESETS, buildHTML } = require("./banner-template");

// ---------- arg parsing ----------
function flag(name, fallback) {
  const i = process.argv.indexOf("--" + name);
  return i !== -1 && process.argv[i + 1] && !process.argv[i + 1].startsWith("--")
    ? process.argv[i + 1]
    : fallback;
}

const OLD = flag("old", process.env.OLD_VERSION || "");
const NEW = flag("new", process.env.NEW_VERSION || process.env.CI_COMMIT_TAG || "");
const LANG = flag("lang", process.env.BANNER_LANG || "ru");
const OUT = flag("out", process.env.BANNER_OUT || "aurora-update.png");
const LAYOUT = flag("layout", "side");
const variantFlag = flag("variant", null);

if (!NEW) {
  console.error("✗ Yangi versiya majburiy. Bering: --new v2.5  (yoki NEW_VERSION / CI_COMMIT_TAG env).");
  process.exit(1);
}
if (!["ru", "uz", "both"].includes(LANG)) {
  console.error(`✗ Noto'g'ri til: "${LANG}". Ruxsat: ru | uz | both.`);
  process.exit(1);
}

// ---------- pick copy variant ----------
const idx =
  variantFlag !== null
    ? ((Number(variantFlag) % PRESETS.length) + PRESETS.length) % PRESETS.length
    : Math.floor(Math.random() * PRESETS.length);
const preset = PRESETS[idx];

const html = buildHTML({ oldVersion: OLD, newVersion: NEW, lang: LANG, preset, layout: LAYOUT });

// ---------- render ----------
(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--force-color-profile=srgb"],
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
  });
  try {
    const page = await browser.newPage();
    // 1456×816 @2x → 2912×1632 px PNG
    await page.setViewport({ width: 1456, height: 816, deviceScaleFactor: 2 });
    await page.setContent(html, { waitUntil: "networkidle0", timeout: 60000 });
    // make sure web fonts are loaded before snapping
    await page.evaluate(async () => {
      if (document.fonts && document.fonts.ready) await document.fonts.ready;
    });

    const el = await page.$("#banner");
    const outPath = path.resolve(OUT);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    await el.screenshot({ path: outPath, type: "png" });

    console.log(
      `✓ Banner tayyor: ${OUT}\n` +
        `  versiya:  ${OLD ? OLD + " → " : "reliz "}${NEW}\n` +
        `  til:      ${LANG}\n` +
        `  variant:  "${preset.name}" (#${idx})`
    );
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error("✗ Render xatosi:", err.message);
  process.exit(1);
});
