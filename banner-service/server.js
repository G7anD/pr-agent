#!/usr/bin/env node
/**
 * Aurora+ banner HTTP service.
 *
 * Endpoints:
 *   GET /health                          -> 200 {"status":"ok"}
 *   GET /banner?new=<v>&old=<v>&lang=ru  -> 200 image/png (banner bytes)
 *
 * Reuses banner-template.buildHTML + PRESETS. Renders to a PNG buffer via
 * headless Chromium and streams it back — never writes to disk.
 */

const http = require("http");
const { URL } = require("url");
const puppeteer = require("puppeteer");
const { PRESETS, buildHTML } = require("./banner-template");

const PORT = Number(process.env.PORT) || 41928;
const HOST = process.env.HOST || "0.0.0.0";

function pickIdx(variant) {
  if (variant !== null && variant !== undefined && variant !== "") {
    const n = Number(variant);
    if (!Number.isNaN(n)) return ((n % PRESETS.length) + PRESETS.length) % PRESETS.length;
  }
  return Math.floor(Math.random() * PRESETS.length);
}

async function renderBanner({ oldVersion, newVersion, lang, variant }) {
  const preset = PRESETS[pickIdx(variant)];
  const html = buildHTML({
    oldVersion: oldVersion || "",
    newVersion,
    lang: lang || "ru",
    preset,
    layout: "side",
  });
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--force-color-profile=srgb"],
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1456, height: 816, deviceScaleFactor: 2 });
    await page.setContent(html, { waitUntil: "networkidle0", timeout: 60000 });
    await page.evaluate(async () => {
      if (document.fonts && document.fonts.ready) await document.fonts.ready;
    });
    const el = await page.$("#banner");
    return await el.screenshot({ type: "png" });
  } finally {
    await browser.close();
  }
}

function sendJson(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(body);
}

const server = http.createServer(async (req, res) => {
  let u;
  try {
    u = new URL(req.url, `http://localhost:${PORT}`);
  } catch {
    return sendJson(res, 400, { error: "bad url" });
  }

  if (u.pathname === "/health") {
    return sendJson(res, 200, { status: "ok" });
  }

  if (u.pathname === "/banner") {
    const newVersion = u.searchParams.get("new");
    if (!newVersion) return sendJson(res, 400, { error: "query param 'new' is required" });
    try {
      const png = await renderBanner({
        oldVersion: u.searchParams.get("old"),
        newVersion,
        lang: u.searchParams.get("lang") || "ru",
        variant: u.searchParams.get("variant"),
      });
      res.writeHead(200, { "Content-Type": "image/png", "Content-Length": png.length });
      res.end(png);
    } catch (e) {
      console.error("render error:", e && e.message ? e.message : e);
      return sendJson(res, 500, { error: String((e && e.message) || e) });
    }
    return;
  }

  return sendJson(res, 404, { error: "not found" });
});

server.listen(PORT, HOST, () => {
  console.log(`aurora banner-service listening on ${HOST}:${PORT}`);
});
