/**
 * Aurora+ update-banner template.
 * Returns a full self-contained HTML string for one banner.
 * Shared by the CLI renderer (generate-banner.js).
 */

// ---- Bilingual motivational copy presets (random pick each run) ----
const PRESETS = [
  {
    name: "Har doim oldinda",
    uz: { t: "Aurora+ har doim oldinda", s: "Yangi versiya tayyor — yanada tez va qulay. Biz to'xtamaymiz." },
    ru: { t: "Aurora+ всегда впереди", s: "Новая версия готова — быстрее и удобнее. Мы не стоим на месте." }
  },
  {
    name: "Yangi imkoniyatlar",
    uz: { t: "Yangi imkoniyatlar tayyor", s: "Aurora+ ni yangilang va eng so'nggi yaxshilanishlardan bahramand bo'ling." },
    ru: { t: "Новые возможности уже здесь", s: "Обновите Aurora+ и откройте для себя последние улучшения." }
  },
  {
    name: "Birga o'samiz",
    uz: { t: "Biz yaxshilanishda davom etamiz", s: "Har bir yangilanish — sizning qulayligingiz uchun. Aurora+ bilan oldinga." },
    ru: { t: "Мы продолжаем расти", s: "Каждое обновление — ради вашего удобства. Вперёд вместе с Aurora+." }
  },
  {
    name: "Ko'proq ulguring",
    uz: { t: "Vaqtni tejang, ko'proq ulguring", s: "Yangi Aurora+ bilan ishlaringiz yanada silliq va tez kechadi." },
    ru: { t: "Экономьте время, успевайте больше", s: "С новым Aurora+ работа идёт ещё легче и быстрее." }
  },
  {
    name: "Kelajak bugun",
    uz: { t: "Kelajak — bugun shu yerda", s: "Aurora+ ning yangi avlodi sizni kutmoqda. Birinchilardan bo'ling." },
    ru: { t: "Будущее — уже сегодня", s: "Новое поколение Aurora+ ждёт вас. Будьте среди первых." }
  }
];

const RELABEL = {
  uz:   "Yangi reliz",
  ru:   "Новый релиз",
  both: "Yangi reliz · Новый релиз"
};

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Build banner HTML.
 * @param {object} o
 * @param {string} o.oldVersion  e.g. "v2.4" ("" → single release tag, no arrow)
 * @param {string} o.newVersion  e.g. "v2.5" (required)
 * @param {string} o.lang        "ru" | "uz" | "both"
 * @param {object} o.preset      one PRESETS entry
 * @param {string} o.layout      "side" | "stacked" (only affects "both")
 */
function buildHTML({ oldVersion = "", newVersion, lang = "ru", preset, layout = "side" }) {
  const both = lang === "both";
  const single = !oldVersion; // no old version → release-tag mode (no arrow)
  const primaryLang = both ? "uz" : lang;

  const cls = ["aura"];
  if (both) cls.push("both");
  if (single) cls.push("single");
  if (layout === "stacked") cls.push("stacked");

  const arrow = `<svg class="varrow" width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h13M13 6l6 6-6 6"/></svg>`;

  const vrow = `
    <div class="vrow">
      <span class="vpill old">${esc(oldVersion || "")}</span>
      ${arrow}
      <span class="vpill new">${esc(newVersion)}</span>
    </div>`;

  const col = (langKey, chipText, tag) => `
    <div class="col ${langKey === "ru" ? "ru" : "uz"}">
      <span class="chip ${langKey === "ru" ? "ru" : ""}">${chipText}</span>
      <${tag}>${esc(preset[langKey].t)}</${tag}>
      <p>${esc(preset[langKey].s)}</p>
    </div>`;

  const cols = both
    ? `
    <div class="cols">
      ${col(primaryLang, "O'ZBEK", "h2")}
      <div class="vline"></div>
      ${col("ru", "РУССКИЙ", "h3")}
    </div>`
    : `
    <div class="cols">
      ${col(primaryLang, primaryLang === "ru" ? "РУССКИЙ" : "O'ZBEK", "h2")}
    </div>`;

  return `<!doctype html>
<html lang="${lang === "ru" ? "ru" : "uz"}">
<head>
<meta charset="utf-8" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --brand:#1e49c7;--brand-2:#1b3ea8;--brand-ink:#142f84;--brand-soft:#eaf0ff;
    --green:#10b981;--amber:#f59e0b;--ink:#0f172a;--muted:#64748b;--line:#e5e7eb;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:#fff;font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;color:var(--ink)}
  #banner{position:relative;width:1456px;height:816px;background:#fff;overflow:hidden}
  .aura::before{content:"";position:absolute;inset:0;
    background:radial-gradient(46% 70% at 16% 64%, rgba(16,185,129,.30), transparent 60%),
               radial-gradient(54% 90% at 88% 50%, rgba(30,73,199,.34), transparent 62%);}
  #banner > *{position:relative;z-index:1}
  .mark{position:absolute;top:60px;left:68px;display:flex;align-items:flex-start;gap:2px;z-index:3}
  .mark .spark{width:22px;height:22px;margin-right:5px;margin-top:7px;color:var(--brand)}
  .mark .word{font-weight:800;font-size:38px;letter-spacing:-.03em;color:var(--brand);line-height:1}
  .mark .plus{font-weight:700;font-size:18px;color:var(--brand);margin-left:1px;margin-top:2px}
  .sparkle{position:absolute}
  #banner > .center{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1}
  .winCard{width:980px;background:#fff;border-radius:26px;
    box-shadow:0 2px 4px rgba(15,23,42,.04), 0 44px 90px -30px rgba(20,47,132,.40);
    overflow:hidden;border:1px solid rgba(255,255,255,.7)}
  .winCard .bar{height:52px;background:#f1f4f9;display:flex;align-items:center;gap:10px;padding:0 20px;border-bottom:1px solid #e7ebf3}
  .winCard .bar i{width:13px;height:13px;border-radius:50%;display:inline-block}
  .winCard .bar .r{background:#ec5b56}.winCard .bar .y{background:#f5bf4f}.winCard .bar .g{background:#5bc466}
  .winCard .body{padding:50px 60px 54px;text-align:center}
  .vhead{margin-bottom:34px}
  .relabel{display:inline-block;font-size:14px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--brand);margin-bottom:18px}
  .vrow{display:flex;align-items:center;justify-content:center;gap:24px}
  .vpill{font-family:'JetBrains Mono',monospace;font-weight:500;font-size:33px;padding:12px 26px;border-radius:14px;line-height:1}
  .vpill.old{background:#eef2f7;color:#94a3b8;text-decoration:line-through;text-decoration-thickness:2px}
  .vpill.new{background:var(--brand);color:#fff;box-shadow:0 14px 30px -10px rgba(30,73,199,.7)}
  .varrow{color:var(--brand);flex:none}
  #banner.single .vpill.old, #banner.single .varrow{display:none}
  .cols{display:flex;align-items:flex-start;justify-content:center;padding-top:30px;border-top:1px solid var(--line)}
  .col{flex:1;padding:0 34px;display:flex;flex-direction:column;align-items:center;text-align:center}
  .vline{width:1px;background:var(--line);flex:none;align-self:stretch;margin:4px 0}
  .chip{display:none;font-size:12px;font-weight:700;letter-spacing:.16em;color:var(--brand);background:var(--brand-soft);border:1px solid #d6e0ff;border-radius:999px;padding:5px 14px;margin-bottom:18px}
  #banner.both .chip{display:inline-block}
  .chip.ru{color:var(--brand-ink)}
  .col h2,.col h3{margin:0 0 11px;font-size:28px;letter-spacing:-.02em;line-height:1.16;font-weight:700;color:var(--ink);text-wrap:balance}
  .col p{margin:0;font-size:17px;color:var(--muted);line-height:1.5;text-wrap:pretty}
  #banner:not(.both) .vline{display:none}
  #banner:not(.both) .col{max-width:620px;padding:0}
  #banner:not(.both) .col h2{font-size:37px}
  #banner:not(.both) .col p{font-size:20px}
  #banner.stacked.both .cols{flex-direction:column;align-items:center;gap:26px}
  #banner.stacked.both .vline{width:60%;height:1px;align-self:center;margin:0}
  #banner.stacked.both .col{max-width:600px;padding:0}
</style>
</head>
<body>
  <div id="banner" class="${cls.join(" ")}">
    <div class="mark">
      <svg class="spark" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0l2.4 7.2L22 9.6l-7.2 2.4L12 19.2 9.6 12 2.4 9.6 9.6 7.2z"/></svg>
      <span class="word">Aurora</span><span class="plus">+</span>
    </div>
    <svg class="sparkle" style="top:130px;right:170px;color:var(--amber);opacity:.9" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0l2.2 7.2L21.6 9.6l-7.4 2.2L12 19.2 9.8 11.8 2.4 9.6 9.8 7.2z"/></svg>
    <svg class="sparkle" style="bottom:140px;left:170px;color:var(--green);opacity:.85" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0l2.2 7.2L21.6 9.6l-7.4 2.2L12 19.2 9.8 11.8 2.4 9.6 9.8 7.2z"/></svg>
    <div class="center">
      <div class="winCard">
        <div class="bar"><i class="r"></i><i class="y"></i><i class="g"></i></div>
        <div class="body">
          <div class="vhead">
            <span class="relabel">${RELABEL[lang] || RELABEL.ru}</span>
            ${vrow}
          </div>
          ${cols}
        </div>
      </div>
    </div>
  </div>
</body>
</html>`;
}

module.exports = { PRESETS, RELABEL, buildHTML };
