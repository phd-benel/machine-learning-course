(function () {
  const progressEl = document.getElementById("progressBar");
  const progressText = document.getElementById("progressText");
  const chapterIndex = Number(document.body.dataset.chapterIndex || 0);
  const chapterTotal = Number(document.body.dataset.chapterTotal || 1);

  if (progressEl && progressText) {
    const pct = Math.round((chapterIndex / chapterTotal) * 100);
    progressEl.style.width = pct + "%";
    progressText.textContent = "Progression du parcours: chapitre " + chapterIndex + " / " + chapterTotal;
  }

  const markButtons = document.querySelectorAll("[data-mark-read]");
  markButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      localStorage.setItem("immovision_last_read", String(chapterIndex));
      btn.textContent = "Chapitre marque comme lu";
      btn.disabled = true;
    });
  });

  const resumeLink = document.getElementById("resumeReading");
  if (resumeLink) {
    const last = Number(localStorage.getItem("immovision_last_read") || 0);
    const map = {
      1: "pages/01-chapitre-1-fondations.html",
      2: "pages/02-chapitre-2-boite-a-outils.html",
      3: "pages/03-chapitre-3-ingenierie.html",
      4: "pages/04-annexes-et-ressources.html"
    };
    if (last > 0 && map[last]) {
      resumeLink.href = map[last];
      resumeLink.style.display = "inline-block";
    }
  }

  function getCookie(name) {
    const m = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
    return m ? decodeURIComponent(m[2]) : null;
  }

  function setLangCookie(value) {
    const encoded = encodeURIComponent(value);
    document.cookie = "googtrans=" + encoded + "; path=/; max-age=31536000; SameSite=Lax";
  }

  function getCurrentLang() {
    const fromCookie = getCookie("googtrans");
    if (fromCookie && fromCookie.indexOf("/fr/") === 0) {
      return fromCookie.split("/")[2] || "fr";
    }
    return localStorage.getItem("immovision_lang") || "fr";
  }

  function setActiveLangButton(lang) {
    document.querySelectorAll(".lang-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.lang === lang);
    });
  }

  function waitForTranslateCombo(timeoutMs) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const combo = document.querySelector(".goog-te-combo");
        if (combo) {
          clearInterval(timer);
          resolve(combo);
          return;
        }
        if (Date.now() - start > timeoutMs) {
          clearInterval(timer);
          resolve(null);
        }
      }, 120);
    });
  }

  async function applyLanguage(lang, allowReloadFallback) {
    localStorage.setItem("immovision_lang", lang);
    setLangCookie("/fr/" + lang);
    setActiveLangButton(lang);

    const combo = await waitForTranslateCombo(3500);
    if (combo) {
      if (combo.value !== lang) {
        combo.value = lang;
        combo.dispatchEvent(new Event("change"));
      }
      return;
    }

    if (allowReloadFallback) {
      window.location.reload();
    }
  }

  function injectLanguageSwitcher() {
    const topbarInner = document.querySelector(".topbar-inner");
    if (!topbarInner) return;

    const switcher = document.createElement("div");
    switcher.className = "lang-switch";
    switcher.innerHTML =
      '<button type="button" class="lang-btn" data-lang="fr" aria-label="Basculer en français">🇫🇷 FR</button>' +
      '<button type="button" class="lang-btn" data-lang="en" aria-label="Switch to English">🇬🇧 EN</button>' +
      '<div id="google_translate_element" class="google-translate-mount" aria-hidden="true"></div>';

    topbarInner.appendChild(switcher);

    const current = getCurrentLang();
    switcher.querySelectorAll(".lang-btn").forEach((btn) => {
      if (btn.dataset.lang === current) btn.classList.add("active");
      btn.addEventListener("click", () => {
        const nextLang = btn.dataset.lang;
        if (nextLang !== getCurrentLang()) applyLanguage(nextLang, true);
      });
    });
  }

  function loadGoogleTranslate() {
    const scriptId = "google-translate-script";
    if (document.getElementById(scriptId)) return;

    window.googleTranslateElementInit = function () {
      if (!window.google || !window.google.translate) return;
      // eslint-disable-next-line no-new
      new window.google.translate.TranslateElement(
        {
          pageLanguage: "fr",
          includedLanguages: "fr,en",
          autoDisplay: false
        },
        "google_translate_element"
      );
    };

    const script = document.createElement("script");
    script.id = scriptId;
    script.src = "https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
    script.async = true;
    document.head.appendChild(script);
  }

  const preferred = getCurrentLang();
  setLangCookie("/fr/" + preferred);
  injectLanguageSwitcher();
  loadGoogleTranslate();
  // Apply persisted language once the widget is ready.
  setTimeout(() => {
    applyLanguage(preferred, false);
  }, 600);

  document.querySelectorAll("[data-audio-player]").forEach((wrap) => {
    const audio = wrap.querySelector("audio");
    const playBtn = wrap.querySelector(".audio-btn--play");
    const stopBtn = wrap.querySelector(".audio-btn--stop");
    if (!audio || !playBtn || !stopBtn) return;

    function syncUi() {
      const playing = !audio.paused;
      wrap.classList.toggle("is-playing", playing);
      playBtn.setAttribute("aria-pressed", playing ? "true" : "false");
      playBtn.setAttribute("aria-label", playing ? "Pause" : "Lecture");
    }

    playBtn.addEventListener("click", () => {
      if (audio.paused) {
        void audio.play();
      } else {
        audio.pause();
      }
    });

    stopBtn.addEventListener("click", () => {
      audio.pause();
      audio.currentTime = 0;
      syncUi();
    });

    audio.addEventListener("play", syncUi);
    audio.addEventListener("pause", syncUi);
    audio.addEventListener("ended", syncUi);
    syncUi();
  });

  /** Sommaire chapitre (TOC) : repliable en vue mobile / tablette */
  (function initChapterTocToggle() {
    const aside = document.querySelector(".chapter-toc");
    const btn = document.getElementById("chapter-toc-toggle");
    const panel = document.getElementById("chapter-toc-panel");
    if (!aside || !btn || !panel) return;

    const mq = window.matchMedia("(max-width: 1099px)");

    function syncState() {
      if (mq.matches) {
        const open = aside.classList.contains("chapter-toc--open");
        btn.setAttribute("aria-expanded", open ? "true" : "false");
      } else {
        aside.classList.add("chapter-toc--open");
        btn.setAttribute("aria-expanded", "true");
      }
    }

    btn.addEventListener("click", function () {
      if (!mq.matches) return;
      aside.classList.toggle("chapter-toc--open");
      btn.setAttribute("aria-expanded", aside.classList.contains("chapter-toc--open") ? "true" : "false");
    });

    panel.querySelectorAll('a[href^="#"]').forEach(function (a) {
      a.addEventListener("click", function () {
        if (mq.matches) {
          aside.classList.remove("chapter-toc--open");
          btn.setAttribute("aria-expanded", "false");
        }
      });
    });

    mq.addEventListener("change", function (e) {
      if (e.matches) {
        aside.classList.remove("chapter-toc--open");
        btn.setAttribute("aria-expanded", "false");
      } else {
        aside.classList.add("chapter-toc--open");
        btn.setAttribute("aria-expanded", "true");
      }
    });

    syncState();
  })();
})();
