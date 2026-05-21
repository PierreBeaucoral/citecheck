// citecheck mascot — reads localStorage and applies preferences to every
// `.coffee-person` element on the page.  Also exposes a global helper
// `window.citecheckMascot` used by the /customize page to save changes.
//
// Storage key: "citecheck-mascot".
// Stored shape:
//   { hair, skin, shirt, cup, hatOn, glassesOn }

(function () {
  const STORAGE_KEY = "citecheck-mascot";

  const DEFAULTS = {
    hair:    "#4b3a2e",
    skin:    "#f4c89f",
    shirt:   "#6d28d9",
    cup:     "#e11d48",
    hatOn:   false,
    glassesOn: false,
  };

  // Palettes shown in the customizer.  Keep counts small so the swatches
  // don't overwhelm the page; users who want exotic colours can override
  // by editing localStorage directly.
  const PALETTES = {
    hair:  ["#4b3a2e", "#1f1f1f", "#d4a373", "#e5a849", "#b91c1c", "#475569", "#ec4899", "#0d9488"],
    skin:  ["#fef3e0", "#f4c89f", "#deb887", "#b58463", "#8d5524", "#6b3e1f", "#3f2611"],
    shirt: ["#6d28d9", "#ec4899", "#16a34a", "#2563eb", "#f97316", "#e11d48", "#facc15", "#0d9488", "#1a1d24"],
    cup:   ["#e11d48", "#1f1f1f", "#ec4899", "#0d9488", "#6d28d9", "#facc15", "#16a34a", "#ffffff"],
  };

  function load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { ...DEFAULTS };
      return { ...DEFAULTS, ...JSON.parse(raw) };
    } catch {
      return { ...DEFAULTS };
    }
  }

  function save(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (_e) {
      // Out of quota / private mode — silently ignore; the in-memory
      // state still applies for the current page.
    }
  }

  function apply(state) {
    document.querySelectorAll(".coffee-person").forEach(el => {
      el.style.setProperty("--hair",  state.hair);
      el.style.setProperty("--skin",  state.skin);
      el.style.setProperty("--shirt", state.shirt);
      el.style.setProperty("--cup",   state.cup);
      el.style.setProperty("--hat-on",     state.hatOn     ? "inline" : "none");
      el.style.setProperty("--glasses-on", state.glassesOn ? "inline" : "none");
    });
  }

  function reset() {
    save({ ...DEFAULTS });
    apply(DEFAULTS);
    return { ...DEFAULTS };
  }

  // Apply persisted state immediately on every page.  Repeat on DOM
  // mutations so HTMX-injected partials (the job-status partial that
  // first introduces the mascot) get the right colours too.
  const initial = load();
  if (document.readyState !== "loading") {
    apply(initial);
  } else {
    document.addEventListener("DOMContentLoaded", () => apply(initial));
  }
  document.addEventListener("htmx:afterSwap", () => apply(load()));

  window.citecheckMascot = {
    load, save, apply, reset,
    PALETTES,
    DEFAULTS,
  };
})();
