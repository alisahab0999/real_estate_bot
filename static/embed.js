// static/embed.js
// WHAT: Client-facing loader script. Reads optional config the client sets
// BEFORE including this script, and passes it to the widget via URL params.
// WHY: Lets each client customize brand color/store name/api key WITHOUT us
// maintaining a separate widget.html per client — one file serves everyone.

(function () {
  // WHY: This MUST be the full, absolute Railway URL — unlike widget.html
  // (which could use a relative "" path since it calls /chat on its OWN
  // origin), embed.js runs on the CLIENT'S website, a completely different
  // domain. It has no way to know your server's address unless we spell
  // it out explicitly here.
  const WIDGET_BASE_URL = "https://realestatebot-production-1ca9.up.railway.app";

  if (window.__storeChatWidgetLoaded) return;
  window.__storeChatWidgetLoaded = true;

  const config = window.StoreChatConfig || {};
  const color = config.color || "#2563eb";
  const storeName = config.storeName || "Store Assistant";
  const apiKey = config.apiKey || "";

  const params = new URLSearchParams({
    color: color,
    storeName: storeName,
    apiKey: apiKey,
  });

  const button = document.createElement("div");
  button.innerHTML = "💬";
  Object.assign(button.style, {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    width: "56px",
    height: "56px",
    borderRadius: "50%",
    background: color,
    color: "white",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "24px",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
    zIndex: "999999",
  });

  const iframe = document.createElement("iframe");
  iframe.src = WIDGET_BASE_URL + "/static/widget.html?" + params.toString();
  Object.assign(iframe.style, {
    position: "fixed",
    bottom: "90px",
    right: "20px",
    width: "360px",
    height: "500px",
    border: "none",
    borderRadius: "12px",
    boxShadow: "0 8px 30px rgba(0,0,0,0.2)",
    zIndex: "999999",
    display: "none",
  });

  let isOpen = false;
  button.addEventListener("click", () => {
    isOpen = !isOpen;
    iframe.style.display = isOpen ? "block" : "none";
    button.innerHTML = isOpen ? "✕" : "💬";
  });

  document.body.appendChild(iframe);
  document.body.appendChild(button);
})();