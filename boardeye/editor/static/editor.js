"use strict";

// Solid glyphs for both colours, tinted by CSS — matching how real pieces look
// (solid objects in a light or dark finish, not hollow outlines).
const GLYPH = { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" };
const NAME = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
const FILES = "abcdefgh";

let state = null;
let selected = { rank: 0, file: 0 };
let queueIndex = -1;
let picking = false;
let picked = [];

const $ = (id) => document.getElementById(id);
const squareName = (r, f) => FILES[f] + (8 - r);

async function api(path, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Request failed", true);
    return null;
  }
  return data;
}

function setStatus(message, isError = false) {
  const el = $("commit-status");
  el.textContent = message || "";
  el.style.color = isError ? "var(--danger)" : "var(--accent)";
}

/* ------------------------------------------------------------------ render */

function squareAt(rank, file) {
  return state.squares.find((s) => s.rank === rank && s.file === file);
}

function renderBoard() {
  const board = $("board");
  board.innerHTML = "";
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const sq = squareAt(r, f);
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "square " + ((r + f) % 2 === 1 ? "dark" : "light");
      if (sq.flagged) cell.classList.add("flagged");
      if (selected.rank === r && selected.file === f) cell.classList.add("selected");
      cell.setAttribute("aria-label", squareName(r, f) + (sq.piece ? " " + NAME[sq.piece.toLowerCase()] : " empty"));

      if (sq.piece) {
        const glyph = document.createElement("span");
        glyph.className = "glyph " + (sq.is_black ? "black" : "white");
        glyph.textContent = GLYPH[sq.piece.toLowerCase()];
        cell.appendChild(glyph);
      }
      const coord = document.createElement("span");
      coord.className = "coord";
      coord.textContent = squareName(r, f);
      cell.appendChild(coord);

      cell.addEventListener("click", () => selectSquare(r, f));
      board.appendChild(cell);
    }
  }
}

function renderReview() {
  const sq = squareAt(selected.rank, selected.file);
  $("crop").src = `/image/crop/${selected.rank}/${selected.file}.png?t=${Date.now()}`;
  $("crop-square").textContent = squareName(selected.rank, selected.file);

  if (!sq.occupied) {
    $("crop-guess").textContent = "read as empty";
  } else {
    const colour = sq.is_black ? "black" : "white";
    const pct = Math.round(sq.piece_confidence * 100);
    // Name whichever of the three readings is the shaky one, rather than
    // showing a single blended number that matches none of them.
    const weakest = Math.min(sq.piece_confidence, sq.colour_confidence, sq.occupancy_confidence);
    let caveat = "";
    if (weakest === sq.occupancy_confidence && weakest < 0.6) caveat = " — unsure anything is here";
    else if (weakest === sq.colour_confidence && weakest < 0.6) caveat = " — unsure of the colour";
    $("crop-guess").textContent =
      `read as ${colour} ${NAME[sq.piece.toLowerCase()]} — ${pct}% confident${caveat}`;
  }

  const list = $("alternatives");
  list.innerHTML = "";
  Object.entries(sq.alternatives || {}).forEach(([type, p]) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${NAME[type]}</span><span>${Math.round(p * 100)}%</span>`;
    list.appendChild(li);
  });

  // Deliberately not phrased as "these are the errors". The queue ranks by
  // confidence, and a model can be confidently wrong, so the board on the left
  // is still the real check.
  $("queue-position").textContent = state.flagged.length
    ? `${state.flagged.length} least-certain square(s). Space steps through them — ` +
      `then compare the board against yours.`
    : "Nothing looked uncertain. Still worth a glance at the board before exporting.";

  $("corrections-count").textContent = state.corrections
    ? `${state.corrections} correction(s) ready to learn from.`
    : "";
}

function renderMeta() {
  $("fen").value = state.fen;
  $("lichess").href = state.lichess;
  $("chesscom").href = state.chesscom;
  $("side-to-move").value = state.side_to_move;
  $("castling").value = state.castling;
  $("en-passant").value = state.en_passant;
  $("learn").checked = state.learn;

  const warnings = $("warnings");
  if (state.warnings.length) {
    warnings.hidden = false;
    warnings.innerHTML =
      "<strong>Check this position:</strong><ul>" +
      state.warnings.map((w) => `<li>${w}</li>`).join("") +
      "</ul>";
  } else {
    warnings.hidden = true;
  }

  const d = state.detection;
  $("detection-info").textContent = `Board located by: ${d.method}.`;
  drawCorners(d.corners);

  if (!state.trained) {
    setStatus(
      "No trained model yet — every piece was guessed as a pawn. Run 'boardeye calibrate' on a photo of the starting position.",
      true
    );
  }
}

function drawCorners(corners) {
  const svg = $("corner-overlay");
  const img = $("photo");
  svg.innerHTML = "";
  if (!corners || !img.naturalWidth) return;
  const sx = img.clientWidth / img.naturalWidth;
  const sy = img.clientHeight / img.naturalHeight;
  const points = corners.map(([x, y]) => `${x * sx},${y * sy}`).join(" ");
  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  poly.setAttribute("points", points);
  poly.setAttribute("fill", "rgba(110,168,254,.12)");
  poly.setAttribute("stroke", "#6ea8fe");
  poly.setAttribute("stroke-width", "2");
  svg.appendChild(poly);
}

function render(newState) {
  if (newState) state = newState;
  renderBoard();
  renderReview();
  renderMeta();
}

/* ------------------------------------------------------------- interaction */

function selectSquare(rank, file) {
  selected = { rank, file };
  queueIndex = state.review_order.findIndex(([r, f]) => r === rank && f === file);
  render();
}

function stepQueue(direction) {
  // Walk the flagged squares, which the server already ordered least-certain
  // first and padded so a confidently-wrong square still appears.
  const flagged = state.flagged;
  if (!flagged.length) {
    setStatus("No flagged squares left.");
    return;
  }
  let index = flagged.findIndex(([r, f]) => r === selected.rank && f === selected.file);
  index = index === -1 ? 0 : (index + direction + flagged.length) % flagged.length;
  const [r, f] = flagged[index];
  selectSquare(r, f);
}

function moveSelection(dr, df) {
  const rank = Math.min(7, Math.max(0, selected.rank + dr));
  const file = Math.min(7, Math.max(0, selected.file + df));
  selectSquare(rank, file);
}

async function setPiece(type) {
  const sq = squareAt(selected.rank, selected.file);
  const data = await api("/api/square", {
    rank: selected.rank,
    file: selected.file,
    piece_type: type,
    // Keep the detected colour when the square already had one; default to
    // white for a square being filled from empty.
    is_black: sq.occupied ? sq.is_black : false,
  });
  if (data) render(data);
}

async function emptySquare() {
  const data = await api("/api/square", {
    rank: selected.rank,
    file: selected.file,
    piece_type: null,
  });
  if (data) render(data);
}

async function flipColour() {
  const data = await api("/api/recolour", { rank: selected.rank, file: selected.file });
  if (data) render(data);
}

async function commitIfNeeded() {
  if (!state.learn || !state.corrections) return;
  const data = await api("/api/commit", {});
  if (data) {
    const message = data.message;
    render(data);
    if (message) setStatus(message);
  }
}

/* ---------------------------------------------------------- corner picking */

function beginPicking() {
  picking = true;
  picked = [];
  $("photo").parentElement.classList.add("picking");
  $("corner-hint").textContent =
    "Click the four corners of the board: top-left, top-right, bottom-right, bottom-left.";
}

function endPicking() {
  picking = false;
  picked = [];
  $("photo").parentElement.classList.remove("picking");
  $("corner-hint").textContent = "";
}

async function onPhotoClick(event) {
  if (!picking) return;
  const img = $("photo");
  const rect = img.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * img.naturalWidth;
  const y = ((event.clientY - rect.top) / rect.height) * img.naturalHeight;
  picked.push([x, y]);
  $("corner-hint").textContent = `${picked.length} of 4 corners picked.`;

  if (picked.length === 4) {
    const data = await api("/api/corners", { corners: picked });
    endPicking();
    if (data) {
      render(data);
      setStatus("Board re-read from the corners you picked.");
    }
  }
}

/* -------------------------------------------------------------------- wire */

function onKey(event) {
  if (event.target.matches("input, textarea, select")) return;
  const key = event.key.toLowerCase();

  if (key in GLYPH) { event.preventDefault(); setPiece(key); return; }
  if (key === "x" || key === "delete" || key === "backspace") { event.preventDefault(); emptySquare(); return; }
  if (key === "c") { event.preventDefault(); flipColour(); return; }
  if (key === " ") { event.preventDefault(); stepQueue(event.shiftKey ? -1 : 1); return; }

  const moves = { arrowup: [-1, 0], arrowdown: [1, 0], arrowleft: [0, -1], arrowright: [0, 1] };
  if (key in moves) { event.preventDefault(); moveSelection(...moves[key]); }
}

function wire() {
  document.addEventListener("keydown", onKey);

  $("piece-buttons").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.dataset.type) setPiece(button.dataset.type);
    else if (button.dataset.action === "empty") emptySquare();
    else if (button.dataset.action === "colour") flipColour();
  });

  $("rotate").addEventListener("click", async () => {
    const data = await api("/api/rotate", { turns: 1 });
    if (data) { render(data); setStatus("Rotated. Castling rights were re-inferred."); }
  });

  const pushSetting = async (payload) => {
    const data = await api("/api/settings", payload);
    if (data) render(data);
  };
  $("side-to-move").addEventListener("change", (e) => pushSetting({ side_to_move: e.target.value }));
  $("castling").addEventListener("change", (e) => pushSetting({ castling: e.target.value }));
  $("en-passant").addEventListener("change", (e) => pushSetting({ en_passant: e.target.value }));
  $("learn").addEventListener("change", (e) => pushSetting({ learn: e.target.checked }));

  $("copy-fen").addEventListener("click", async () => {
    await navigator.clipboard.writeText(state.fen);
    setStatus("FEN copied.");
    commitIfNeeded();
  });
  $("lichess").addEventListener("click", commitIfNeeded);
  $("chesscom").addEventListener("click", commitIfNeeded);

  $("pick-corners").addEventListener("click", () => (picking ? endPicking() : beginPicking()));
  $("photo").addEventListener("click", onPhotoClick);
  $("photo").addEventListener("load", () => state && drawCorners(state.detection.corners));
  window.addEventListener("resize", () => state && drawCorners(state.detection.corners));
}

(async function start() {
  const data = await api("/api/state");
  if (!data) return;
  state = data;
  // Open on the least-confident square, since that is where attention belongs.
  const first = state.flagged[0];
  if (first) selected = { rank: first[0], file: first[1] };
  wire();
  render();
})();
