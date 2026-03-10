/* chatPF2E — Chat logic */

// Session UUID: fresh per page load — refresh = new conversation (true incognito)
// crypto.randomUUID() requires a secure context; fall back for plain HTTP on LAN
function generateUUID() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}
const SESSION_ID = generateUUID();

const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('user-input');
const sendBtn    = document.getElementById('btn-send');
const newChatBtn = document.getElementById('btn-new-chat');
const welcomeEl  = document.getElementById('welcome');

let isWaiting = false;

// ── Markdown ──────────────────────────────────────────

marked.use({ breaks: true, gfm: true });

function renderMarkdown(text) {
  const html = marked.parse(text);
  // Wrap tables in a scrollable div without touching the renderer API
  return html.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, '</table></div>');
}

// ── Helpers ───────────────────────────────────────────

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

function setWaiting(waiting) {
  isWaiting = waiting;
  inputEl.disabled = waiting;
  sendBtn.disabled = waiting;
  if (!waiting) inputEl.focus();
}

function hideWelcome() {
  if (welcomeEl) welcomeEl.style.display = 'none';
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function makeRow(role) {
  const isUser = role === 'user';
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = isUser ? '✦' : '⚔';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = isUser ? 'You' : 'GM Assistant';

  body.appendChild(label);
  row.appendChild(avatar);
  row.appendChild(body);
  return { row, body };
}

// ── Message rendering ─────────────────────────────────

function appendUserMessage(text) {
  hideWelcome();
  const { row, body } = makeRow('user');
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;  // plain text — user input is not markdown
  body.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function appendThinkingRow() {
  const { row, body } = makeRow('agent');
  row.id = 'thinking-row';
  const bubble = document.createElement('div');
  bubble.className = 'thinking-bubble';
  bubble.innerHTML = `
    <div class="thinking-dots"><span></span><span></span><span></span></div>
    <span class="thinking-text">Searching…</span>
  `;
  body.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function removeThinking() {
  document.getElementById('thinking-row')?.remove();
}

function appendAgentMessage(text, isError = false) {
  removeThinking();
  const { row, body } = makeRow('agent');
  if (isError) row.classList.add('error');

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.innerHTML = isError ? escapeHtml(text) : renderMarkdown(text);
  body.appendChild(bubble);

  if (!isError) {
    body.appendChild(makeFeedbackButtons(text));
  }

  messagesEl.appendChild(row);

  // Syntax highlight any code blocks
  bubble.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));

  scrollToBottom();
}

function makeFeedbackButtons(responseText) {
  const row = document.createElement('div');
  row.className = 'feedback-row';

  const upBtn   = makeThumbBtn('up',   '👍');
  const downBtn = makeThumbBtn('down', '👎');

  async function vote(rating, activeClass, btn, other) {
    row.classList.add('voted');
    btn.classList.add(activeClass);
    btn.disabled = true;
    other.disabled = true;

    try {
      await fetch('api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': SESSION_ID,
        },
        body: JSON.stringify({
          rating,
          preview: responseText.slice(0, 300),
        }),
      });
    } catch (_) { /* best-effort */ }
  }

  upBtn.addEventListener('click',   () => vote('up',   'active-up',   upBtn,   downBtn));
  downBtn.addEventListener('click', () => vote('down', 'active-down', downBtn, upBtn));

  row.appendChild(upBtn);
  row.appendChild(downBtn);
  return row;
}

function makeThumbBtn(rating, emoji) {
  const btn = document.createElement('button');
  btn.className = 'btn-feedback';
  btn.title = rating === 'up' ? 'Good response' : 'Bad response';
  btn.textContent = emoji;
  return btn;
}

// ── API ───────────────────────────────────────────────

async function sendMessage(text) {
  if (!text.trim() || isWaiting) return;

  appendUserMessage(text);
  inputEl.value = '';
  autoResize();
  setWaiting(true);
  appendThinkingRow();

  try {
    const resp = await fetch('api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': SESSION_ID,
      },
      body: JSON.stringify({ message: text }),
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      appendAgentMessage(data.error || `Server error (${resp.status})`, true);
    } else {
      appendAgentMessage(data.response);
    }
  } catch (err) {
    appendAgentMessage(`Connection error: ${err.message}`, true);
  } finally {
    setWaiting(false);
  }
}

async function resetConversation() {
  if (isWaiting) return;
  try {
    await fetch('api/reset', {
      method: 'POST',
      headers: { 'X-Session-ID': SESSION_ID },
    });
  } catch (_) { /* best-effort */ }
  window.location.reload();
}

// ── Textarea auto-resize ──────────────────────────────

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + 'px';
}

// ── Suggestion chips ──────────────────────────────────

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    inputEl.value = chip.textContent;
    autoResize();
    sendMessage(inputEl.value);
  });
});

// ── Event listeners ───────────────────────────────────

inputEl.addEventListener('input', autoResize);

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(inputEl.value);
  }
});

sendBtn.addEventListener('click', () => sendMessage(inputEl.value));
newChatBtn.addEventListener('click', resetConversation);

inputEl.focus();
