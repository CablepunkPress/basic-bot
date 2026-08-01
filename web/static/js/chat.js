// ----------------------
// State
// ----------------------

/** @type {Record<string, {display_name: string, effort_levels: string[]|null, thinking_type: string}>} */
let models = {};

/** @type {string} */
let defaultModel = '';

/** @type {string} */
let selectedModel = '';

/** @type {string|null} */
let selectedEffort = null;

/** @type {boolean} */
let thinkingEnabled = false;

/** @type {number} */
let nextSeq = 1;

// ----------------------
// Marked configuration
// ----------------------

const renderer = new marked.Renderer();
const originalLink = /** @type {(args: object) => string} */ (renderer.link.bind(renderer));
/** @param {object} args */
renderer.link = function(args) {
  const html = originalLink(args);
  return html.replace('<a ', '<a target="_blank" rel="noopener" ');
};
marked.setOptions({ renderer });

// ----------------------
// Init
// ----------------------

document.addEventListener('DOMContentLoaded', async () => {
  // Load models
  try {
    const response = await fetch('/models');
    if (response.ok) {
      const data = await response.json();
      models = data.models;
      defaultModel = data.default;
      selectedModel = defaultModel;
      populateModelSelector();
    }
  } catch (error) {
    console.error('Failed to load models:', error);
  }

  // Sidebar toggle (mobile)
  const sidebarToggle = document.getElementById('sidebarToggle');
  const sidebarClose = document.getElementById('sidebarClose');
  const sidebarOverlay = document.getElementById('sidebarOverlay');
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => toggleSidebar(true));
  }
  if (sidebarClose) {
    sidebarClose.addEventListener('click', () => toggleSidebar(false));
  }
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', () => toggleSidebar(false));
  }

  // Model selector
  const modelSelect = document.getElementById('modelSelect');
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      selectedModel = /** @type {HTMLSelectElement} */ (e.target).value;
      updateModelControls();
    });
  }

  // Effort selector
  const effortSelect = document.getElementById('effortSelect');
  if (effortSelect) {
    effortSelect.addEventListener('change', (e) => {
      selectedEffort = /** @type {HTMLSelectElement} */ (e.target).value;
    });
  }

  // Thinking toggle
  const thinkingToggle = document.getElementById('thinkingToggle');
  if (thinkingToggle) {
    thinkingToggle.addEventListener('change', (e) => {
      thinkingEnabled = /** @type {HTMLInputElement} */ (e.target).checked;
    });
  }

  // Send button
  const sendBtn = document.getElementById('sendBtn');
  if (sendBtn) {
    sendBtn.addEventListener('click', sendMessage);
  }

  // Textarea
  setupTextarea();

  // Load history
  await loadHistory();

  // Focus input
  const messageInput = document.getElementById('messageInput');
  if (messageInput) {
    messageInput.focus();
  }
});

// ----------------------
// Model controls
// ----------------------

function populateModelSelector() {
  const modelSelect = /** @type {HTMLSelectElement|null} */ (
    document.getElementById('modelSelect')
  );
  if (!modelSelect) return;

  modelSelect.innerHTML = Object.entries(models)
    .map(([id, config]) =>
      `<option value="${id}"${id === selectedModel ? ' selected' : ''}>${config.display_name}</option>`
    ).join('');

  updateModelControls();
}

function updateModelControls() {
  const model = models[selectedModel];
  const effortSection = document.getElementById('effortSection');
  const effortSelect = /** @type {HTMLSelectElement|null} */ (
    document.getElementById('effortSelect')
  );

  if (!effortSection || !effortSelect) return;

  if (model && model.effort_levels) {
    effortSection.style.display = '';
    const previousEffort = selectedEffort;

    effortSelect.innerHTML = model.effort_levels
      .map(level => `<option value="${level}">${capitalize(level)}</option>`)
      .join('');

    if (previousEffort && model.effort_levels.includes(previousEffort)) {
      effortSelect.value = previousEffort;
      selectedEffort = previousEffort;
    } else {
      effortSelect.value = 'low';
      selectedEffort = 'low';
    }
  } else {
    effortSection.style.display = 'none';
    selectedEffort = null;
  }
}

// ----------------------
// History
// ----------------------

async function loadHistory() {
  try {
    const response = await fetch('/history');
    if (!response.ok) {
      console.error('Failed to load history:', response.status);
      return;
    }

    const data = await response.json();
    const messages = data.messages;

    if (!messages || messages.length === 0) return;

    const firstSeq = messages[0].seq;
    const lastSeq = messages[messages.length - 1].seq;

    for (const msg of messages) {
      if (msg.role === 'assistant') {
        const meta = msg.metadata ? {
          display_name: msg.metadata.display_name || null,
          effort: msg.metadata.effort || null,
          thinking: msg.metadata.thinking || false,
          fallback: msg.metadata.fallback || false,
        } : null;
        addMessage('agent', msg.content, meta, msg.seq);
      } else {
        addMessage('user', msg.content, null, msg.seq);
      }
    }

    const historyNote = document.getElementById('historyNote');
    if (historyNote) {
      historyNote.textContent =
        `Last ${messages.length} messages loaded: sequence ${firstSeq}–${lastSeq}`;
    }

    nextSeq = lastSeq + 1;

  } catch (error) {
    console.error('History load error:', error);
  }
}

// ----------------------
// Sidebar
// ----------------------

/** @param {string} str */
function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/** @param {boolean} open */
function toggleSidebar(open) {
  const sidebar = document.getElementById('chatSidebar');
  const overlay = document.getElementById('sidebarOverlay');
  if (sidebar) sidebar.classList.toggle('open', open);
  if (overlay) overlay.classList.toggle('open', open);
}

function setupTextarea() {
  const textarea = /** @type {HTMLTextAreaElement|null} */ (
    document.getElementById('messageInput')
  );
  if (!textarea) return;

  const maxHeight = 200;

  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.overflow = 'hidden';
    if (textarea.scrollHeight > maxHeight) {
      textarea.style.height = maxHeight + 'px';
      textarea.style.overflow = 'auto';
    } else {
      textarea.style.height = textarea.scrollHeight + 'px';
    }
  });

  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// ----------------------
// Send message
// ----------------------

async function sendMessage() {
  const textarea = /** @type {HTMLTextAreaElement|null} */ (
    document.getElementById('messageInput')
  );
  const sendBtn = /** @type {HTMLButtonElement|null} */ (
    document.getElementById('sendBtn')
  );
  if (!textarea) return;

  const message = textarea.value.trim();
  if (!message) return;

  addMessage('user', message, null, nextSeq);
  textarea.value = '';
  textarea.style.height = 'auto';

  textarea.disabled = true;
  if (sendBtn) sendBtn.disabled = true;

  const thinkingId = addThinking();

  console.log(
    `[${new Date().toISOString()}] Sending — model: ${selectedModel}, ` +
    `effort: ${selectedEffort}, thinking: ${thinkingEnabled}`
  );

  try {
    /** @type {Record<string, string|boolean>} */
    const payload = {
      message: message,
      model: selectedModel,
    };

    if (selectedEffort) {
      payload.effort = selectedEffort;
    }
    if (thinkingEnabled) {
      payload.thinking = true;
    }

    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    removeMessage(thinkingId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    const assistantSeq = data.seq + 1;
    nextSeq = data.seq + 2;

    /** @type {{display_name: string|null, effort: string|null, thinking: boolean, fallback: boolean}} */
    const meta = {
      display_name: data.display_name || null,
      effort: data.effort || null,
      thinking: data.thinking || false,
      fallback: data.fallback || false,
    };

    addMessage('agent', data.response, meta, assistantSeq);

  } catch (error) {
    removeMessage(thinkingId);
    addMessage('system', 'Something went wrong. Please try again.');
    console.error('Chat error:', error);
  }

  textarea.disabled = false;
  if (sendBtn) sendBtn.disabled = false;
  textarea.focus();
}

// ----------------------
// Message helpers
// ----------------------

/**
 * @param {string} role
 * @param {string} text
 * @param {{display_name: string|null, effort: string|null, thinking: boolean, fallback: boolean}|null} [meta]
 * @param {number|null} [seq]
 * @returns {string}
 */
function addMessage(role, text, meta = null, seq = null) {
  const messages = document.getElementById('chatMessages');
  if (!messages) return '';

  const div = document.createElement('div');
  const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  div.id = id;
  div.className = `message message-${role}`;

  if (seq !== null) {
    const seqSpan = document.createElement('span');
    seqSpan.className = 'message-seq';
    seqSpan.textContent = `#${seq}`;
    div.appendChild(seqSpan);
  }

  if (role === 'agent') {
    const cleanText = text.replace(/<!-- seq:\d+ -->/g, '').trim();
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = marked.parse(cleanText);
    div.appendChild(contentDiv);

    if (meta) {
      const metaDiv = document.createElement('div');
      metaDiv.className = 'message-meta';
      /** @type {string[]} */
      const parts = [];
      if (meta.display_name) parts.push(meta.display_name);
      if (meta.effort) parts.push(capitalize(meta.effort));
      if (meta.thinking) parts.push('Deep Reasoning');
      if (meta.fallback) parts.push('Fallback');
      if (parts.length) {
        metaDiv.textContent = parts.join(' · ');
        div.appendChild(metaDiv);
      }
    }
  } else {
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    div.appendChild(contentDiv);
  }

  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return id;
}

/** @returns {string} */
function addThinking() {
  const messages = document.getElementById('chatMessages');
  if (!messages) return '';

  const div = document.createElement('div');
  const id = `msg-thinking-${Date.now()}`;
  div.id = id;
  div.className = 'message message-thinking';
  div.innerHTML = 'Thinking<span class="thinking-dots"></span>';
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return id;
}

/** @param {string} id */
function removeMessage(id) {
  const msg = document.getElementById(id);
  if (msg) msg.remove();
}
