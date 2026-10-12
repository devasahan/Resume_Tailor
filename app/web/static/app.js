// Shared helpers. All dynamic text is inserted with textContent, never innerHTML, so model output cannot inject markup.
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value === null || value === undefined) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

// replaceChildren() turns null/false into the text "null"/"false". Drop them, and allow arrays, so conditional children are safe.
const _replaceChildren = Element.prototype.replaceChildren;
Element.prototype.replaceChildren = function (...nodes) {
  return _replaceChildren.apply(this, nodes.flat().filter(n => n !== null && n !== undefined && n !== false));
};

// Who is logged in (set by the page): per-account browser storage, and the admin-only parts of a page.
const ME = { id: document.body.dataset.userId || '', admin: document.body.dataset.role === 'admin' };
const STORE_PREFIX = 'resumeTailor.';
function storeKey(name) { return `${STORE_PREFIX}${name}.${ME.id}`; }
function recallValue(name) { try { return localStorage.getItem(storeKey(name)); } catch (e) { return null; } }
function rememberValue(name, value) { try { localStorage.setItem(storeKey(name), String(value)); } catch (e) { /* storage blocked */ } }
function forgetAll() {
  try { Object.keys(localStorage).filter(k => k.startsWith(STORE_PREFIX)).forEach(k => localStorage.removeItem(k)); }
  catch (e) { /* storage blocked */ }
}

async function api(path, { method = 'GET', body = null, form = null } = {}) {
  const headers = {};
  const init = { method, headers };
  if (form) init.body = form;
  else if (body !== null) { headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
  const response = await fetch(path, init);
  let data = null;
  try { data = await response.json(); } catch (e) { /* not JSON */ }
  if (response.status === 401 && ME.id) {  // the session ended (logged out elsewhere, disabled, expired)
    location.href = '/login?next=' + encodeURIComponent(location.pathname + location.search);
    throw new Error('Please log in again.');
  }
  if (!response.ok) {
    let message = (data && data.detail) || response.statusText || 'Request failed';
    if (Array.isArray(message)) message = message.map(d => (d.loc ? d.loc.slice(1).join('.') + ': ' : '') + d.msg).join('; ');
    throw new Error(message);
  }
  return data;
}

function money(x) {
  if (!x) return '$0';
  if (x < 0.01) return '<$0.01';
  return '$' + x.toFixed(x < 1 ? 3 : 2);
}
function seconds(ms) { return (ms / 1000).toFixed(1) + 's'; }
function notice(kind, text, items) {
  return el('div', { class: 'notice ' + kind }, text, items && items.length ? el('ul', {}, items.map(i => el('li', { text: i }))) : null);
}
function chip(text, kind = '') { return el('span', { class: 'chip ' + kind, text }); }

const logoutButton = document.getElementById('logout');
if (logoutButton) {
  logoutButton.addEventListener('click', async () => {
    try { await api('/api/logout', { method: 'POST' }); } catch (e) { /* the session is gone either way */ }
    forgetAll();
    location.href = '/login';
  });
}
