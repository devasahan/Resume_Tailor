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

function adminHeaders() {
  const token = sessionStorage.getItem('adminToken');
  return token ? { 'X-Admin-Token': token } : {};
}

async function api(path, { method = 'GET', body = null, form = null, admin = false } = {}) {
  const headers = admin ? adminHeaders() : {};
  const init = { method, headers };
  if (form) init.body = form;
  else if (body !== null) { headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
  const response = await fetch(path, init);
  let data = null;
  try { data = await response.json(); } catch (e) { /* not JSON */ }
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
