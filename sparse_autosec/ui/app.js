const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const codeEl = document.getElementById('code');
const patchedEl = document.getElementById('patched');

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  return await res.json();
}

document.getElementById('trainBtn').addEventListener('click', async () => {
  statusEl.textContent = 'Training...';
  const epochs = Number(document.getElementById('epochs').value || 2);
  const out = await postJson('/api/train', {epochs});
  statusEl.textContent = `Model exported: ${out.model}`;
  resultsEl.textContent = JSON.stringify(out, null, 2);
});

document.getElementById('scanBtn').addEventListener('click', async () => {
  statusEl.textContent = 'Scanning...';
  const out = await postJson('/api/scan', {code: codeEl.value});
  statusEl.textContent = out.patched ? `Patched (${out.patch_signature})` : 'No patch applied';
  patchedEl.value = out.patched_code || '';
  resultsEl.textContent = JSON.stringify(out, null, 2);
});
