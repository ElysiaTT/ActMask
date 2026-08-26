(() => {
  const $ = (id) => document.getElementById(id);
  const state = { rows: [], currentIndex: 0, tool: 'brush', brush: 24, zoom: 1, panX: 0, panY: 0, dirty: false, drawing: false, last: null, undo: [], annotator: localStorage.getItem('robomindAnnotator') || '', overlay: true };
  const imageCanvas = $('imageCanvas');
  const maskCanvas = $('maskCanvas');
  const maskDataCanvas = $('maskDataCanvas');
  const imageCtx = imageCanvas.getContext('2d');
  const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
  const dataCtx = maskDataCanvas.getContext('2d', { willReadFrequently: true });
  const stage = $('canvasStage');
  let toastTimer;

  const current = () => state.rows[state.currentIndex];
  const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const taskName = (row) => row.task_id.replace(/^robogene_/, '').replaceAll('_', ' ');
  const showToast = (message, error = false) => { const box = $('toast'); box.textContent = message; box.classList.toggle('error', error); box.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => box.classList.remove('show'), 3400); };
  const setDirty = (dirty) => { state.dirty = dirty; const chip = $('saveState'); chip.textContent = dirty ? '未保存' : (current()?.completed ? '已保存' : '未标注'); chip.classList.toggle('saved', !dirty && !!current()?.completed); };
  const updateStage = () => { stage.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`; $('zoomText').textContent = `${Math.round(state.zoom * 100)}%`; };
  const resetView = () => { state.zoom = 1; state.panX = 0; state.panY = 0; updateStage(); };
  const setTool = (tool) => { state.tool = tool; $('brushBtn').classList.toggle('active', tool === 'brush'); $('eraserBtn').classList.toggle('active', tool === 'eraser'); $('toolText').textContent = tool === 'brush' ? '画笔' : '橡皮'; maskCanvas.style.cursor = tool === 'brush' ? 'crosshair' : 'cell'; };
  const renderOverlay = () => { const data = dataCtx.getImageData(0, 0, maskDataCanvas.width, maskDataCanvas.height); const visual = maskCtx.createImageData(maskCanvas.width, maskCanvas.height); for (let i = 0; i < data.data.length; i += 4) { if (data.data[i] > 127) { visual.data[i] = 255; visual.data[i + 1] = 64; visual.data[i + 2] = 80; visual.data[i + 3] = 142; } } maskCtx.putImageData(visual, 0, 0); };
  const pushUndo = () => { state.undo.push(dataCtx.getImageData(0, 0, maskDataCanvas.width, maskDataCanvas.height)); if (state.undo.length > 30) state.undo.shift(); };
  const clearMask = () => { dataCtx.clearRect(0, 0, maskDataCanvas.width, maskDataCanvas.height); maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height); };
  const getMaskDataUrl = () => maskDataCanvas.toDataURL('image/png');

  const refreshProgress = () => {
    const done = state.rows.filter((row) => row.completed).length;
    $('progressText').textContent = `已完成 ${done} / ${state.rows.length}`;
    $('progressBar').style.width = `${state.rows.length ? (done / state.rows.length) * 100 : 0}%`;
  };
  const renderFrameList = () => {
    const filter = $('filterSelect').value;
    const shown = state.rows.map((row, index) => ({ row, index })).filter(({row}) => filter === 'all' || (filter === 'completed' ? row.completed : !row.completed));
    $('listCount').textContent = `${shown.length} 帧`;
    $('frameList').innerHTML = shown.map(({row, index}) => `<button class="frame-item ${index === state.currentIndex ? 'active' : ''}" data-index="${index}"><div class="frame-title"><i class="dot ${row.completed ? 'done' : ''}"></i> ${escapeHtml(taskName(row))}</div><div class="frame-sub"><span>ep ${String(row.episode_index).padStart(6,'0')}</span><span>帧 ${row.frame_index}</span></div></button>`).join('') || '<p class="muted">没有符合筛选条件的帧</p>';
    document.querySelectorAll('.frame-item').forEach((button) => button.addEventListener('click', () => switchFrame(Number(button.dataset.index))));
  };
  const updateMeta = () => {
    const row = current();
    $('positionText').textContent = `第 ${state.currentIndex + 1} / ${state.rows.length} 帧 · ${row.completed ? '已保存' : '待标注'}`;
    $('taskTitle').textContent = taskName(row);
    $('frameMeta').textContent = `轨迹 ${row.episode_id} · 原始帧 ${row.frame_index} · 采样：${row.sampling_stratum}`;
    $('noteInput').value = row.note || '';
    setDirty(false);
  };
  const imageLoad = (src) => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src; });
  const maskLoad = async (row) => {
    clearMask();
    if (!row.completed) return;
    const response = await fetch(`/api/mask/${encodeURIComponent(row.id)}`, {cache:'no-store'});
    if (!response.ok) return;
    const blob = await response.blob(); const source = await imageLoad(URL.createObjectURL(blob));
    dataCtx.drawImage(source, 0, 0, maskDataCanvas.width, maskDataCanvas.height); renderOverlay();
  };
  const loadFrame = async () => {
    const row = current(); $('loading').style.display = 'block';
    try {
      const image = await imageLoad(`/api/image/${encodeURIComponent(row.id)}`);
      imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
      imageCtx.drawImage(image, 0, 0, imageCanvas.width, imageCanvas.height);
      await maskLoad(row); state.undo = []; updateMeta(); renderFrameList();
    } catch (error) { showToast(`图像载入失败：${error}`, true); }
    finally { $('loading').style.display = 'none'; }
  };
  const switchFrame = async (index) => {
    if (index < 0 || index >= state.rows.length || index === state.currentIndex) return;
    if (state.dirty && !window.confirm('本帧有未保存修改。确定放弃并切换吗？')) return;
    state.currentIndex = index; await loadFrame();
  };
  const nextPending = () => {
    for (let offset = 1; offset <= state.rows.length; offset += 1) { const i = (state.currentIndex + offset) % state.rows.length; if (!state.rows[i].completed) return switchFrame(i); }
    showToast('全部帧都已经保存。');
  };
  const point = (event) => { const rect = maskCanvas.getBoundingClientRect(); return {x: (event.clientX - rect.left) * maskCanvas.width / rect.width, y: (event.clientY - rect.top) * maskCanvas.height / rect.height}; };
  const draw = (from, to) => { const mode = state.tool === 'brush' ? 'source-over' : 'destination-out'; const stroke = (ctx, color) => { ctx.save(); ctx.globalCompositeOperation = mode; ctx.strokeStyle = color; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = state.brush; ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke(); ctx.restore(); }; stroke(dataCtx, '#ffffff'); stroke(maskCtx, 'rgba(255,64,80,.56)'); };
  const save = async () => {
    const row = current(); const note = $('noteInput').value; const annotator = $('annotatorInput').value;
    const pixels = dataCtx.getImageData(0, 0, maskDataCanvas.width, maskDataCanvas.height).data;
    let foreground = 0; for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 127) foreground += 1;
    if (foreground < 16) { showToast('当前掩码为空，请涂出可见的机械臂和夹爪。', true); return; }
    $('saveBtn').disabled = true; $('saveBtn').textContent = '正在保存…';
    try {
      const response = await fetch(`/api/save/${encodeURIComponent(row.id)}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({png_base64:getMaskDataUrl(), annotator, note})});
      const value = await response.json(); if (!response.ok) throw new Error(value.error || '保存失败');
      Object.assign(row, {completed:true, note, annotator, foreground_pixels:value.result.foreground_pixels, saved_at:value.result.saved_at}); localStorage.setItem('robomindAnnotator', annotator); setDirty(false); refreshProgress(); renderFrameList(); updateMeta(); showToast(`已保存：${value.result.foreground_pixels.toLocaleString()} 个机器人像素`);
    } catch (error) { showToast(`保存失败：${error.message}`, true); }
    finally { $('saveBtn').disabled = false; $('saveBtn').innerHTML = '保存本帧 <kbd>Ctrl</kbd>+<kbd>S</kbd>'; }
  };
  const bindCanvas = () => {
    maskCanvas.addEventListener('pointerdown', (event) => { if (event.button !== 0) return; pushUndo(); state.drawing = true; state.last = point(event); maskCanvas.setPointerCapture(event.pointerId); draw(state.last, state.last); setDirty(true); });
    maskCanvas.addEventListener('pointermove', (event) => { if (!state.drawing) return; const now = point(event); draw(state.last, now); state.last = now; });
    const stop = () => { state.drawing = false; state.last = null; }; maskCanvas.addEventListener('pointerup', stop); maskCanvas.addEventListener('pointercancel', stop);
  };
  const bindControls = () => {
    $('annotatorInput').value = state.annotator; $('brushSize').addEventListener('input', (event) => { state.brush = Number(event.target.value); $('brushSizeValue').textContent = `${state.brush} px`; });
    $('brushBtn').onclick = () => setTool('brush'); $('eraserBtn').onclick = () => setTool('eraser');
    $('undoBtn').onclick = () => { const prior = state.undo.pop(); if (!prior) return; dataCtx.putImageData(prior, 0, 0); renderOverlay(); setDirty(true); };
    $('clearBtn').onclick = () => { if (!window.confirm('确定清空当前画布吗？未保存的内容将消失。')) return; pushUndo(); clearMask(); setDirty(true); };
    $('overlayToggle').onchange = (event) => { state.overlay = event.target.checked; maskCanvas.style.visibility = state.overlay ? 'visible' : 'hidden'; };
    $('previousBtn').onclick = () => switchFrame(state.currentIndex - 1); $('nextBtn').onclick = () => switchFrame(state.currentIndex + 1); $('nextPendingBtn').onclick = nextPending;
    $('zoomInBtn').onclick = () => { state.zoom = Math.min(3, state.zoom + .15); updateStage(); }; $('zoomOutBtn').onclick = () => { state.zoom = Math.max(.4, state.zoom - .15); updateStage(); }; $('resetViewBtn').onclick = resetView; $('saveBtn').onclick = save; $('filterSelect').onchange = renderFrameList;
    window.addEventListener('keydown', (event) => { const tag = document.activeElement?.tagName; if (['INPUT','TEXTAREA','SELECT'].includes(tag) && !(event.ctrlKey || event.metaKey)) return; if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); $('undoBtn').click(); } else if (event.key.toLowerCase() === 'b') setTool('brush'); else if (event.key.toLowerCase() === 'e') setTool('eraser'); else if (event.key.toLowerCase() === 'a' || event.key === 'ArrowLeft') switchFrame(state.currentIndex - 1); else if (event.key.toLowerCase() === 'd' || event.key === 'ArrowRight') switchFrame(state.currentIndex + 1); else if (event.key.toLowerCase() === 'n') nextPending(); else if (event.key === '[') { state.brush = Math.max(2, state.brush - 3); $('brushSize').value = state.brush; $('brushSizeValue').textContent = `${state.brush} px`; } else if (event.key === ']') { state.brush = Math.min(100, state.brush + 3); $('brushSize').value = state.brush; $('brushSizeValue').textContent = `${state.brush} px`; } else if (event.key.toLowerCase() === 'h') { $('overlayToggle').click(); } });
  };
  const start = async () => { bindControls(); bindCanvas(); setTool('brush'); try { const response = await fetch('/api/manifest'); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || '清单载入失败'); state.rows = payload.rows; const firstPending = state.rows.findIndex((row) => !row.completed); state.currentIndex = firstPending >= 0 ? firstPending : 0; refreshProgress(); renderFrameList(); resetView(); await loadFrame(); } catch (error) { $('loading').textContent = `初始化失败：${error.message}`; showToast(`初始化失败：${error.message}`, true); } };
  start();
})();
