"""
app.py  —  Interfaz web local con beam search
Uso: python app.py  →  http://localhost:5000
"""

import io, os
import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

import config
from dataset import Vocabulary, get_transforms
from model import ImageCaptioner

VOCAB_PATH = "vocab.pkl"
CKPT_PATH  = os.path.join(config.CHECKPOINT_DIR, "best.pth")

print(f"[app] Dispositivo: {config.DEVICE}")
vocab = Vocabulary.load(VOCAB_PATH)
model = ImageCaptioner(len(vocab)).to(config.DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=config.DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print("[app] Modelo listo ✓")

transform = get_transforms(train=False)

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ImageCaption · AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
  <style>
    :root{--bg:#0a0a0f;--surface:#13131a;--border:#2a2a3a;--accent:#7c6af7;--accent2:#e05fe0;--text:#e8e8f0;--muted:#6a6a80;--success:#4ade9f;--r:12px}
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 20px 80px}
    body::before{content:'';position:fixed;top:-200px;left:50%;transform:translateX(-50%);width:800px;height:400px;background:radial-gradient(ellipse,rgba(124,106,247,.18) 0%,transparent 70%);pointer-events:none;z-index:0}
    .container{position:relative;z-index:1;width:100%;max-width:700px}
    header{text-align:center;margin-bottom:48px}
    .logo{display:inline-block;font-size:11px;font-family:'Space Mono',monospace;letter-spacing:4px;text-transform:uppercase;color:var(--accent);border:1px solid var(--accent);padding:5px 14px;border-radius:99px;margin-bottom:20px}
    h1{font-size:clamp(2rem,6vw,3.2rem);font-weight:800;line-height:1.1;background:linear-gradient(135deg,#fff 30%,var(--accent2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .subtitle{margin-top:12px;color:var(--muted);font-size:15px;font-family:'Space Mono',monospace}
    .drop-zone{border:2px dashed var(--border);border-radius:var(--r);padding:60px 30px;text-align:center;cursor:pointer;transition:border-color .25s,background .25s;background:var(--surface);position:relative}
    .drop-zone:hover,.drop-zone.over{border-color:var(--accent);background:rgba(124,106,247,.06)}
    .drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
    .drop-icon{font-size:40px;margin-bottom:14px;display:block}
    .drop-text{font-size:15px;color:var(--muted);font-family:'Space Mono',monospace}
    .drop-text strong{color:var(--accent)}
    #preview-section{display:none;margin-top:32px}
    .preview-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}
    .preview-img-wrap{width:100%;max-height:440px;overflow:hidden;background:#000;display:flex;align-items:center;justify-content:center}
    .preview-img-wrap img{width:100%;height:100%;object-fit:contain;max-height:440px}
    .result-area{padding:24px 28px;border-top:1px solid var(--border)}
    .result-label{font-size:10px;font-family:'Space Mono',monospace;letter-spacing:3px;text-transform:uppercase;color:var(--accent);margin-bottom:10px}
    .caption-text{font-size:1.25rem;font-weight:700;line-height:1.4;min-height:2em}
    .caption-text.loading{color:var(--muted);font-size:14px;font-family:'Space Mono',monospace;font-weight:400}
    .caption-text.success{color:var(--text)}
    .caption-text.error{color:#f87171;font-size:14px}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
    .cursor{animation:blink 1s infinite}
    .btn-row{margin-top:20px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    .btn{padding:10px 22px;border-radius:8px;font-family:'Space Mono',monospace;font-size:13px;cursor:pointer;border:1px solid var(--border);background:transparent;color:var(--text);transition:background .2s,border-color .2s}
    .btn:hover{background:rgba(255,255,255,.06);border-color:var(--accent)}
    .btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
    .btn.primary:hover{background:#9580ff}
    .beam-select{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 12px;font-family:'Space Mono',monospace;font-size:13px;cursor:pointer}
    .beam-select:focus{outline:none;border-color:var(--accent)}
    .stats{margin-top:14px;display:flex;gap:20px;flex-wrap:wrap}
    .stat{font-family:'Space Mono',monospace;font-size:11px;color:var(--muted)}
    .stat span{color:var(--success)}
    #history{margin-top:48px}
    .history-title{font-size:11px;font-family:'Space Mono',monospace;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:16px}
    .history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px}
    .history-item{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;transition:border-color .2s}
    .history-item:hover{border-color:var(--accent)}
    .history-item img{width:100%;height:120px;object-fit:cover;display:block}
    .history-caption{padding:10px;font-size:12px;color:var(--muted);line-height:1.4;font-family:'Space Mono',monospace}
    footer{margin-top:64px;text-align:center;font-family:'Space Mono',monospace;font-size:11px;color:var(--muted)}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">ResNet-50 + LSTM · Beam Search</div>
    <h1>Image Captioning</h1>
    <p class="subtitle">// Sube una imagen y la IA la describe</p>
  </header>

  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept="image/*"/>
    <span class="drop-icon">🖼</span>
    <p class="drop-text">Arrastra una imagen aquí<br>o haz clic para <strong>seleccionar</strong></p>
  </div>

  <section id="preview-section">
    <div class="preview-card">
      <div class="preview-img-wrap"><img id="previewImg" src="" alt="preview"/></div>
      <div class="result-area">
        <div class="result-label">Descripción generada</div>
        <div class="caption-text" id="captionText">—</div>
        <div class="stats" id="statsBar" style="display:none">
          <span class="stat">Tiempo: <span id="statTime">—</span></span>
          <span class="stat">Tokens: <span id="statTokens">—</span></span>
          <span class="stat">Beam: <span id="statBeam">—</span></span>
          <span class="stat">Device: <span id="statDevice">{{ device }}</span></span>
        </div>
        <div class="btn-row">
          <button class="btn primary" id="retryBtn">↺ Reintentar</button>
          <select class="beam-select" id="beamSelect" title="Beam size">
            <option value="1">Greedy</option>
            <option value="3">Beam 3</option>
            <option value="5" selected>Beam 5</option>
            <option value="7">Beam 7</option>
          </select>
          <button class="btn" id="copyBtn">⎘ Copiar</button>
          <button class="btn" id="newBtn">+ Nueva imagen</button>
        </div>
      </div>
    </div>
  </section>

  <section id="history" style="display:none">
    <div class="history-title">// Historial de sesión</div>
    <div class="history-grid" id="historyGrid"></div>
  </section>
</div>

<footer>Modelo local · {{ device }} · vocab={{ vocab_size }} tokens</footer>

<script>
  const dropZone=document.getElementById('dropZone'),fileInput=document.getElementById('fileInput'),
        preview=document.getElementById('preview-section'),previewImg=document.getElementById('previewImg'),
        captionEl=document.getElementById('captionText'),statsBar=document.getElementById('statsBar'),
        statTime=document.getElementById('statTime'),statTokens=document.getElementById('statTokens'),
        statBeam=document.getElementById('statBeam'),history=document.getElementById('history'),
        historyGrid=document.getElementById('historyGrid'),beamSelect=document.getElementById('beamSelect');
  let currentFile=null,currentDataURL=null;

  dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('over')});
  dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('over'));
  dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.classList.remove('over');if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0])});
  fileInput.addEventListener('change',()=>{if(fileInput.files[0])handleFile(fileInput.files[0])});

  function handleFile(file){
    if(!file.type.startsWith('image/')){alert('Solo imágenes.');return}
    currentFile=file;
    const r=new FileReader();
    r.onload=e=>{
      currentDataURL=e.target.result;
      previewImg.src=currentDataURL;
      preview.style.display='block';
      dropZone.style.display='none';
      statsBar.style.display='none';
      runCaption(file);
    };
    r.readAsDataURL(file);
  }

  function runCaption(file){
    const beam=beamSelect.value;
    captionEl.className='caption-text loading';
    captionEl.innerHTML=`Analizando (beam=${beam})<span class="cursor">_</span>`;
    const t0=Date.now(),fd=new FormData();
    fd.append('image',file);fd.append('beam',beam);
    fetch('/caption',{method:'POST',body:fd})
      .then(r=>r.json())
      .then(data=>{
        const elapsed=((Date.now()-t0)/1000).toFixed(2);
        if(data.error){captionEl.className='caption-text error';captionEl.textContent='⚠ '+data.error}
        else{
          captionEl.className='caption-text success';captionEl.textContent=data.caption;
          statTime.textContent=elapsed+'s';statTokens.textContent=data.caption.split(' ').length;
          statBeam.textContent=beam==='1'?'greedy':'beam '+beam;
          statsBar.style.display='flex';
          addToHistory(currentDataURL,data.caption);
        }
      })
      .catch(err=>{captionEl.className='caption-text error';captionEl.textContent='⚠ '+err});
  }

  document.getElementById('retryBtn').addEventListener('click',()=>{if(currentFile)runCaption(currentFile)});
  document.getElementById('copyBtn').addEventListener('click',()=>{
    navigator.clipboard.writeText(captionEl.textContent).then(()=>{
      const b=document.getElementById('copyBtn');b.textContent='✓ Copiado';
      setTimeout(()=>b.textContent='⎘ Copiar',1800);
    });
  });
  document.getElementById('newBtn').addEventListener('click',()=>{
    preview.style.display='none';dropZone.style.display='block';fileInput.value='';currentFile=null;
  });

  function addToHistory(dataURL,caption){
    history.style.display='block';
    const item=document.createElement('div');item.className='history-item';
    item.innerHTML=`<img src="${dataURL}" alt=""/><div class="history-caption">${caption}</div>`;
    historyGrid.prepend(item);
  }
</script>
</body>
</html>"""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

@app.route("/")
def index():
    return render_template_string(HTML, device=str(config.DEVICE), vocab_size=len(vocab))

@app.route("/caption", methods=["POST"])
def caption_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió imagen."}), 400
    file      = request.files["image"]
    beam_size = int(request.form.get("beam", 5))
    try:
        img    = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            caption = model.caption(tensor, vocab, beam_size=beam_size)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[app] Servidor en http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)