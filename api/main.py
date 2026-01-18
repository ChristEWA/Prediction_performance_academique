from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Student Academic Performance Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Charge le modèle pipeline "performance"
model = joblib.load("model.pkl")


# --------- Schéma de données ----------
class StudentFeatures(BaseModel):
    Age: float = Field(..., ge=0, le=120)
    Gender: str = Field(..., description="Male or Female")
    Department: str = Field(..., description="Business / Engineering / Medical / Science")

    Sleep_Duration: float = Field(..., ge=0, le=24)
    Study_Hours: float = Field(..., ge=0, le=24)
    Social_Media_Hours: float = Field(..., ge=0, le=24)
    Physical_Activity: float = Field(..., ge=0, le=300)
    Stress_Level: float = Field(..., ge=1, le=10)


# --------- PAGE WEB ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Prédiction — Performance académique</title>
  <style>
    :root{
      --bg:#0b1220;
      --card:rgba(255,255,255,.08);
      --text:rgba(255,255,255,.92);
      --muted:rgba(255,255,255,.70);
      --border:rgba(255,255,255,.14);
      --shadow:0 20px 60px rgba(0,0,0,.40);
      --accent:#6ee7b7;
      --accent2:#60a5fa;
      --danger:#fb7185;
      --warn:#fbbf24;
      --ok:#34d399;
    }
    *{ box-sizing:border-box; }
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color:var(--text);
      background:
        radial-gradient(1200px 700px at 10% 10%, rgba(96,165,250,.25), transparent 60%),
        radial-gradient(900px 600px at 90% 20%, rgba(110,231,183,.18), transparent 60%),
        radial-gradient(800px 600px at 50% 100%, rgba(251,113,133,.12), transparent 60%),
        var(--bg);
      min-height:100vh;
      padding:32px 16px;
    }
    .wrap{ max-width: 980px; margin: 0 auto; }
    .top{
      display:flex; align-items:flex-start; justify-content:space-between;
      gap:16px; margin-bottom:18px;
    }
    .brand{ display:flex; gap:12px; align-items:center; }
    .logo{
      width:44px; height:44px; border-radius:14px;
      background: linear-gradient(135deg, rgba(110,231,183,.9), rgba(96,165,250,.9));
      box-shadow: 0 10px 30px rgba(96,165,250,.20);
    }
    h1{ font-size:22px; margin:0; letter-spacing:.2px; }
    .subtitle{ margin:4px 0 0 0; color:var(--muted); font-size:13px; line-height:1.35; }
    .pill{
      padding:8px 10px; border:1px solid var(--border); border-radius:999px;
      background: rgba(255,255,255,.06); color:var(--muted); font-size:12px; white-space:nowrap;
    }
    .kbd{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono";
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.06);
      color: rgba(255,255,255,.85);
    }

    .grid{ display:grid; grid-template-columns: 1.25fr .75fr; gap:16px; }
    @media (max-width: 900px){ .grid{ grid-template-columns: 1fr; } }

    .card{
      background:var(--card);
      border:1px solid var(--border);
      border-radius:18px;
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .card-head{
      padding:16px 18px;
      border-bottom:1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,.10), transparent);
    }
    .card-head h2{ margin:0; font-size:16px; letter-spacing:.2px; }
    .card-head p{ margin:6px 0 0 0; color:var(--muted); font-size:13px; line-height:1.4; }
    .card-body{ padding:18px; }

    form{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    @media (max-width: 560px){ form{ grid-template-columns:1fr; } }
    .field{ display:flex; flex-direction:column; gap:6px; }
    label{ font-size:12px; color:var(--muted); }
    input, select{
      width:100%;
      padding:11px 12px;
      border-radius:12px;
      border:1px solid var(--border);
      background: rgba(255,255,255,.06);
      color: var(--text);
      outline: none;
    }
    input::placeholder{ color: rgba(255,255,255,.42); }
    input:focus, select:focus{
      border-color: rgba(96,165,250,.65);
      box-shadow: 0 0 0 4px rgba(96,165,250,.16);
    }
    option{ color:#0b1220; }

    .actions{
      grid-column:1/-1;
      display:flex; gap:10px; align-items:center;
      margin-top:6px;
    }
    .btn{
      appearance:none;
      border:0;
      border-radius:12px;
      padding:12px 14px;
      font-weight:800;
      cursor:pointer;
      color:#0b1220;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      box-shadow: 0 14px 30px rgba(96,165,250,.20);
      min-width: 190px;
    }
    .btn:disabled{ opacity:.55; cursor:not-allowed; }
    .ghost{
      border:1px solid var(--border);
      background: rgba(255,255,255,.06);
      color: var(--text);
      box-shadow:none;
      font-weight:700;
    }
    .hint{ color:var(--muted); font-size:12px; line-height:1.4; }

    .status{
      display:flex; justify-content:space-between; align-items:center; gap:12px;
      padding:14px; border-radius:14px;
      border:1px solid var(--border);
      background: rgba(255,255,255,.06);
    }
    .big{ font-size:16px; font-weight:900; margin:0; letter-spacing:.2px; }
    .muted{ color:var(--muted); font-size:12px; margin:3px 0 0 0; }
    .badge{
      font-size:12px;
      font-weight:900;
      padding:8px 12px;
      border-radius:999px;
      border:1px solid var(--border);
      background: rgba(255,255,255,.06);
      white-space: nowrap;
    }
    .badge.low{ background: rgba(251,113,133,.95); color:#0b1220; border-color:transparent; }
    .badge.mid{ background: rgba(251,191,36,.95); color:#0b1220; border-color:transparent; }
    .badge.high{ background: rgba(52,211,153,.95); color:#0b1220; border-color:transparent; }
    .badge.wait{ color: var(--text); }

    .probBox{
      margin-top: 12px;
      display:flex;
      flex-direction:column;
      gap:10px;
    }
    .probRow{
      display:grid;
      grid-template-columns: 90px 1fr 60px;
      gap: 10px;
      align-items:center;
    }
    .bar{
      height:10px;
      border-radius:999px;
      background: rgba(255,255,255,.10);
      overflow:hidden;
      border: 1px solid var(--border);
    }
    .bar > div{
      height:100%;
      width:0%;
      transition: width .35s ease;
      background: linear-gradient(90deg, rgba(96,165,250,.95), rgba(110,231,183,.95));
    }

    .error{
      border-color: rgba(251,113,133,.55);
      background: rgba(251,113,133,.10);
    }
    .footer{
      margin-top: 14px;
      color: rgba(255,255,255,.55);
      font-size: 12px;
      text-align:center;
    }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">
        <div class="logo"></div>
        <div>
          <h1>Prédiction de la performance académique</h1>
          <p class="subtitle">
            Basée sur <b>la santé mentale</b> (stress) et les <b>habitudes</b> (sommeil, étude, réseaux sociaux, sport).
          </p>
        </div>
      </div>
      <div class="pill">API: <span class="kbd">POST /predict</span></div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-head">
          <h2>Entrées</h2>
          <p>Remplis les champs puis clique sur <b>Prédire</b>.</p>
        </div>
        <div class="card-body">
          <form id="form" onsubmit="return false;">
            <div class="field">
              <label for="age">Âge</label>
              <input type="number" id="age" placeholder="ex: 21" required min="0" max="120" step="1" />
            </div>

            <div class="field">
              <label for="gender">Genre</label>
              <select id="gender" required>
                <option value="Male">Homme</option>
                <option value="Female">Femme</option>
              </select>
            </div>

            <div class="field">
              <label for="dept">Département</label>
              <select id="dept" required>
                <option value="Business">Business</option>
                <option value="Engineering">Engineering</option>
                <option value="Medical">Medical</option>
                <option value="Science">Science</option>
              </select>
            </div>

            <div class="field">
              <label for="sleep">Sommeil (h / jour)</label>
              <input type="number" id="sleep" placeholder="ex: 7.5" required min="0" max="24" step="0.1" />
            </div>

            <div class="field">
              <label for="study">Étude (h / jour)</label>
              <input type="number" id="study" placeholder="ex: 3.0" required min="0" max="24" step="0.1" />
            </div>

            <div class="field">
              <label for="social">Réseaux sociaux (h / jour)</label>
              <input type="number" id="social" placeholder="ex: 3.0" required min="0" max="24" step="0.1" />
            </div>

            <div class="field">
              <label for="sport">Activité physique (min / semaine)</label>
              <input type="number" id="sport" placeholder="ex: 60" required min="0" max="300" step="1" />
            </div>

            <div class="field">
              <label for="stress">Stress (1–10)</label>
              <input type="number" id="stress" placeholder="ex: 4" required min="1" max="10" step="1" />
            </div>

            <div class="actions">
              <button id="btn" class="btn" type="button" onclick="predict()">Prédire</button>
              <button class="btn ghost" type="button" onclick="resetForm()">Réinitialiser</button>
              <div class="hint">Résultat : <b>Low / Medium / High</b> + probabilités.</div>
            </div>
          </form>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h2>Résultat</h2>
          <p>Classe prédite + distribution de probabilités.</p>
        </div>
        <div class="card-body">
          <div id="status" class="status">
            <div>
              <p id="title" class="big">En attente d’une prédiction</p>
              <p id="desc" class="muted">Remplis le formulaire puis clique sur “Prédire”.</p>
            </div>
            <div id="badge" class="badge wait">—</div>
          </div>

          <div class="probBox" id="probBox" style="display:none;">
            <div class="probRow">
              <div class="hint"><b>Low</b></div>
              <div class="bar"><div id="pLow"></div></div>
              <div class="hint" id="tLow">0%</div>
            </div>
            <div class="probRow">
              <div class="hint"><b>Medium</b></div>
              <div class="bar"><div id="pMed"></div></div>
              <div class="hint" id="tMed">0%</div>
            </div>
            <div class="probRow">
              <div class="hint"><b>High</b></div>
              <div class="bar"><div id="pHigh"></div></div>
              <div class="hint" id="tHigh">0%</div>
            </div>
          </div>

          <div class="footer">
            Projet Data Science · FastAPI + sklearn · <span class="kbd">/predict</span>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  function getNumber(id){
    const raw = document.getElementById(id).value;
    if (raw === "" || raw === null || raw === undefined) return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }

  function setLoading(isLoading){
    const btn = document.getElementById("btn");
    btn.disabled = isLoading;
    btn.textContent = isLoading ? "Prédiction..." : "Prédire";
  }

  function badgeClass(label){
    if (label === "Low") return "badge low";
    if (label === "Medium") return "badge mid";
    if (label === "High") return "badge high";
    return "badge wait";
  }

  function setError(message){
    const status = document.getElementById("status");
    const title = document.getElementById("title");
    const desc = document.getElementById("desc");
    const badge = document.getElementById("badge");
    const probBox = document.getElementById("probBox");

    status.classList.add("error");
    title.textContent = "Erreur";
    desc.textContent = message;
    badge.textContent = "!";
    badge.className = "badge wait";
    probBox.style.display = "none";
  }

  function resetForm(){
    document.getElementById("form").reset();
    const status = document.getElementById("status");
    const title = document.getElementById("title");
    const desc = document.getElementById("desc");
    const badge = document.getElementById("badge");
    const probBox = document.getElementById("probBox");

    status.classList.remove("error");
    title.textContent = "En attente d’une prédiction";
    desc.textContent = "Remplis le formulaire puis clique sur “Prédire”.";
    badge.textContent = "—";
    badge.className = "badge wait";
    probBox.style.display = "none";
  }

  function setProbBars(probs){
    // probs = {"Low":0.12,"Medium":0.70,"High":0.18}
    const probBox = document.getElementById("probBox");
    probBox.style.display = "block";

    const low = Math.max(0, Math.min(1, Number(probs.Low ?? 0)));
    const med = Math.max(0, Math.min(1, Number(probs.Medium ?? 0)));
    const high = Math.max(0, Math.min(1, Number(probs.High ?? 0)));

    document.getElementById("pLow").style.width  = (low*100).toFixed(1) + "%";
    document.getElementById("pMed").style.width  = (med*100).toFixed(1) + "%";
    document.getElementById("pHigh").style.width = (high*100).toFixed(1) + "%";

    document.getElementById("tLow").textContent  = (low*100).toFixed(1) + "%";
    document.getElementById("tMed").textContent  = (med*100).toFixed(1) + "%";
    document.getElementById("tHigh").textContent = (high*100).toFixed(1) + "%";
  }

  async function predict(){
    const payload = {
      Age: getNumber("age"),
      Gender: document.getElementById("gender").value,
      Department: document.getElementById("dept").value,
      Sleep_Duration: getNumber("sleep"),
      Study_Hours: getNumber("study"),
      Social_Media_Hours: getNumber("social"),
      Physical_Activity: getNumber("sport"),
      Stress_Level: getNumber("stress"),
    };

    const required = ["Age","Sleep_Duration","Study_Hours","Social_Media_Hours","Physical_Activity","Stress_Level"];
    const missing = required.filter(k => payload[k] === null);
    if (missing.length){
      setError("Champs manquants/invalides : " + missing.join(", "));
      return;
    }

    try{
      setLoading(true);
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok){
        const txt = await res.text();
        setError("Erreur API (" + res.status + ") : " + txt);
        return;
      }

      const data = await res.json();

      const status = document.getElementById("status");
      const title = document.getElementById("title");
      const desc = document.getElementById("desc");
      const badge = document.getElementById("badge");

      status.classList.remove("error");
      title.textContent = "Performance prédite : " + data.predicted_profile;
      desc.textContent = "Estimation basée sur les habitudes et le stress.";
      badge.textContent = data.predicted_profile;
      badge.className = badgeClass(data.predicted_profile);

      if (data.probabilities){
        setProbBars(data.probabilities);
      } else {
        document.getElementById("probBox").style.display = "none";
      }

    } catch(e){
      setError("Erreur JS / Connexion : " + e);
    } finally {
      setLoading(false);
    }
  }
</script>
</body>
</html>
    """


# --------- API ----------
@app.post("/predict")
def predict(payload: StudentFeatures):
    try:
        df = pd.DataFrame([payload.model_dump()])

        pred = model.predict(df)[0]
        predicted_profile = str(pred)

        probs = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(df)[0]
            if hasattr(model, "classes_"):
                probs = {str(c): float(v) for c, v in zip(model.classes_, p)}

                # On force les clés Low/Medium/High si possible (affichage stable)
                probs = {
                    "Low": float(probs.get("Low", 0.0)),
                    "Medium": float(probs.get("Medium", 0.0)),
                    "High": float(probs.get("High", 0.0)),
                }
            else:
                probs = None

        return {
            "predicted_profile": predicted_profile,
            "probabilities": probs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")
