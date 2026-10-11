from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pipeline

os.environ["RISK_PROFILE_PATH"] = "risk_profile.json"
APP_TITLE = "CT Surgical Planning Report Server"

SEG_WEIGHTS = os.getenv("SEGMENTATION_WEIGHTS", "unet_final.pth")
LANDMARK_WEIGHTS = os.getenv("LANDMARK_WEIGHTS", "best_landmark_model.pth")
DATASET_ROOT = os.getenv("DATASET_ROOT", "").strip() or None
RISK_PROFILE_PATH = os.getenv("RISK_PROFILE_PATH", "").strip() or None
REPORT_DIR = os.getenv("REPORT_DIR", "reports")

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

DATASET_LINK = "https://zenodo.org/records/3355272"


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>CT Surgical Planning Report Server</title>
      <style>
        * {{
          box-sizing: border-box;
        }}

        html, body {{
          margin: 0;
          padding: 0;
          width: 100%;
          min-height: 100%;
        }}

        body {{
          min-height: 100vh;
          font-family: Arial, sans-serif;
          background-image: url("/static/background.jpeg");
          background-size: cover;
          background-position: center center;
          background-repeat: no-repeat;
          background-attachment: fixed;
          overflow-x: hidden;
        }}

        .page {{
          min-height: 100vh;
          display: flex;
          align-items: stretch;
        }}

        .sidebar {{
          width: 380px;
          max-width: 92vw;
          min-height: 100vh;
          padding: 100px 20px 22px 20px;
          background: linear-gradient(
            to right,
            rgba(0, 0, 0, 0.30),
            rgba(0, 0, 0, 0.12)
          );
          backdrop-filter: blur(4px);
          -webkit-backdrop-filter: blur(4px);
          border-right: 1px solid rgba(255, 255, 255, 0.10);
          display: flex;
          flex-direction: column;
          gap: 16px;
        }}

        .title {{
          margin: 0;
          font-size: 28px;
          line-height: 1.15;
          color: #fff;
          text-shadow: 0 2px 10px rgba(0, 0, 0, 0.40);
        }}

        .subtitle {{
          margin: 8px 0 0 0;
          color: rgba(255, 255, 255, 0.92);
          line-height: 1.5;
          font-size: 14px;
          text-shadow: 0 1px 6px rgba(0, 0, 0, 0.28);
        }}

        .panel {{
          border: 1px solid rgba(255, 255, 255, 0.18);
          border-radius: 18px;
          padding: 16px;
          background: rgba(255, 255, 255, 0.12);
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
        }}

        .panel h2 {{
          margin: 0 0 14px 0;
          font-size: 18px;
          color: #fff;
          text-shadow: 0 1px 6px rgba(0, 0, 0, 0.26);
        }}

        .field {{
          margin-bottom: 14px;
        }}

        .label {{
          display: block;
          margin-bottom: 8px;
          font-weight: 600;
          color: #fff;
          font-size: 14px;
          text-shadow: 0 1px 5px rgba(0, 0, 0, 0.22);
        }}

        input[type="text"] {{
          width: 100%;
          padding: 11px 13px;
          border: 1px solid rgba(255, 255, 255, 0.45);
          border-radius: 12px;
          font-size: 14px;
          outline: none;
          background: rgba(255, 255, 255, 0.86);
          color: #111;
        }}

        input[type="text"]:focus {{
          border-color: #ffffff;
          box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.14);
        }}

        .file-box {{
          border: 1px dashed rgba(255, 255, 255, 0.55);
          border-radius: 14px;
          padding: 12px;
          background: rgba(255, 255, 255, 0.08);
        }}

        input[type="file"] {{
          width: 100%;
          font-size: 14px;
          color: #fff;
        }}

        input[type="file"]::file-selector-button {{
          border: none;
          padding: 9px 12px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.92);
          color: #111;
          cursor: pointer;
          margin-right: 10px;
        }}

        .file-name {{
          margin-top: 8px;
          font-size: 13px;
          color: #fff;
          background: rgba(0, 0, 0, 0.20);
          border: 1px solid rgba(255, 255, 255, 0.14);
          border-radius: 10px;
          padding: 8px 10px;
          word-break: break-word;
          min-height: 34px;
          display: flex;
          align-items: center;
        }}

        .file-name.empty {{
          color: rgba(255, 255, 255, 0.75);
        }}

        .check-row {{
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 10px 0 14px;
          color: #fff;
          font-size: 14px;
          text-shadow: 0 1px 5px rgba(0, 0, 0, 0.22);
        }}

        button {{
          width: 100%;
          border: none;
          border-radius: 14px;
          padding: 12px 16px;
          background: rgba(17, 24, 39, 0.92);
          color: white;
          font-size: 15px;
          font-weight: 700;
          cursor: pointer;
          transition: 0.2s ease;
        }}

        button:hover {{
          transform: translateY(-1px);
          background: rgba(15, 23, 42, 0.98);
        }}

        .note {{
          margin-top: auto;
          color: rgba(255, 255, 255, 0.92);
          font-size: 13px;
          line-height: 1.5;
          text-shadow: 0 1px 5px rgba(0, 0, 0, 0.22);
        }}

        .hero-space {{
          flex: 1;
          min-height: 100vh;
          padding: 20px 50px 80px 50px;
          position: relative;
        }}

        .info-card {{
          max-width: 620px;
          background: rgba(0, 0, 0, 0.22);
          border: 1px solid rgba(255, 255, 255, 0.16);
          border-radius: 22px;
          padding: 28px 30px;
          color: white;
          text-shadow: 0 1px 5px rgba(0, 0, 0, 0.35);
          backdrop-filter: blur(6px);
          -webkit-backdrop-filter: blur(6px);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
          line-height: 1.6;
          position: sticky;
          top: 120px;
          margin-left: auto;
        }}

        .info-card h2 {{
          margin: 0 0 14px 0;
          font-size: 22px;
        }}

        .info-card p {{
          margin: 0 0 14px 0;
          font-size: 15px;
        }}

        .info-card a {{
          color: #d7f0ff;
          text-decoration: underline;
          word-break: break-word;
        }}

        .top-banner {{
          position: fixed;
          top: 20px;
          left: calc(50% + 15px);
          transform: translateX(-50%);
          color: white;
          font-size: 22px;
          font-weight: 700;
          letter-spacing: .5px;
          text-shadow: 0 3px 12px rgba(0, 0, 0, 0.7);
          background: rgba(0, 0, 0, 0.15);
          backdrop-filter: blur(5px);
          -webkit-backdrop-filter: blur(5px);
          padding: 10px 22px;
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.15);
          z-index: 1000;
        }}

        @media (max-width: 900px) {{
          .page {{
            flex-direction: column;
          }}

          .sidebar {{
            width: 100%;
            max-width: 100%;
            min-height: auto;
            border-right: none;
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
          }}

          .hero-space {{
            padding: 20px;
          }}

          .info-card {{
            position: relative;
            top: auto;
            margin-left: 0;
            margin-top: 20px;
            max-width: 100%;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="top-banner">
        Project Research Prototype
      </div>

      <div class="page">
        <div class="sidebar">
          <div>
            <h1 class="title">CT Surgical Planning Report Server</h1>
            <p class="subtitle">
              Upload CT scan files, optional mask and description, then run prediction or generate the PDF report.
            </p>
          </div>

          <form class="panel" action="/predict" method="post" enctype="multipart/form-data">
            <h2>Predict only</h2>

            <div class="field">
              <label class="label">Subject name</label>
              <input type="text" name="subject" placeholder="uploaded_case" />
            </div>

            <div class="field">
              <label class="label">CT scan</label>
              <div class="file-box">
                <input id="predict_ct" type="file" name="ct_file" accept=".nii,.nii.gz" required />
                <div id="predict_ct_name" class="file-name empty">No file chosen</div>
              </div>
            </div>

            <button type="submit">Generate Prediction Report</button>
          </form>

          <form class="panel" action="/generate-report" method="post" enctype="multipart/form-data">
            <h2>Generate report</h2>

            <div class="field">
              <label class="label">Subject name</label>
              <input type="text" name="subject" placeholder="uploaded_case" />
            </div>

            <div class="field">
              <label class="label">CT scan</label>
              <div class="file-box">
                <input id="report_ct" type="file" name="ct_file" accept=".nii,.nii.gz" required />
                <div id="report_ct_name" class="file-name empty">No file chosen</div>
              </div>
            </div>

            <div class="field">
              <label class="label">Mask</label>
              <div class="file-box">
                <input id="report_mask" type="file" name="mask_file" accept=".nii,.nii.gz" />
                <div id="report_mask_name" class="file-name empty">No file chosen</div>
              </div>
            </div>

            <div class="field">
              <label class="label">Description</label>
              <div class="file-box">
                <input id="report_desc" type="file" name="desc_file" accept=".json" />
                <div id="report_desc_name" class="file-name empty">No file chosen</div>
              </div>
            </div>

            <label class="check-row">
              <input type="checkbox" name="show_in_notebook" value="true" />
              Show in notebook
            </label>

            <button type="submit">Generate PDF Report</button>
          </form>
        </div>

        <div class="hero-space">
          <div class="info-card">
            <h2>Project Description</h2>
            <p>
              This research prototype focuses on anatomical complexity assessment for cochlear implantation.
              It supports pre-operative screening and decision support by giving the surgeon an early impression of the case
              and acting as a second-eye quality check.
            </p>
            <p>
              Dataset used in this project:
              <a href="{DATASET_LINK}" target="_blank" rel="noopener noreferrer">{DATASET_LINK}</a>
            </p>
          </div>
        </div>
      </div>

      <script>
        function bindFileName(inputId, labelId) {{
          const input = document.getElementById(inputId);
          const label = document.getElementById(labelId);

          input.addEventListener("change", () => {{
            if (input.files && input.files.length > 0) {{
              label.textContent = input.files[0].name;
              label.classList.remove("empty");
            }} else {{
              label.textContent = "No file chosen";
              label.classList.add("empty");
            }}
          }});
        }}

        bindFileName("predict_ct", "predict_ct_name");
        bindFileName("report_ct", "report_ct_name");
        bindFileName("report_mask", "report_mask_name");
        bindFileName("report_desc", "report_desc_name");
      </script>
    </body>
    </html>
    """


ARTIFACTS = {
    "device": None,
    "segmentation_model": None,
    "landmark_model": None,
    "risk_profile": None,
}


def _cleanup_path(path: str):
    try:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    except Exception:
        pass


def _save_upload(upload: UploadFile, folder: str, filename: Optional[str] = None) -> str:
    os.makedirs(folder, exist_ok=True)
    name = filename or upload.filename or "upload.bin"
    out_path = os.path.join(folder, name)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return out_path


@app.on_event("startup")
def load_artifacts():
    device = "cuda" if pipeline.torch.cuda.is_available() else "cpu"

    if not os.path.exists(SEG_WEIGHTS):
        raise RuntimeError(f"Segmentation weights not found: {SEG_WEIGHTS}")
    if not os.path.exists(LANDMARK_WEIGHTS):
        raise RuntimeError(f"Landmark weights not found: {LANDMARK_WEIGHTS}")

    seg_model = pipeline.load_segmentation_model(SEG_WEIGHTS, device)
    landmark_model = pipeline.load_landmark_model(LANDMARK_WEIGHTS, device)

    if RISK_PROFILE_PATH and os.path.exists(RISK_PROFILE_PATH):
        risk_profile = pipeline.load_risk_profile(RISK_PROFILE_PATH)
    elif DATASET_ROOT:
        risk_profile = pipeline.load_risk_profile(
            dataset_root=DATASET_ROOT,
            segmentation_model=seg_model,
            landmark_model=landmark_model,
            device=device,
        )
    else:
        risk_profile = pipeline.load_risk_profile()

    ARTIFACTS["device"] = device
    ARTIFACTS["segmentation_model"] = seg_model
    ARTIFACTS["landmark_model"] = landmark_model
    ARTIFACTS["risk_profile"] = risk_profile


@app.get("/health")
def health():
    ready = all(ARTIFACTS[k] is not None for k in ("device", "segmentation_model", "landmark_model", "risk_profile"))
    return {
        "status": "ok" if ready else "not_ready",
        "dataset_root": DATASET_ROOT,
        "seg_weights": SEG_WEIGHTS,
        "landmark_weights": LANDMARK_WEIGHTS,
        "report_dir": REPORT_DIR,
    }


@app.post("/generate-report")
async def generate_report(
    background_tasks: BackgroundTasks,
    ct_file: UploadFile = File(...),
    mask_file: Optional[UploadFile] = File(None),
    desc_file: Optional[UploadFile] = File(None),
    subject: str = Form("uploaded_case"),
    show_in_notebook: bool = Form(False),
):
    if ARTIFACTS["segmentation_model"] is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    workdir = tempfile.mkdtemp(prefix="ct_report_")
    cleanup_paths = [workdir]

    try:
        ct_path = _save_upload(ct_file, workdir, ct_file.filename or "ct.nii.gz")
        if not (ct_path.endswith(".nii") or ct_path.endswith(".nii.gz")):
            pass

        nib.load(ct_path)

        sample = {
            "subject": subject,
            "ct": ct_path,
        }

        if mask_file is not None:
            mask_path = _save_upload(mask_file, workdir, mask_file.filename or "mask.nii.gz")
            nib.load(mask_path)
            sample["mask"] = mask_path
            cleanup_paths.append(mask_path)

        if desc_file is not None:
            desc_path = _save_upload(desc_file, workdir, desc_file.filename or "desc.json")
            sample["desc"] = desc_path
            cleanup_paths.append(desc_path)

        result = pipeline.run_full_pipeline_from_heatmaps(
            ct_path=ct_path,
            segmentation_model=ARTIFACTS["segmentation_model"],
            landmark_model=ARTIFACTS["landmark_model"],
            device=ARTIFACTS["device"],
            risk_profile=ARTIFACTS["risk_profile"],
            gt_mask_path=sample.get("mask"),
            show=False,
        )

        out_pdf = os.path.join(REPORT_DIR, f"{subject}_surgical_report.pdf")
        pdf_path = pipeline.generate_case_report(
            sample=sample,
            result=result,
            out_pdf_path=out_pdf,
            dataset_name=pipeline.DATASET_NAME,
            show_in_notebook=bool(show_in_notebook),
        )

        for p in cleanup_paths:
            background_tasks.add_task(_cleanup_path, p)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path),
            background=background_tasks,
        )

    except Exception as e:
        background_tasks.add_task(_cleanup_path, workdir)
        raise HTTPException(status_code=400, detail=f"Report generation failed: {e}")


@app.post("/predict")
async def predict_only(
    background_tasks: BackgroundTasks,
    ct_file: UploadFile = File(...),
    subject: str = Form("uploaded_case"),
):
    if ARTIFACTS["segmentation_model"] is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    workdir = tempfile.mkdtemp(prefix="ct_predict_")
    try:
        ct_path = _save_upload(ct_file, workdir, ct_file.filename or "ct.nii.gz")
        nib.load(ct_path)

        sample = {
            "subject": subject,
            "ct": ct_path,
        }

        result = pipeline.run_full_pipeline_from_heatmaps(
            ct_path=ct_path,
            segmentation_model=ARTIFACTS["segmentation_model"],
            landmark_model=ARTIFACTS["landmark_model"],
            device=ARTIFACTS["device"],
            risk_profile=ARTIFACTS["risk_profile"],
            gt_mask_path=None,
            show=False,
        )

        out_pdf = os.path.join(REPORT_DIR, f"{subject}_prediction_report.pdf")
        pdf_path = pipeline.generate_case_report(
            sample=sample,
            result=result,
            out_pdf_path=out_pdf,
            dataset_name=pipeline.DATASET_NAME,
            show_in_notebook=False,
            include_mask_visualization=False,
        )

        background_tasks.add_task(_cleanup_path, workdir)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path),
            background=background_tasks,
        )
    except Exception as e:
        background_tasks.add_task(_cleanup_path, workdir)
        raise HTTPException(status_code=400, detail=f"Prediction report generation failed: {e}")
