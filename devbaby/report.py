from pathlib import Path
import csv
import html
import json
import subprocess
import numpy as np

def make_video(episode,destination,hz):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image,ImageDraw
    episode,destination=Path(episode),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    silent=destination.with_name(destination.stem+"_silent.mp4")
    with np.load(episode/"observations.npz",allow_pickle=False) as f: frames=f["rgb"][:-1]
    with imageio.get_writer(silent,fps=hz,codec="libx264",quality=8) as writer:
        for i,frame in enumerate(frames):
            im=Image.fromarray(frame).resize((512,512),Image.Resampling.NEAREST)
            draw=ImageDraw.Draw(im)
            draw.rectangle([0,0,512,25],fill="black")
            draw.text((8,7),f"DevBaby | infant view | t={i/hz:.3f}s",fill="white")
            writer.append_data(np.array(im))
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-i",str(silent),"-i",str(episode/"audio.wav"),
         "-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-shortest",str(destination)]
    subprocess.run(cmd,check=True)
    silent.unlink()

def write_report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root)
    rows=[json.loads(p.read_text()) for p in sorted((root/"metrics").glob("eval_*.json"))]
    if not rows:return
    if rows[0].get("evaluation_mode")=="physics":
        from .physical_report import member_report
        return member_report(root,rows)
    cfg=json.loads((root/"config.json").read_text())
    x=[r["interactions"] for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for cond in ["iid","combination","layout","voice","physics","joint"]:
        axes[0,0].plot(x,[r["grounding"][cond]["accuracy"] for r in rows],".-",label=cond)
    axes[0,0].axhline(.25,color="gray",linestyle="--",label="chance")
    axes[0,0].set(title="Audio-to-object retrieval (4 choices)",ylabel="Accuracy",ylim=(0,1))
    axes[0,0].legend(fontsize=7,ncol=2)
    for key in ["hand_position_mse","hand_copy_mse","shuffled_action_hand_mse"]:
        axes[0,1].plot(x,[r["dynamics"]["iid"][key] for r in rows],".-",label=key)
    axes[0,1].set(title="Next hand-position prediction",ylabel="MSE (m²)")
    axes[0,1].legend(fontsize=7)
    for key in ["learned","random","analytic_reference"]:
        axes[1,0].plot(x,[r["motor"][key]["final_distance_m"] for r in rows],".-",label=key)
    axes[1,0].set(title="Executed 3D reaching (coordinate goals)",ylabel="Final distance (m)")
    axes[1,0].legend(fontsize=7)
    axes[1,1].plot(x,[r["grounding"]["event_diagnostic"]["accuracy"] for r in rows],".-",label="event retrieval")
    axes[1,1].axhline(1/3,color="gray",linestyle="--",label="chance")
    axes[1,1].set(title="Controlled event diagnostic (3 choices)",ylabel="Accuracy",ylim=(0,1))
    axes[1,1].legend(fontsize=7)
    for ax in axes.flat:ax.set_xlabel("Training environment transitions");ax.grid(alpha=.2)
    fig.savefig(root/"learning_curves.png",dpi=160);plt.close(fig)
    fields=["update","interactions","condition","accuracy","n","chance","ci_low","ci_high"]
    with open(root/"learning_curves.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            for cond,v in r["grounding"].items():
                w.writerow(dict(update=r["update"],interactions=r["interactions"],condition=cond,accuracy=v["accuracy"],
                                n=v["n"],chance=v["chance"],ci_low=v["wilson95"][0],ci_high=v["wilson95"][1]))
    last=rows[-1]
    table="".join(f"<tr><td>{html.escape(k)}</td><td>{v['accuracy']:.3f}</td><td>{v['n']}</td><td>{v['chance']:.3f}</td></tr>" for k,v in last["grounding"].items())
    video='<video controls width="512" src="preview.mp4"></video>' if (root/"preview.mp4").exists() else ""
    page=f'''<!doctype html><meta charset="utf-8"><title>DevBaby experiment report</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:35px auto;padding:20px;color:#172434;background:#f6f8fa}}h1{{color:#174e65}}img{{width:100%}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap;background:white;padding:16px}}.note{{background:#fff2ce;padding:18px;border-radius:10px}}</style>
<h1>DevBaby: synthetic developmental learning</h1>
<p>Seed {cfg['seed']} · q={cfg['correspondence']} · {html.escape(cfg['exploration'])} · ablation={html.escape(cfg['ablation'])}</p>
<p>Recorded training interactions: {last['interactions']} · gradient updates: {last['update']}</p>
<div class="note">This report contains measured results, including failures. Audio grounding is separate from control. Reaching receives coordinate goals; it is not verbal instruction following. Ridge probes are supervised diagnostics. No consciousness claim or human developmental age is inferred.</div>
<img src="learning_curves.png" alt="Learning curves">
<h2>Last checkpoint grounding</h2><table><tr><th>Condition</th><th>Accuracy</th><th>Trials</th><th>Chance</th></tr>{table}</table>
<h2>Recorded first-person experience with synthetic audio</h2>{video}
<h2>Raw results</h2><p><a href="learning_curves.csv">CSV</a> · <a href="metrics/eval_{last['update']:08d}.json">Full final metrics</a> · <a href="config.json">Exact configuration</a></p>
<p>Within-run Wilson intervals describe trial uncertainty only. Independent model seeds are required for claims about training reproducibility. Training samples never include the annotation files.</p>
<h2>Executed control diagnostic</h2><pre>{html.escape(json.dumps(last['motor'],indent=2))}</pre>'''
    (root/"report.html").write_text(page,encoding="utf-8")
