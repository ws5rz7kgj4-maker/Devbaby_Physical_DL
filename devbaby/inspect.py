"""Inspect a saved model on a recorded episode. Labels are only drawn afterwards."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .config import Config
from .data import episode_batch
from .model import Baby

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--episode',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=Config(**ck['config']);model=Baby(cfg).eval();model.load_state_dict(ck['model'])
    episode=Path(args.episode);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    batch=episode_batch(episode/'observations.npz','cpu',steps=cfg.clip_steps)
    with torch.no_grad():r=model(batch)
    np.savez_compressed(out/'internal_states.npz',memory=r['h'].numpy(),predicted_latent=r['z'].numpy(),
                        predicted_body=r['pred_body'].numpy(),actual_body=batch['proprio'][:,1:].numpy())
    labels=[json.loads(line) for line in (episode/'annotations/frames.jsonl').read_text().splitlines()]
    canvas=Image.new('RGB',(576,150*min(8,cfg.clip_steps)+30),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,5),'Current view       Actual next view       Predicted next view',fill='black')
    for i in range(min(8,cfg.clip_steps)):
        arrays=[batch['rgb'][0,i].permute(1,2,0).numpy(),batch['rgb'][0,i+1].permute(1,2,0).numpy(),r['pred_rgb'][0,i].permute(1,2,0).numpy()]
        for j,a in enumerate(arrays):canvas.paste(Image.fromarray(np.uint8(np.clip(a,0,1)*255)).resize((128,128)),(j*192,30+i*150))
        draw.text((5,158+i*150),f"t={labels[i]['time_s']:.2f}s",fill='black')
    canvas.save(out/'prediction_inspection.png')
    (out/'annotations.json').write_text(json.dumps(labels[:cfg.clip_steps+1],indent=2))
    print(out.resolve())

if __name__=='__main__':main()
