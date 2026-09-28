"""Materialize missing T1 cache entries from the validated DA3 route.

This is an explicit per-entry fallback, never a fake adapter result: existing
Temporal-LiDAR outputs are preserved, while missing entries retain DA3 and are
marked ``adapter_applied=false`` in their report for downstream auditing.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); args=p.parse_args()
    rows=[]
    for source_report in sorted(args.source.glob('*/*/*/report.json')):
        rel=source_report.relative_to(args.source); dest=args.output/rel.parent; dest.mkdir(parents=True,exist_ok=True)
        out_report=dest/'report.json'
        if out_report.exists() and (dest/'fused_depth.npy').exists():
            try:
                meta=json.loads(out_report.read_text())
                if meta.get('temporal_adapter_checkpoint') == str(args.checkpoint):
                    rows.append({'path':str(rel.parent),'adapter_applied':meta.get('adapter_applied',True)}); continue
            except (OSError,json.JSONDecodeError): pass
        for item in source_report.parent.iterdir():
            target=dest/item.name
            if target.exists(): target.unlink()
            if item.is_file() and item.name != 'report.json': shutil.copy2(item,target)
        meta=json.loads(source_report.read_text())
        meta['temporal_adapter_checkpoint']=str(args.checkpoint)
        meta['temporal_adapter_contract']='DA3 fallback; no Temporal-LiDAR adapter applied for this entry'
        meta['adapter_applied']=False
        out_report.write_text(json.dumps(meta,indent=2))
        rows.append({'path':str(rel.parent),'adapter_applied':False})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'adapter_cache_manifest.json').write_text(json.dumps({'checkpoint':str(args.checkpoint),'rows':rows,'fallback_entries':sum(not r['adapter_applied'] for r in rows)},indent=2))
    print(json.dumps({'entries':len(rows),'fallback_entries':sum(not r['adapter_applied'] for r in rows),'output':str(args.output)},indent=2))
if __name__=='__main__': main()
