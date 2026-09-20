import os
import json
import argparse
import sys
import tempfile
from pathlib import Path
import numpy as np

DATASETS = [("asap", 32), ("batik", 30), ("vienna", 84)]
TOTAL_PIECES = 146

def evaluate_native(output_dir, methods):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from matchmaker_eval.utils import compute_event_pooled_summary

    root = Path(output_dir)
    data = {}
    for method in methods:
        pieces = {}
        for dataset, count in DATASETS:
            run_dir = root / method / dataset
            completion = json.loads((run_dir / "completion.json").read_text())
            if not completion["complete"] or completion["completed"] != count:
                raise ValueError(f"Incomplete run: {run_dir}")
            for index in range(1, count + 1):
                pieces[dataset, index] = json.loads((run_dir / f"{index}.json").read_text())
        data[method] = pieces
    common = set.intersection(*(
        {key for key, result in pieces.items() if result["tracked"]}
        for pieces in data.values()
    ))
    report = {"common_tracked_count": len(common), "common_tracked": sorted(common), "methods": {}}
    for method, pieces in data.items():
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp)
            results = {"Index": [], "tracked": [], "rtf": []}
            for pooled_index, ((dataset, index), result) in enumerate(pieces.items(), 1):
                for kind in ("wp", "gt"):
                    source = root / method / dataset / f"{kind}_{index}.tsv"
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    (staged / f"{kind}_{pooled_index}.tsv").symlink_to(source.resolve())
                results["Index"].append(pooled_index)
                results["tracked"].append(result["tracked"])
                results["rtf"].append(result["rtf"])
            own = compute_event_pooled_summary(results, staged, tracked_only=True)
            common_results = dict(results, tracked=[key in common for key in pieces])
            shared = compute_event_pooled_summary(common_results, staged, tracked_only=True)
        tracked = [key for key, result in pieces.items() if result["tracked"]]
        smoothness = [compute_piece_sparc_30s(root / method / ds / f"wp_{i}.tsv") for ds, i in tracked]
        if any(value is None for value in smoothness):
            raise ValueError(f"Missing tracked SPARC: {method}")
        report["methods"][method] = {
            "datasets": {ds: sum(pieces[ds, i]["tracked"] for i in range(1, count + 1)) for ds, count in DATASETS},
            "all_tracked": own,
            "common_tracked": shared,
            "sparc": float(np.mean(smoothness)) if smoothness else None,
            "rtf_all_pieces": float(np.mean(results["rtf"])),
        }
    (root / "paper_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

def compute_sparc(vel, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral Arc Length (SPARC) smoothness metric."""
    if len(vel) < 2 or np.all(vel == vel[0]):
        return -100.0
    n = len(vel)
    n_fft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    vel_centered = vel - np.mean(vel)
    spec = np.abs(np.fft.rfft(vel_centered, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    
    mask = freqs <= fc
    freqs = freqs[mask]
    spec = spec[mask]
    
    if len(spec) < 2:
        return -100.0
        
    max_amp = np.max(spec)
    if max_amp == 0:
        return -100.0
    spec_norm = spec / max_amp
    
    valid = spec_norm >= amp_th
    if not np.any(valid):
        return -100.0
    last_idx = np.where(valid)[0][-1]
    if last_idx < 1:
        return -100.0
        
    freqs_sub = freqs[:last_idx + 1]
    spec_sub = spec_norm[:last_idx + 1]
    
    df = np.diff(freqs_sub) / fc
    dmag = np.diff(spec_sub)
    arc_length = -np.sum(np.sqrt(df**2 + dmag**2))
    return float(arc_length)

def compute_piece_sparc_30s(wp_path, window_sec=30.0, hop_sec=15.0, fps=50.0):
    """Compute average SPARC over sliding 30-second windows along the trajectory."""
    if not os.path.exists(wp_path):
        return None
    try:
        wp = np.loadtxt(wp_path, skiprows=1)
        if wp.ndim != 2 or wp.shape[0] < int(window_sec * fps * 0.5):
            return None
        t = wp[:, 0]
        pos = wp[:, 1]
        
        t_min, t_max = t[0], t[-1]
        if t_max - t_min < window_sec:
            vel = np.gradient(pos, t)
            return compute_sparc(vel, fps)
            
        t_uniform = np.arange(t_min, t_max, 1.0 / fps)
        pos_uniform = np.interp(t_uniform, t, pos)
        vel_uniform = np.gradient(pos_uniform, 1.0 / fps)
        
        win_len = int(window_sec * fps)
        hop_len = int(hop_sec * fps)
        
        sparcs = []
        for start in range(0, len(vel_uniform) - win_len + 1, hop_len):
            chunk = vel_uniform[start:start + win_len]
            sp = compute_sparc(chunk, fps)
            if np.isfinite(sp) and sp > -500:
                sparcs.append(sp)
        return float(np.mean(sparcs)) if sparcs else None
    except Exception:
        return None

def load_benchmark_data(output_dir, methods):
    output_dir = Path(output_dir)
    pieces_dir = output_dir / "pieces"
    data = {m: {} for m in methods}
    
    for ds, count in DATASETS:
        for i in range(1, count + 1):
            key = f"{ds}/{i:03d}"
            for m in methods:
                res_path = pieces_dir / key / m / "result.json"
                wp_path = pieces_dir / key / m / "wp.tsv"
                if res_path.exists():
                    try:
                        res = json.loads(res_path.read_text())
                        res["wp_path"] = str(wp_path)
                        data[m][key] = res
                    except Exception:
                        pass
    return data

def evaluate(output_dir, methods=None):
    if methods is None:
        methods = ["softoltw", "softoltw_no_imm"]
    
    data = load_benchmark_data(output_dir, methods)
    
    print("=" * 140)
    print(f"BENCHMARK REPORT: {output_dir}")
    print("=" * 140)
    
    for m in methods:
        m_data = data[m]
        done = len(m_data)
        tracked_keys = [k for k, r in m_data.items() if r.get("tracked")]
        n_tr = len(tracked_keys)
        
        ds_stats = {}
        for ds, count in DATASETS:
            ds_keys = [f"{ds}/{i:03d}" for i in range(1, count + 1)]
            ds_done = [k for k in ds_keys if k in m_data]
            ds_tr = [k for k in ds_done if m_data[k].get("tracked")]
            ds_stats[ds] = (len(ds_tr), len(ds_done), count)
            
        means = [m_data[k]["beat"]["mean"] for k in tracked_keys if "beat" in m_data[k]]
        meds = [m_data[k]["beat"]["median"] for k in tracked_keys if "beat" in m_data[k]]
        le1 = [m_data[k]["beat"]["le_1"] for k in tracked_keys if "beat" in m_data[k]]
        rtfs = [v.get("inference_rtf", v.get("replay_rtf", 0.0)) for v in m_data.values()]
        
        mean_ae = np.mean(means) if means else 0.0
        med_ae = np.mean(meds) if meds else 0.0
        pct_le1 = np.mean(le1) * 100.0 if le1 else 0.0
        avg_rtf = np.mean(rtfs) if rtfs else 0.0
        
        asap_tr, _, _ = ds_stats["asap"]
        batik_tr, _, _ = ds_stats["batik"]
        vienna_tr, _, _ = ds_stats["vienna"]
        
        sparcs = []
        for k in tracked_keys:
            wp_p = m_data[k].get("wp_path")
            if wp_p and os.path.exists(wp_p):
                sp = compute_piece_sparc_30s(wp_p)
                if sp is not None:
                    sparcs.append(sp)
        avg_sparc = np.mean(sparcs) if sparcs else 0.0
        
        print(f"Method: {m:<20} | Done: {done:3d}/{TOTAL_PIECES} | TR: {n_tr:3d}/{TOTAL_PIECES} ({n_tr/TOTAL_PIECES*100:5.1f}%)")
        print(f"  Breakdown: ASAP {asap_tr:2d}/32 ({asap_tr/32*100:5.1f}%), Batik {batik_tr:2d}/30 ({batik_tr/30*100:5.1f}%), Vienna {vienna_tr:2d}/84 ({vienna_tr/84*100:5.1f}%)")
        print(f"  Tracked Errors: MeanAE = {mean_ae:.4f}b, MedAE = {med_ae:.4f}b, <=1.0b = {pct_le1:.1f}%, SPARC(30s) = {avg_sparc:.4f}, RTF = {avg_rtf:.4f}")
        print("-" * 140)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--methods", nargs="+", default=["softoltw", "softoltw_no_imm"])
    parser.add_argument("--native", action="store_true", help="Read test_audio.py output in METHOD/DATASET directories and pool events")
    args = parser.parse_args()
    if args.native:
        evaluate_native(args.output, args.methods)
    else:
        evaluate(args.output, args.methods)
