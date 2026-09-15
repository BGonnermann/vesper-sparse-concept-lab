"""Verify finished checkpoint-off baselines and write the bounded experiment report."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf-8")
probes={v:read(HERE/(v+"-probe-result.json")) for v in ("dense","moe")}
records={}
for v in probes:
    assert probes[v]["gate_passed"] and probes[v]["comfortable_memory"] and probes[v]["correctness_passed"]
    for kind in ("smoke","baseline"):
        p=ROOT/"runs/autoresearch"/f"checkpoint-20260915T010035Z-{v}-{kind}"
        r=read(p/"result.json")
        assert r["status"]=="completed" and r["returncode"]==0
        assert r["protocol"]["activation_checkpointing"] is False
        assert r["artifacts"]["model"]["activation_checkpointing"] is False
        assert r["artifacts"]["training"]["activation_checkpointing"] is False
        assert r["execution"]["verified"] and r["execution"]["isolated"]
        for rel,sha in r["snapshot_files"].items():assert digest(p/rel)==sha
        records[v,kind]=r
fields=("protocol","upstream","data_seal","seed","runner_sha256","adapter_sha256",
        "model_sha256","bootstrap_sha256","record_version","orchestrator_sha256","checkpoint_probe_hashes")
for key in fields:
    a=records["dense","baseline"][key]
    assert all(r[key]==a for r in records.values()),key
for rel,sha in records["dense","baseline"]["data_seal"].items():
    assert digest(ROOT/".autoresearch/cache"/rel)==sha
old=read(HERE/"prior-protocol.json")
assert {k:v for k,v in records["dense","baseline"]["protocol"].items() if k!="activation_checkpointing"}==old
training={}
for v in probes:
    r=records[v,"baseline"]
    folder=ROOT/"runs/autoresearch"/f"checkpoint-20260915T010035Z-{v}-baseline"
    t=r["artifacts"]["training"]; routing=r["artifacts"]["routing"]; model=r["artifacts"]["model"]
    assert r["candidate"]==records[v,"smoke"]["candidate"]
    assert r["metrics"]["train_batch_size"]==r["protocol"]["microbatch_size"]==2
    assert t["training_tokens"]==t["optimizer_updates"]*16384
    assert t["timed_training_tokens"]==(t["optimizer_updates"]-11)*16384
    assert t["timed_training_seconds"]>=300 and t["warmup_updates"]==11
    assert routing["train"]["dropped_tokens"]==routing["eval"]["dropped_tokens"]==0
    assert routing["router_gradients_finite"]
    if v=="moe":assert all(x>0 for x in routing["router_gradient_max_abs"])
    memory=read(folder/"memory.json")
    training[v]={"validation_bpb":r["metrics"]["val_bpb"],
                 "timed_tokens_per_second":t["timed_training_tokens"]/t["timed_training_seconds"],
                 "training_tokens":t["training_tokens"],"timed_tokens":t["timed_training_tokens"],
                 "updates":t["optimizer_updates"],"timed_seconds":t["timed_training_seconds"],
                 "wall_seconds":r["wall_seconds"],"model":model,"routing":routing,
                 "memory":memory,"allocated_mib":memory["peak_allocated_bytes"]/2**20,
                 "reserved_mib":memory["peak_reserved_bytes"]/2**20,"path":str(folder.relative_to(ROOT))}
prior=read(HERE/"prior-preservation.json")
for e in prior:assert digest(Path(e["Path"])).upper()==e["Hash"].upper(),e["Path"]
manifest=read(HERE/"probe-manifest.json")
for rel,sha in manifest["files"].items():assert digest(HERE/rel)==sha
result={"status":"complete","cpu_tests_passed":27,"cuda_tests_passed":27,
        "preserved_prior_files":len(prior),"matched_fields":fields,"probes":probes,"training":training,
        "review":"No concrete correctness/configuration findings in independent read-only review",
        "limitations":["Three short paired measurements, one saved checkpoint per model",
                       "No profiler in throughput; standalone memory passes are separate",
                       "Replay omits CPU packing; actual baselines include it",
                       "Background GPU allocations/clocks were not controlled",
                       "One seed and unequal dense/MoE parameter budgets; quality comparison preliminary",
                       "No fresh checkpoint-on training controls; previous on baselines are historical"],
        "stopped_after":"Two off smoke gates and exactly one 300-second checkpoint-off baseline per model"}
write(HERE/"comparison.json",result)
probe_rows=[]
for v,p in probes.items():
    for mode in ("on","off"):
        a=p["modes"][mode];m=p["memory"][mode]
        probe_rows.append(f"| {v} | {mode} | {a['tokens_per_second']:,.1f} | {m['peak_allocated_bytes']/2**20:.2f} | {m['peak_reserved_bytes']/2**20:.2f} |")
training_rows=[]
d,m=training["dense"],training["moe"]
for key,label,fmt in [
    ("validation_bpb","Validation BPB",".6f"),("timed_tokens_per_second","Timed tokens/s",",.1f"),
    ("training_tokens","Total tokens including warmup",","),("timed_tokens","Timed tokens excluding warmup",","),
    ("updates","Optimizer updates",","),("timed_seconds","Timed seconds",".3f"),
    ("wall_seconds","Launch-to-record wall seconds",".3f"),
    ("allocated_mib","Peak allocated MiB",".2f"),("reserved_mib","Peak reserved MiB",".2f")]:
    training_rows.append(f"| {label} | {format(d[key],fmt)} | {format(m[key],fmt)} |")
util=[]
for layer,(tr,ev) in enumerate(zip(m["routing"]["train"]["fractions"],m["routing"]["eval"]["fractions"])):
    util.append(f"| {layer} | "+" / ".join(f"{100*x:.2f}%" for x in tr)+" | "+" / ".join(f"{100*x:.2f}%" for x in ev)+" |")
trade=[]
for v,p in probes.items():
    on,off=p["memory"]["on"],p["memory"]["off"]
    gains=[100*x["off_throughput_gain"] for x in p["pairs"]]
    trade.append(f"- **{v}: +{100*p['off_throughput_gain']:.2f}% throughput**, all three pairs improved ({min(gains):.2f}%-{max(gains):.2f}%). Extra allocated memory: {(off['peak_allocated_bytes']-on['peak_allocated_bytes'])/2**20:.2f} MiB; extra reserved: {(off['peak_reserved_bytes']-on['peak_reserved_bytes'])/2**20:.2f} MiB. Minimum free VRAM sampled during the off memory pass: {off['minimum_free_bytes_at_update_boundaries']/2**30:.2f} GiB.")
report=f"""# Activation checkpointing: controlled speed/memory experiment

## Outcome

Checkpointing off passed correctness, comfortable-memory and consistent-throughput gates for both dense and packed MoE. Exactly one fresh 300-second checkpoint-off baseline per model completed after fresh off smoke gates.

{chr(10).join(trade)}

These are controlled **replay speed** results, not learning-quality claims. The fresh training comparison below is preliminary: one seed, unequal dense/MoE parameter budgets, and no fresh checkpoint-on baseline controls.

## Configuration and correctness

The captured protocol now requires a strict boolean @@activation_checkpointing@@. Default @@experiments/autoresearch/protocol.json@@ remains true. @@protocol-no-checkpoint.json@@ is false; every other protocol field is unchanged. The runner accepts @@--protocol@@, includes the complete protocol in smoke fingerprints/snapshots, and checks the actual model/training checkpoint flag against it. Old artifacts are untouched.

Both implementations remain depth 6/width 384, context 512, microbatch **2**, 16,384 tokens/update with 16 accumulation microbatches, BF16 expert/model execution and FP32 MoE router math. Attention, optimizer, routing/auxiliary loss, tokenizer/data and evaluation budgets are unchanged. No batch-size tuning.

**27 CPU and 27 CUDA tests passed.** On/off tests compare logits, loss, all parameter gradients and routing counts, including nonzero expert projections. Absolute tolerance 1e-6, relative 1e-5; no tolerance loosening. Existing tests cover causal behavior, empty experts, router task gradients, optimizer coverage and snapshot isolation.

Full depth-6/context-512 saved-checkpoint comparisons also passed. Maximum gradient absolute differences: dense **{probes['dense']['max_gradient_error']}**, MoE **{probes['moe']['max_gradient_error']}**. Checkpoint tensors were unchanged after profiling. Independent review found no concrete configuration/correctness issue.

The first log could not be opened for writing during two logging attempts. Those attempts were not accepted as test evidence. Fresh log writers were held open before announcements, then a start marker released each child. The subsequent expected test-first configuration failure and all passing results are preserved.

## Paired uninstrumented profiling

Same saved weights and replay batches within each model's on/off comparison. Three warmup updates per condition, then exactly **three paired measurements of two updates/condition**: 98,304 timed tokens per condition. Order on/off, off/on, on/off. No profiler or extra measurement hooks in timed updates; learning rates zero, full optimizer work retained.

| Model | Checkpointing | Replay tokens/s | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|
{chr(10).join(probe_rows)}

Memory came from separate standalone passes: one resident model/optimizer, unused allocator cache released before each condition, three warmup and two memory updates. Cache clearing and memory instrumentation were excluded from reported throughput. Peak statistics cover each whole memory pass. Reserved memory therefore does not simply carry over from off into on.

The gate required all three paired off results faster, aggregate gain >=5%, off reserved memory <75% of physical VRAM, and free VRAM >=25% at update boundaries. Both passed with wide headroom. CUDA allocator numbers exclude other processes/driver memory. Background allocations changed during these runs, so free-memory differences are not attributed solely to checkpointing.

Replay includes transfer, forward, backward, optimizer and gradient clearing, but omits CPU tokenization/packing. Raw paired samples, frozen-weight checks, identical routing counts, minimum free-memory readings and captured protocols are in each model's probe result.

## Fresh checkpoint-off training

Both fresh smoke gates passed. Actual captured model/training receipts confirm checkpointing **false**. Protocol, seed 42, data/tokenizer hashes, runtime and executed source identities match across dense/MoE runs. Only their established architecture-specific candidate fields differ. Other settings match the prior protocol exactly.

| Metric | Dense off | Packed MoE off |
|---|---:|---:|
{chr(10).join(training_rows)}

Dense parameters: total/active **{d['model']['total_parameters']:,}/{d['model']['active_parameters']:,}**. MoE parameters: **{m['model']['total_parameters']:,}/{m['model']['active_parameters']:,}**. No memory tables.

Timed throughput excludes the first 11 warmup updates; total tokens include them. The protocol targets 300 seconds and stops at an update boundary, so actual timed seconds can slightly exceed 300. Wall seconds exclude log-announcement waiting and snapshot preparation. BPB contains cross-entropy only, not routing auxiliary loss.

### Expert utilization

Share of tokens routed to experts 0 / 1 / 2 / 3:

| Layer | Training | Validation |
|---|---|---|
{chr(10).join(util)}

Zero dropped tokens in training/validation; all routers have finite nonzero gradients. Dense has no experts. MoE mean training auxiliary loss **{m['routing']['training_auxiliary_loss']:.6f}**, weighted contribution **{m['routing']['weighted_training_auxiliary_loss']:.6f}**, recorded separately from validation BPB.

## Interpretation and limits

Turning checkpointing off retains activations instead of recomputing blocks during backward. Here the extra memory was modest relative to GPU capacity and saved substantial repeated attention/MLP/dispatch work. Packed MoE still pays for four smaller expert GEMMs, routing/dispatch and optimizer work across all expert matrices. Similar active parameter counts do not imply equal training cost.

The paired measurements support a speed/memory tradeoff on this GPU at this exact batch/context. Three pairs, uncontrolled clocks/background load, fixed saved-weight replay and one fresh training seed do not establish general performance or a statistically robust quality gain. Previous checkpoint-on baseline BPBs are historical, not contemporaneous controls. The training table must not be conflated with the frozen-weight profiling table.

## Preservation and stop

**{len(prior)} prior files verified unchanged**, including all earlier depth, MoE, packed-dispatch and profiling artifacts. No batch-size tuning, new architecture features or cloud jobs. Stopped after this report.

Artifacts in this directory: @@comparison.json@@, @@probe-manifest.json@@, @@dense-probe-result.json@@, @@moe-probe-result.json@@, paired sample logs, captured @@source/@@, on/off protocols, tests and preservation receipts. Reproducible orchestration: @@checkpoint_probe.py@@, @@arm_probe.py@@, @@gated_trial.py@@, @@final_report.py@@.

Fresh training logs/results: @@{d['path']}@@ and @@{m['path']}@@. Both retain captured source/configuration identities, memory/routing reports and weights-only checkpoints. Every child run's log and exact PowerShell tail command were made available before it started.
"""
(HERE/"report.md").write_text(report.replace("@@",chr(96)),encoding="utf-8")
write(HERE/"completion.json",{"status":"complete","preserved_prior_files":len(prior),
                            "baselines":2,"cloud_jobs":0,"report_sha256":digest(HERE/"report.md")})
print(json.dumps({v:{k:x for k,x in t.items() if k not in ("model","routing","memory")} for v,t in training.items()},indent=2))
print("Profile summary:",[{"variant":v,"gain":p["off_throughput_gain"],"on_tps":p["modes"]["on"]["tokens_per_second"],"off_tps":p["modes"]["off"]["tokens_per_second"],"max_gradient_error":p["max_gradient_error"]} for v,p in probes.items()])
print("Training utilization range:",min(x for row in m["routing"]["train"]["fractions"] for x in row),max(x for row in m["routing"]["train"]["fractions"] for x in row))
print("PASS: completed; prior files unchanged:",len(prior))
