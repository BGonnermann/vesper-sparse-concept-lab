# ngram-diagnostics-20260915/DG

Outcome: **failed**. Budget family: **fixed_updates**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
run.log:
Traceback (most recent call last):
    raise RuntimeError(f'Enabled BPB reproduction failed: {delta} > {meta["bpb_absolute_tolerance"]}')
RuntimeError: Enabled BPB reproduction failed: 0.0020793106738951073 > 2e-06
```
