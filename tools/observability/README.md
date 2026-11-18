# EdgeServe observability (Prometheus + Grafana)

Visualises KV pool occupancy, per-slot state and KV, task flow, throughput/goodput, GPU and
llama-server rates for every run. The exporter only reads the runner's files (`events.jsonl`,
`server_cmd.txt`, `meta.json`); it never queries llama-server, so watching does not perturb a run.

| Piece | Where | What |
|---|---|---|
| `exporter.py live` | 127.0.0.1:9101 | follows the newest `results/**/events.jsonl`, serves its current state |
| Prometheus v3.1.0 | 127.0.0.1:9090 | scrapes the exporter every 1 s; keeps replayed runs |
| Grafana 11.4.0 | 127.0.0.1:3000 | dashboards "EdgeServe live" and "EdgeServe replay" (anonymous admin, localhost only) |
| renderer (optional) | 127.0.0.1:8081 | panel → PNG, e.g. for report figures |

## Start / stop

```bash
cd tools/observability
docker compose up -d                       # Prometheus + Grafana (add --profile render for PNG export)
tmux new -d -s obs "cd ~/mem-aware && uv run python tools/observability/exporter.py live"
docker compose stop                        # stop (data kept in docker volumes)
```

Open Grafana from your laptop through a tunnel: `ssh -L 3000:localhost:3000 mem-aws`, then
http://localhost:3000 (VS Code Remote: forward port 3000 in the PORTS tab).

## Replay finished runs

```bash
uv run python tools/observability/exporter.py replay results/p2a_stage1/l0/rep0 results/p2a_stage1/l2/rep0
```

Every replayed run is placed at 2026-01-01 00:00:00 UTC + seconds since its start, so the replay
dashboard shows all runs on one time axis (00:00 = run start); pick runs with the `run` variable and
each gets its own row. Replaying a run again replaces its earlier data.

Export a panel as PNG (renderer running): `curl -o x.png "localhost:3000/render/d-solo/edgeserve-replay/edgeserve-replay?panelId=11&var-run=<run>&from=1767225600000&to=1767225900000&width=1600&height=500&tz=UTC"`.

## What the panels mean

- **KV pool occupancy**: tokens held, by what the owner is doing: active (prefill/decode, or about to
  send), tool_wait, stale (finished task, KV kept until the slot's next task), shared_prefix (forked
  system prompt, counted once). Dashed line = pool size (`-c`).
- **Slot timeline**: one row per slot: prefill, decode, tool wait, gate wait, stale, dropped, offloaded,
  shared prefix. Replay splits prefill/decode from `t_first_token`; live shows "active".
- Without a KV manager, slot KV is rebuilt from per-request token counts (as in
  `analysis/kv_timeline.py`); llama-server's own purges are not visible. With a manager, its
  `kv_state` events are used directly.
- GPU memory is flat by design (weights + preallocated pool); pool occupancy is what changes.

Dashboards are generated: edit `make_dashboards.py`, then run it (Grafana reloads within ~10 s).
