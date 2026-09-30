// Run a split-ONNX Laya export (e.g. the q4 WebGPU build) over the holdout with laya-ts on CPU and
// write {id, probs} rows that `eval.external` can score next to the student.
//
// Usage: node tools/laya_onnx_predict.mjs <laya-ts dir> <model dir> <holdout.jsonl> <out.jsonl>
//   laya-ts dir: e.g. ../laya-webgpu/vendor/laya-ts (needs onnxruntime-node resolvable from there)
// Prints a JSON summary (p50/p95 latency, bytes on disk) to stdout; paste it into eval.external.
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [, , layaTs, modelDir, holdoutPath, outPath] = process.argv;
if (!outPath) {
  console.error("usage: node laya_onnx_predict.mjs <laya-ts dir> <model dir> <holdout.jsonl> <out.jsonl>");
  process.exit(2);
}
const log = (...a) => console.error("[laya_onnx_predict]", ...a);
const { Agent } = await import(pathToFileURL(join(resolve(layaTs), "index.js")).href);
const questions = JSON.parse(readFileSync(new URL("../configs/questions.json", import.meta.url), "utf8"));
const order = { department: ["billing", "technical", "account", "sales"], urgency: ["0", "1", "2"] };
const rows = readFileSync(holdoutPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));

const t0 = performance.now();
const agent = await Agent.load(resolve(modelDir), { device: "cpu" });
log(`loaded ${modelDir} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
await agent.predict("warm-up", questions);

const lat = [];
const out = [];
for (const r of rows) {
  const t = performance.now();
  const res = await agent.predict(r.text, questions);
  lat.push(performance.now() - t);
  const a = res.answers;
  out.push({
    id: r.id,
    probs: {
      department: order.department.map((k) => a.department.probabilities[k]),
      urgency: order.urgency.map((k) => a.urgency.probabilities[k]),
      churn_risk: [1 - a.churn_risk.noul, a.churn_risk.noul],
    },
  });
}
writeFileSync(outPath, out.map((o) => JSON.stringify(o)).join("\n") + "\n");
lat.sort((x, y) => x - y);
const q = (p) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))];
const bytes = readdirSync(modelDir).reduce((s, f) => s + statSync(join(modelDir, f)).size, 0);
console.log(JSON.stringify({ rows: rows.length, size_bytes: bytes,
  latency_ms: { p50: q(0.5), p95: q(0.95), runs: lat.length, what: "laya-ts + onnxruntime-node CPU, batch=1" } }));
