/** ChronosAudit Express API entrypoint. */
import { config } from "./config.js";
import { createApp } from "./app.js";

async function main() {
  const app = await createApp();
  app.listen(config.port, config.host, () => {
    console.log(`[chronos-audit-api] listening on http://${config.host}:${config.port}`);
    console.log(`[chronos-audit-api] worker bridge: ${config.workerBaseUrl}`);
    console.log(`[chronos-audit-api] dashboard:     http://localhost:${config.port}/ui`);
  });
}

main().catch((err) => {
  console.error("[chronos-audit-api] fatal:", err);
  process.exit(1);
});