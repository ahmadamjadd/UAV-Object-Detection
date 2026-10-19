// Check the pinned toolchain with a disposable fixture, without starting a server.
import { mkdtempSync, writeFileSync, symlinkSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dependencies = join(root, "frontend", "node_modules");
const fixture = mkdtempSync(join(tmpdir(), "foxtrot-frontend-"));

function run(tool, args) {
  const result = spawnSync(process.execPath, [join(dependencies, tool), ...args], {
    cwd: fixture,
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${tool} failed (${result.status})`);
}

try {
  symlinkSync(dependencies, join(fixture, "node_modules"), "dir");
  writeFileSync(join(fixture, "package.json"), JSON.stringify({ type: "module" }));
  writeFileSync(join(fixture, "index.html"),
    '<div id="root"></div><script type="module" src="/main.tsx"></script>');
  writeFileSync(join(fixture, "main.tsx"), `
import { createRoot } from "react-dom/client";
createRoot(document.getElementById("root")!).render(<p>H1 toolchain fixture</p>);
`);
  writeFileSync(join(fixture, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022", module: "ESNext", moduleResolution: "Bundler",
      jsx: "react-jsx", strict: true, noEmit: true,
      lib: ["ES2022", "DOM"], skipLibCheck: true,
    },
    include: ["main.tsx"],
  }));
  run("typescript/bin/tsc", ["--project", join(fixture, "tsconfig.json")]);
  run("vite/bin/vite.js", ["build", fixture]);
  if (!existsSync(join(fixture, "dist", "index.html"))) {
    throw new Error("Vite did not produce the fixture entry page");
  }
  console.log("React/TypeScript/Vite temporary build: PASS");
} finally {
  // Only this invocation's owned temporary fixture is removed.
  rmSync(fixture, { recursive: true, force: true });
}
