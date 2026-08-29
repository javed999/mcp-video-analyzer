#!/usr/bin/env node

import { VERSION } from './version.js';

const USAGE = `mcp-video-analyzer ${VERSION}

Usage:
  mcp-video-analyzer                          Start the MCP server (stdio)
  mcp-video-analyzer analyze <url> [options]  One-shot analysis: JSON on stdout
  mcp-video-analyzer analyze --help           Show analyze options
`;

const cmd = process.argv[2];

if (cmd === 'analyze') {
  const { runCli } = await import('./cli.js');
  const code = await runCli(process.argv.slice(3));
  process.exitCode = code;
  // The one-shot CLI has emitted its complete document and is finished, so it
  // must terminate — even if a dependency left a handle open behind our back.
  // tesseract.js does exactly that when a worker fails to load its
  // `.traineddata`: it abandons the worker thread without ever settling the
  // promise that would hand us the handle to terminate, and the orphaned
  // MessagePort keeps the event loop alive forever. Waiting on that would hang
  // a command whose JSON is already on stdout, so flush and leave regardless.
  await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
  process.exit(code);
} else if (cmd === '--version' || cmd === '-v') {
  process.stdout.write(`${VERSION}\n`);
} else if (cmd === '--help' || cmd === '-h') {
  process.stdout.write(USAGE);
} else if (cmd === undefined) {
  const { createServer } = await import('./server.js');
  createServer().start({ transportType: 'stdio' });
} else {
  process.stderr.write(`Unknown command "${cmd}".\n\n${USAGE}`);
  process.exitCode = 1;
}
