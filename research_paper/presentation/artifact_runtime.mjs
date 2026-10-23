// Resolve the authoring SDK without keeping a node_modules junction in Git.
import {createRequire} from 'node:module';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const localRequire=createRequire(import.meta.url);
let entry;
try {
  entry=localRequire.resolve('@oai/artifact-tool');
} catch {
  const modules=process.env.RUNTIME_NODE_MODULES ?? 'C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
  entry=createRequire(path.join(modules,'package.json')).resolve('@oai/artifact-tool');
}
const sdk=await import(pathToFileURL(entry).href);
export const {Presentation, PresentationFile, FileBlob}=sdk;
