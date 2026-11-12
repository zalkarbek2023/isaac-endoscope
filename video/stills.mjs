// node stills.mjs <CompositionId> <frame> [...]: render stills of any composition (bundle once)
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import path from 'node:path';
const [id, ...frames] = process.argv.slice(2);
const serveUrl = await bundle({entryPoint: path.resolve('src/index.ts')});
const composition = await selectComposition({serveUrl, id, chromiumOptions: {gl: 'angle'}});
for (const f of frames.map(Number)) {
  await renderStill({serveUrl, composition, frame: f, output: `out/${id}_${f}.png`, chromiumOptions: {gl: 'angle'}});
  console.log('still', id, f);
}
