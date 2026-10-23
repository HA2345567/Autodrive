import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {PresentationFile, FileBlob} from './artifact_runtime.mjs';

const root='D:/Documents/AutoDrive';
const build=root+'/tmp/presentation_video';
const skill='C:/Users/Harsh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
process.env.RUNTIME_NODE_MODULES='C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE='C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe';
const source=root+'/output/presentation/AutoDrive_Final_Presentation_Ready.pptx';
const candidate=build+'/candidate-preserved-with-video.pptx';
const final=root+'/output/presentation/AutoDrive_Final_Presentation_With_Smooth_Video.pptx';
const tables=[2,6,8,9,11,20,24,25,26,27,28,29,30,31];
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const result=await finalizePresentation({
  workspaceDir:root,candidatePath:candidate,finalPath:final,
  pythonExecutable:'C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
  layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:31,
  requiredNativeTableOwnerSlides:tables,
  requiredNativeChartOwnerSlides:[12,13,16],
  requiredEmbeddedWorkbookChartOwnerSlides:[12,13,16],
  materializeLiteralChartWorkbooks:false,
  fontPolicy:{basis:'reference',families:['Calibri','Calibri Light'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
  verifyArtifactToolImport:true,
  receiptPath:build+'/final-video.validation.json',
});
console.log('Finalized',final);
console.log('Layout warnings',JSON.stringify(result.presentationLayout?.warnings??[]));
const p=await PresentationFile.importPptx(await FileBlob.load(final));
const evidence=await p.inspect({kind:'slide,shape,image,textbox,notes',search:'AutoDrive demonstration',maxChars:4000});
await fs.writeFile(build+'/changed-slide.inspect.ndjson',evidence.ndjson);
console.log('Imported slides',p.slides.items.length);
