import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from './artifact_runtime.mjs';
process.env.RUNTIME_NODE_MODULES='C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE='C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe';
const ROOT='D:/Documents/AutoDrive',TMP=ROOT+'/tmp/presentation_rebuild';
const SKILL=process.env.AUTODRIVE_PRESENTATION_SKILL_DIR ?? 'C:/Users/Harsh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const utils=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const proto=JSON.parse(await fs.readFile(TMP+'/presentation.json','utf8'));
const pres=Presentation.load(proto);
console.log('Loaded slides',pres.slides.items.length);
const ref='D:/Telegram Desktop/Minor_synopsis PPT (2).pdf';
const tables=[2,6,8,9,11,20,24,25,26,27,28,29,30,31];
const result=await utils.finalizePresentation({workspaceDir:ROOT,candidatePath:TMP+'/candidate-v2.pptx',finalPath:ROOT+'/output/presentation/AutoDrive_Final_Presentation.pptx',pythonExecutable:'C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:31,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[12,13,16],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:['Calibri','Calibri Light'],referencePath:ref,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(ref)).digest('hex')},verifyArtifactToolImport:true,receiptPath:TMP+'/final_validation.json'});
console.log('Finalized',JSON.stringify(result));
for(let i=0;i<pres.slides.items.length;i++){
 const png=await pres.export({slide:pres.slides.items[i],format:'png',scale:1});
 await fs.writeFile(TMP+'/renders/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 console.log('Rendered',i+1);
}
const pdf=await pres.export({format:'pdf'});
await fs.writeFile(ROOT+'/output/presentation/AutoDrive_Final_Presentation.pdf',new Uint8Array(await pdf.arrayBuffer()));
console.log('PDF saved');
