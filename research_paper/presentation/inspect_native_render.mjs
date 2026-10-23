import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from './artifact_runtime.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/Documents/AutoDrive/output/presentation/AutoDrive_Final_Presentation.pptx'));
console.log('Imported',p.slides.items.length);
for(const n of [1,14,17,22]){
 const b=await p.export({slide:p.slides.items[n-1],format:'png',scale:1});
 await fs.writeFile('D:/Documents/AutoDrive/tmp/presentation_rebuild/native-'+n+'.png',new Uint8Array(await b.arrayBuffer()));
 console.log('Rendered',n);
}
