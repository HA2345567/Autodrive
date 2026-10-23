const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 for(const key of ['autodrive_workflow','dave2_reference']) {
  let svg=fs.readFileSync(path.join(__dirname,'diagrams',key+'.svg'),'utf8');
  const page=await browser.newPage();
  await page.setContent('<html><head><style>html,body{margin:0;padding:0;background:white}svg{display:block}</style></head><body>'+svg+'</body></html>');
  await page.evaluate(()=>{
   const svg=document.querySelector('svg');
   for(const e of svg.querySelectorAll('rect,polygon,circle,ellipse')) {
    e.style.setProperty('fill','white','important');
    e.style.setProperty('stroke','black','important');
   }
   for(const e of svg.querySelectorAll('text,tspan,.label,.nodeLabel')) {
    e.style.setProperty('fill','black','important');e.style.setProperty('color','black','important');
   }
   for(const e of svg.querySelectorAll('.edgePaths path,.flowchart-link')) e.style.setProperty('stroke','black','important');
   for(const e of svg.querySelectorAll('marker path,marker polygon')) {
    e.style.setProperty('stroke','black','important');e.style.setProperty('fill','black','important');
   }
  });
  await page.evaluate(()=>document.fonts.ready);
  const size=await page.locator('svg').boundingBox();
  await page.pdf({path:path.join(__dirname,'diagrams',key+'_monochrome.pdf'),width:size.width+'px',height:size.height+'px',printBackground:true,margin:{top:0,bottom:0,left:0,right:0}});
  await page.close();
 }
 await browser.close();
 console.log('Rendered two monochrome Mermaid diagrams.');
})().catch(e=>{console.error(e);process.exit(1)});
