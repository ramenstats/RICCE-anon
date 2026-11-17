// Headless check of annotator.html logic (no browser): syntax, seeded order, CSV export on a MOCK session.
// The ratings here are SYNTHETIC mock values used only to test CSV serialization; never real annotations.
const fs=require('fs');
const html=fs.readFileSync(process.argv[2],'utf8');
const m=html.match(/<script>([\s\S]*)<\/script>/);
const code=m[1];
new Function('module',code); // syntax check (throws on error)
const mod={exports:{}};
new Function('module',code)(mod);
const {buildCsv,orderFor,COLS,AXES}=mod.exports;
// STUDY is closure-local, so re-evaluate with access
const src=code.replace('if(typeof module!=="undefined")','globalThis.__T={STUDY,allTasks,orderFor,buildCsv,COLS};if(typeof module!=="undefined")');
new Function('module',src)({exports:{}});
const T=globalThis.__T;
const tasks=T.allTasks();
const o1=T.orderFor('annX',tasks).map(t=>t.id), o2=T.orderFor('annX',tasks).map(t=>t.id), o3=T.orderFor('annY',tasks).map(t=>t.id);
if(JSON.stringify(o1)!==JSON.stringify(o2)) throw new Error('order not deterministic');
if(JSON.stringify(o1)===JSON.stringify(o3)) throw new Error('order identical across annotators');
if(new Set(o1).size!==tasks.length) throw new Error('order not a permutation');
const ord=T.orderFor('annX',tasks);
const st={annotator_id:'annX',r:{}};
const a=ord.find(t=>t.task==='A'),b=ord.find(t=>t.task==='B');
st.r[a.id]={object:'NA',action:3,source_target:'NA',location:5,quantity:'NA',change_nochange:5,specificity:4,completeness:2,hallucination:1,overall:3,notes:'has, comma "and quote"\nnewline',seconds_spent:42,timestamp_utc:'2000-01-01T00:00:00Z'};
st.r[b.id]={preference:'Tie'};
const csv=T.buildCsv(st,ord);
const lines=csv.trim().split('\r\n');
console.log(csv.split('\r\n')[0]);
if(lines[0]!==T.COLS.join(',')) throw new Error('header');
if(!csv.includes('"has, comma ""and quote""\nnewline"')) throw new Error('escaping');
if(csv.split('\r\n').filter(x=>x).length<3) throw new Error('rows');
if(!csv.includes(',B,'+b.id)||!csv.includes(',Tie,')) throw new Error('pair row');
console.log('OK items',T.STUDY.items.length,'pairs',T.STUDY.pairs.length);
