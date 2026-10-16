export function summarize(rows,threshold=.5){
  if(!Number.isFinite(threshold)||threshold<0||threshold>1)throw new Error('Threshold must be between zero and one.');
  if(!Array.isArray(rows)||rows.some(p=>![0,1].includes(p.actual)||!Number.isFinite(p.probability)||p.probability<0||p.probability>1))throw new Error('Invalid held-out prediction records.');
  let tp=0,tn=0,fp=0,fn=0;for(const p of rows){const predicted=p.probability>=threshold;if(predicted&&p.actual)tp++;else if(predicted)fp++;else if(p.actual)fn++;else tn++;}
  const precision=tp+fp?tp/(tp+fp):null,recall=tp+fn?tp/(tp+fn):null;
  const positives=rows.filter(p=>p.actual===1),negatives=rows.filter(p=>p.actual===0);let rank=0;
  for(const p of positives)for(const n of negatives)rank+=p.probability>n.probability?1:p.probability===n.probability?.5:0;
  const auc=positives.length&&negatives.length?rank/(positives.length*negatives.length):null;
  const cutoffs=[Infinity,...new Set(rows.map(p=>p.probability).sort((a,b)=>b-a)),-Infinity];
  const roc=cutoffs.map(t=>({fpr:negatives.length?negatives.filter(p=>p.probability>=t).length/negatives.length:0,tpr:positives.length?positives.filter(p=>p.probability>=t).length/positives.length:0}));
  return {n:rows.length,tp,tn,fp,fn,accuracy:rows.length?(tp+tn)/rows.length:null,precision,recall,f1:2*tp+fp+fn?2*tp/(2*tp+fp+fn):null,auc,brier:rows.length?rows.reduce((s,p)=>s+(p.probability-p.actual)**2,0)/rows.length:null,roc};
}
export function filterRows(rows,{sex='all',pclass='all'}={}){return rows.filter(p=>(sex==='all'||p.sex===sex)&&(pclass==='all'||p.class===Number(pclass)));}
export function csv(rows,threshold){summarize(rows,threshold);return ['passenger_id,actual,probability,threshold,predicted,correct',...rows.map(p=>`${p.id},${p.actual},${p.probability},${threshold},${Number(p.probability>=threshold)},${Number((p.probability>=threshold)===Boolean(p.actual))}`)].join('\n');}
