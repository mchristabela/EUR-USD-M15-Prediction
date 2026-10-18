const PIP=1e-4;
function parseTime(s){const m=String(s).match(/(\d{4})[-./](\d{1,2})[-./](\d{1,2})[ T]+(\d{1,2}):(\d{2})/);return m?Date.UTC(+m[1],m[2]-1,+m[3],+m[4],+m[5]):NaN}
function parseCSV(text){
  const lines=text.replace(/^\uFEFF/,"").split(/\r?\n/).map(l=>l.trim()).filter(Boolean),notes=[];
  if(!lines.length)return{bars:[],notes};
  const d=lines[0].includes("\t")?"\t":lines[0].includes(";")?";":lines[0].includes(",")?",":/\s+/;
  const rows=lines.map(l=>l.split(d).map(x=>x.trim()));
  let start=0;if(isNaN(parseFloat(rows[0][1]))&&isNaN(parseFloat(rows[0][2])))start=1;
  const bars=[];
  for(let r=start;r<rows.length;r++){
    const a=rows[r];let k=0,t;
    if(/^\d{1,2}:\d{2}/.test(a[1]||"")){t=parseTime(a[0]+" "+a[1]);k=2}else{t=parseTime(a[0]);k=1}
    const v=a.slice(k).map(x=>parseFloat(x.replace(",",".")));
    if(isNaN(t)||v.length<4||v.slice(0,4).some(isNaN))continue;
    const ex=v.slice(4).filter(x=>!isNaN(x));
    bars.push({t,o:v[0],h:v[1],l:v[2],c:v[3],v:ex.length?ex[0]:0,s:ex.length>1?ex[ex.length-1]:NaN});
  }
  bars.sort((x,y)=>x.t-y.t);
  const u=bars.filter((b,i)=>i===0||b.t!==bars[i-1].t);
  if(u.length&&u.every(b=>isNaN(b.s))){u.forEach(b=>b.s=5);notes.push("Spread column not found; assuming 5 points (0.5 pip).")}
  else u.forEach(b=>{if(isNaN(b.s))b.s=5});
  return{bars:u,notes};
}
const roll=(a,w,fn)=>{const o=new Float64Array(a.length).fill(NaN);for(let i=w-1;i<a.length;i++){let ok=true;for(let j=i-w+1;j<=i;j++)if(isNaN(a[j])){ok=false;break}if(ok)o[i]=fn(a,i-w+1,i+1)}return o};
const mean=(a,s,e)=>{let t=0;for(let i=s;i<e;i++)t+=a[i];return t/(e-s)};
const sd=(a,s,e)=>{const m=mean(a,s,e);let t=0;for(let i=s;i<e;i++)t+=(a[i]-m)**2;return Math.sqrt(t/(e-s-1))};
function computeFeatures(B){
  const n=B.length,c=B.map(b=>b.c),o=B.map(b=>b.o),h=B.map(b=>b.h),l=B.map(b=>b.l),vol=B.map(b=>b.v),sp=B.map(b=>b.s);
  const F={},NaNs=()=>new Float64Array(n).fill(NaN);
  const diff=k=>c.map((x,i)=>i>=k?x-c[i-k]:NaN);
  for(const k of[1,2,3,5,10,20,48,96])F["ret"+k]=diff(k).map(x=>x/PIP);
  const rng=h.map((x,i)=>x-l[i]);
  F.body=c.map((x,i)=>(x-o[i])/PIP);F.range=rng.map(x=>x/PIP);
  F.upper_wick=h.map((x,i)=>(x-Math.max(o[i],c[i]))/PIP);F.lower_wick=l.map((x,i)=>(Math.min(o[i],c[i])-x)/PIP);
  F.close_pos=c.map((x,i)=>rng[i]===0?.5:(x-l[i])/rng[i]);
  F.vol20=roll(F.ret1,20,sd);F.vol96=roll(F.ret1,96,sd);F.vol_ratio=F.vol20.map((x,i)=>x/F.vol96[i]);
  const tr=c.map((x,i)=>i?Math.max(rng[i],Math.abs(h[i]-c[i-1]),Math.abs(l[i]-c[i-1]))/PIP:rng[i]/PIP);
  F.atr14=roll(tr,14,mean);
  const s20=roll(c,20,mean),s96=roll(c,96,mean);
  F.dev_sma20_z=c.map((x,i)=>(x-s20[i])/PIP/F.vol96[i]);F.dev_sma96_z=c.map((x,i)=>(x-s96[i])/PIP/F.vol96[i]);
  F.sma20_slope=s20.map((x,i)=>i>=5?(x-s20[i-5])/PIP:NaN);
  const dl=diff(1),up=roll(dl.map(x=>x>0?x:0).map((x,i)=>isNaN(dl[i])?NaN:x),14,mean),dn=roll(dl.map((x,i)=>isNaN(dl[i])?NaN:(x<0?-x:0)),14,mean);
  F.rsi14=up.map((x,i)=>dn[i]===0?NaN:100-100/(1+x/dn[i]));
  F.logvol=vol.map(Math.log1p);
  const vm=roll(vol,96,mean),vs=roll(vol,96,sd),sm=roll(sp,96,mean),ss=roll(sp,96,sd);
  F.volume_z96=vol.map((x,i)=>(x-vm[i])/vs[i]);F.spread_pips=sp.map(x=>x/10);F.spread_z96=sp.map((x,i)=>(x-sm[i])/ss[i]);
  const dt=B.map(b=>new Date(b.t)),hr=dt.map(x=>x.getUTCHours()),hh=dt.map((x,i)=>hr[i]+x.getUTCMinutes()/60);
  F.hour_sin=hh.map(x=>Math.sin(2*Math.PI*x/24));F.hour_cos=hh.map(x=>Math.cos(2*Math.PI*x/24));
  F.dow=dt.map(x=>(x.getUTCDay()+6)%7);
  F.sess_asia=hr.map(x=>+(x<7));F.sess_london=hr.map(x=>+(x>=7&&x<16));F.sess_ny=hr.map(x=>+(x>=13&&x<22));F.sess_overlap=hr.map(x=>+(x>=13&&x<16));
  return F;
}
function predictAt(F,i,P){
  let p=P.b;const contrib=[];
  for(let j=0;j<P.features.length;j++){const x=F[P.features[j]][i];if(x===undefined||isNaN(x))return null;const z=(x-P.mean[j])/P.scale[j],cj=P.coef[j]*z;p+=cj;contrib.push([P.features[j],cj,x])}
  return{pred:p,contrib};
}
if(typeof module!=="undefined")module.exports={parseCSV,computeFeatures,predictAt};
