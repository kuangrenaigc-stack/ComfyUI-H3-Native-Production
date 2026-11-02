"""Publish unchanged native MP4 files into one flat numbered server folder."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,shutil,tempfile
STATE=Path('/opt/h3/state')
VIDEOS=Path('/opt/h3/videos')
DRAFTS=Path('/opt/h3/drafts')
def atomic_json(path,data):
 fd,temp=tempfile.mkstemp(prefix='.index-',dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(data,f,ensure_ascii=False,indent=2)
 os.replace(temp,path)
def reservations():
 path=STATE/'draft-reservations.json'
 return json.loads(path.read_text()) if path.exists() else []
def numbers():
 return [int(m.group(1)) for p in VIDEOS.iterdir() if (m:=re.fullmatch(r'影片_(\d+)\.mp4',p.name))]
def publish_draft(source,final_source,label):
 source=Path(source).resolve(strict=True);final_source=str(Path(final_source).resolve())
 STATE.mkdir(parents=True,exist_ok=True);VIDEOS.mkdir(exist_ok=True);DRAFTS.mkdir(exist_ok=True)
 digest=sha(source)
 with (STATE/'publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  rows=reservations();row=next((r for r in rows if r['final_source']==final_source),None)
  if row:
   if row['sha256']!=digest:raise ValueError('A different draft already owns this job number')
  else:
   number=max(numbers()+[r['number'] for r in rows],default=0)+1
   row=dict(number=number,filename=f'草稿_{number:04d}.mp4',source=str(source),final_source=final_source,
            label=label,sha256=digest,bytes=source.stat().st_size,status='awaiting_user_approval')
   rows.append(row)
  dest=DRAFTS/row['filename']
  if dest.is_symlink():
   if dest.resolve()!=source:raise ValueError('Draft destination points to another job')
  elif dest.exists():raise ValueError('Draft destination already exists')
  else:dest.symlink_to(source)
  atomic_json(STATE/'draft-reservations.json',rows)
  return row
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024**2):h.update(b)
 return h.hexdigest()
def publish(source,label,expected=None,preferred=None):
 source=Path(source).resolve(strict=True);assert source.is_file() and source.suffix.lower()=='.mp4'
 VIDEOS.mkdir(exist_ok=True);STATE.mkdir(parents=True,exist_ok=True)
 with (STATE/'publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  index=STATE/'index.json';rows=json.loads(index.read_text()) if index.exists() else []
  reserved=reservations();reservation=next((r for r in reserved if r['final_source']==str(source)),None)
  digest=sha(source)
  if expected and digest!=expected:raise ValueError('Source SHA256 mismatch: '+str(source))
  for row in rows:
   if row['sha256']==digest and (not reservation or row['number']==reservation['number']):
    assert (VIDEOS/row['filename']).is_file();return row
  nums=numbers()+[r['number'] for r in reserved]
  if reservation:
   if preferred is not None and preferred!=reservation['number']:raise ValueError('Final number must match approved draft')
   number=reservation['number']
  else:
   if preferred is not None and any(r['number']==preferred for r in reserved):raise ValueError('Number reserved by another draft')
   number=preferred if preferred is not None else max(nums,default=0)+1
  dest=VIDEOS/f'影片_{number:04d}.mp4';assert not dest.exists()
  fd,temp=tempfile.mkstemp(prefix='.publishing-',dir=VIDEOS);os.close(fd)
  try:
   shutil.copyfile(source,temp);assert sha(Path(temp))==digest
   os.chmod(temp,0o644);os.replace(temp,dest)
  finally:Path(temp).unlink(missing_ok=True)
  row=dict(number=number,filename=dest.name,source=str(source),label=label,sha256=digest,bytes=dest.stat().st_size,native_unchanged=True)
  rows.append(row);fd,temp=tempfile.mkstemp(prefix='.index-',dir=STATE)
  with os.fdopen(fd,'w') as f:json.dump(rows,f,ensure_ascii=False,indent=2)
  os.replace(temp,index)
  if reservation:
   reservation.update(status='finalized',published_video=row)
   atomic_json(STATE/'draft-reservations.json',reserved)
  return row
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--label',default='');args=p.parse_args()
 print(json.dumps(publish(args.source,args.label),ensure_ascii=False))
