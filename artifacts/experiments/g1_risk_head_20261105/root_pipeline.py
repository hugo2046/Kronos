from pathlib import Path
import subprocess,sys,json,time,os,fcntl
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path('/home/user/workspace/Kronos-g1-risk-head')
out=root/'artifacts/experiments/g1_risk_head_20261105'
lock=(out/'root_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
py='/home/user/miniconda3/envs/quant/bin/python'
status=out/'root_pipeline_status.json'
def save(**kw):
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(pid=os.getpid(),at=time.time(),**kw)));os.replace(temp,status)
def run(name,arguments):
 while True:
  with (out/(name+'.log')).open('a') as log:
   child=subprocess.Popen([py,*arguments],cwd=root,stdout=log,stderr=subprocess.STDOUT)
   save(stage=name,worker_pid=child.pid);code=child.wait()
  if code==75:
   save(stage='resource_guard',resume_stage=name)
   while True:
    now=datetime.now(ZoneInfo('Asia/Shanghai'));minutes=now.hour*60+now.minute
    if not 985<=minutes<1005:break
    time.sleep(15)
   continue
  if code:save(stage='failed',failed_stage=name,exit_code=code);sys.exit(code)
  return
run('A_model',['-u','-m','g1_risk_head.run','pilot'])
run('A_parity',['-u','-m','g1_risk_head.accounts','--parity'])
run('features_development',['-u','-m','g1_risk_head.run','features'])
run('training',['-u','-m','g1_risk_head.run','train'])
gate=json.loads((out/'mechanism_gate.json').read_text())
if gate['passed']:
 run('features_evaluation',['-u','-m','g1_risk_head.run','eval_features'])
 run('accounts',['-u','-m','g1_risk_head.accounts'])
run('report',['-u','-m','g1_risk_head.report'])
save(stage='delivery_pending',mechanism_passed=gate['passed'])
