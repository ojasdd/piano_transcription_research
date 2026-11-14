import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ['PIANO_TRANSCRIBER_STUB']='1'
from piano_transcriber.transcriber import transcribe
r=transcribe('unused.wav')
print('keys:',list(r),'notes:',r['notes'],'roll shape:',len(r['frame_roll']),len(r['frame_roll'][0]))
