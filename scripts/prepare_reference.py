"""Transient PUBLIC COCO/MMPose fixture. Never reads or uploads the user's Drive videos."""
import base64, json, os, pathlib, subprocess, urllib.request
root = pathlib.Path('private/reference'); root.mkdir(parents=True, exist_ok=True)
# Pin the upstream revision rather than following a moving branch.
revision = '759b39c13fea6ba094afc1fa932f51dc1b11cbf9'
def fetch(name):
    url = f'https://api.github.com/repos/open-mmlab/mmpose/contents/tests/data/coco/{name}?ref={revision}'
    headers = {'Accept':'application/vnd.github+json','User-Agent':'Row-Analyzer-validation'}
    if os.environ.get('GH_TOKEN'): headers['Authorization'] = 'Bearer ' + os.environ['GH_TOKEN']
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        content = base64.b64decode(json.load(r)['content'])
    (root / name).write_bytes(content)
    return content
fetch('000000000785.jpg')
annotations = json.loads(fetch('test_coco.json'))
image = next(i for i in annotations['images'] if i['id'] == 785)
people = [a for a in annotations['annotations'] if a['image_id'] == 785 and a.get('num_keypoints',0)>0]
assert len(people) == 1
(root / 'truth.json').write_text(json.dumps({'image':image, 'annotation':people[0], 'scope':'Single public non-rowing COCO image translated across eight frames; NOT a rowing accuracy benchmark.'}))
# Translate the original still by 3 pixels/frame, preserving known labeled joint positions.
subprocess.run(['ffmpeg','-y','-loop','1','-framerate','10','-i',str(root/'000000000785.jpg'),'-vf',"pad=664:426:0:0:white,crop=640:426:x='24-3*n':y=0",'-frames:v','8','-c:v','libx264','-pix_fmt','yuv420p','-bf','0',str(root/'reference.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print('Prepared transient public reference fixture; eight frames, x offset = -24 + 3*frame_index.')
